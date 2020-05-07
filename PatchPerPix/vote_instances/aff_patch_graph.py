import logging
import os
import scipy.spatial
import networkx as nx
import numpy as np
import random

if __package__ is None or __package__ == '':
    from get_patch_sets import *
    from utilVoteInstances import *
    from cuda_code import *
else:
    from .get_patch_sets import *
    from .utilVoteInstances import *
    from .cuda_code import *


logger = logging.getLogger(__name__)

def loadAffgraph(affgraph, selected_patch_pairs):
    if affgraph.endswith(".npy"):
        graphMat = np.load(affgraph)
        computed_pairs = np.load(selected_patch_pairs)
        affgraph = setAffgraph(graphMat, computed_pairs)
    else:
        logger.error("invalid affgraph file")
        exit(-1)
    return affgraph


def setAffgraph(graphMat, computed_pairs):
    affgraph = nx.Graph()
    logger.info("len graphmat %s, num pairs %s",
                len(graphMat), computed_pairs.shape[0])
    for idx, p in enumerate(graphMat):
        if p != 0:
            affgraph.add_edge(tuple(computed_pairs[idx, :3]),
                              tuple(computed_pairs[idx, 3:6]),
                              aff=p)
    return affgraph


def computeAndStorePatchPairs(selected_patches_list, patchshape, **kwargs):
    logger.info("sorting patches")
    selected_patches_list.sort(key=lambda p: p[0][2])
    numPatches = len(selected_patches_list)
    pts = np.zeros((numPatches, 3), dtype=np.uint32)
    for idx, p in enumerate(selected_patches_list):
        pts[idx, 0] = p[0][0]
        pts[idx, 1] = p[0][1]
        pts[idx, 2] = p[0][2]
    logger.info("creating tree")
    tree = scipy.spatial.cKDTree(pts,
                                 leafsize=4)

    logger.info("query tree")
    pairs = tree.query_pairs(2*np.sum(patchshape), p=1)
    logger.info("num pairs %s", len(pairs))
    to_delete = []

    max_ps_dist = kwargs.get("max_total_patch_distance_in_ps_multiples", 2)
    for idx, p in enumerate(pairs):
        if np.any(np.abs(
            pts[p[0]].astype(np.float32) - pts[p[1]].astype(np.float32)
        ) > max_ps_dist * patchshape):
            to_delete.append(p)

    for dlt in to_delete:
        pairs.remove(dlt)

    numPairs = len(pairs)
    if kwargs['includeSinglePatchCCS']:
        numPairsTotal = numPairs + numPatches
    else:
        numPairsTotal = numPairs

    if numPairsTotal == 0:
        logger.info('Sorry, no patch pairs in sample! Returning...')
        return None

    if kwargs['cuda']:
        arr = alloc_zero_array((numPairsTotal, 6), np.uint32)
    else:
        arr = np.zeros((numPairs + numPatches, 6), dtype=np.uint32)
    logger.info("num pairs %s", len(pairs))
    for idx, p in enumerate(pairs):
        arr[idx, 0] = pts[p[0]][0]
        arr[idx, 1] = pts[p[0]][1]
        arr[idx, 2] = pts[p[0]][2]
        arr[idx, 3] = pts[p[1]][0]
        arr[idx, 4] = pts[p[1]][1]
        arr[idx, 5] = pts[p[1]][2]
    if kwargs['includeSinglePatchCCS']:
        logger.info("num pairs (incl single patch ccs) %s",
                    len(pairs) + len(selected_patches_list))
        for idx, p in enumerate(selected_patches_list):
            arr[idx + numPairs, 0] = p[0][0]
            arr[idx + numPairs, 1] = p[0][1]
            arr[idx + numPairs, 2] = p[0][2]
            arr[idx + numPairs, 3] = p[0][0]
            arr[idx + numPairs, 4] = p[0][1]
            arr[idx + numPairs, 5] = p[0][2]
    logger.info("storing pairs")
    if not kwargs['save_no_intermediates']:
        np.save(os.path.join(kwargs['result_folder'],
                             "selected_patch_pairs.npy"), arr)
        np.save(os.path.join(kwargs['result_folder'],
                             "selected_patches_list.npy"), pts)

    return arr


def computePatchGraph_cuda(labels, consensus_vote_array,
                           selected_patch_pairsIDs,
                           patchshape, neighshape, **kwargs):
    if kwargs.get("flip_cons_arr_axes", False):
        cuda_fn = "cuda/computePatchGraph6.cu"
    else:
        cuda_fn = "cuda/computePatchGraph.cu"
    code = loadKernelFromFile(cuda_fn, labels.shape,
                              patchshape, neighshape,
                              kwargs['patch_threshold'])

    build_options = setKernelBuildOptions(step="patch_graph", **kwargs)

    kernels = make_kernel(code, options=build_options)

    num_pairs = selected_patch_pairsIDs.shape[0]

    affinity_graph_mat = alloc_zero_array((num_pairs), np.float32)

    blockshape = (min(512, num_pairs), 1, 1)
    gridshape = ((num_pairs+blockshape[0]-1)//blockshape[0], 1, 1)

    kernel = kernels.get_function("computePatchGraph")

    if kwargs.get("num_parallel_samples", 1) == 1:
        # cuda scheduler seems to be having problem with many dense patches
        # split manually, might be slower for small number of or sparse patches
        bs = 1024
        num_blocks = int(np.ceil(num_pairs / bs))
        logger.info('num blocks: %s', num_blocks)
        for i in range(num_blocks):
            if i == num_blocks - 1:
                num_pairs_block = num_pairs - (i * bs)
            else:
                num_pairs_block = bs
            logger.info("%s %s %s %s %s",
                        num_pairs, i, bs, num_blocks, num_pairs_block)
            gridshape = (
                (num_pairs_block + blockshape[0] - 1) // blockshape[0], 1, 1)
            offset = np.int32(i * bs)
            with kwargs['mutex']:
                kernel(labels, consensus_vote_array, affinity_graph_mat,
                       selected_patch_pairsIDs, np.uint64(num_pairs_block),
                       offset, block=blockshape, grid=gridshape)
                logger.info("syncing")
                sync(kwargs['context'])
    else:
        # create without manually splitting in blocks
        offset = np.int32(0)
        with kwargs['mutex']:
            kernel(labels, consensus_vote_array, affinity_graph_mat,
                   selected_patch_pairsIDs, np.uint64(num_pairs),
                   offset, block=blockshape, grid=gridshape)

            sync(kwargs['context'])

    # save aff graph if required
    if kwargs.get("save_patch_graph", False) or \
       kwargs.get("termAfterPatchGraph", False):
        fn = os.path.splitext(os.path.basename(kwargs['affinities']))[0]
        np.save(os.path.join(kwargs['result_folder'],
                             fn + "_selected_patch_pairs.npy"),
                selected_patch_pairsIDs)
        np.save(os.path.join(kwargs['result_folder'],
                             fn + "_aff_graph.npy"),
                affinity_graph_mat)

    if kwargs.get('return_intermediates'):
        return affinity_graph_mat

    logger.info("affinity_graph_mat: %s %s",
                np.min(affinity_graph_mat), np.max(affinity_graph_mat))
    affgraph = setAffgraph(affinity_graph_mat, selected_patch_pairsIDs)
    return affgraph


def computePatchGraph(
        selected_patches_list,
        num_selected,
        selected_patch_pairsIDs,
        labels,
        foreground_to_cover,
        patchshape,
        neighshape,
        rad,
        multiple_worms,
        lookup,
        consensus_vote_array,
        **kwargs
):
    if kwargs['cuda']:
        return computePatchGraph_cuda(labels, consensus_vote_array,
                                      selected_patch_pairsIDs,
                                      patchshape, neighshape,
                                      **kwargs)
    affinity_graph = nx.Graph()

    selected_patch_foregrounds = [
        get_foreground_set(rp[0], labels, foreground_to_cover,
                           patchshape, rad, kwargs['patch_threshold'],
                           sample=kwargs['sample'])
        for rp in selected_patches_list]

    debugidx = [[20, 15], [21, 15], [24, 15], [37, 54], [51, 53]]
    for r1, ranked_patch1 in enumerate(selected_patches_list):
        if r1 % 10 == 0:
            logger.info("compute patch graph: selected patch %s of %s",
                        r1, num_selected)
        idx1 = ranked_patch1[0]
        found_edge = False

        if kwargs['includeSinglePatchCCS']:
            # heads up: including single-patch ccs
            rnge = enumerate(selected_patches_list[r1:], r1)
        else:
            # heads up: not including single-patch ccs
            rnge = enumerate(selected_patches_list[r1 + 1:], r1 + 1)
        for r2, ranked_patch2 in rnge:
            pf1 = selected_patch_foregrounds[r1]
            idx2 = ranked_patch2[0]

            if multiple_worms[tuple(idx1)] and multiple_worms[tuple(idx2)]:
                continue

            offset = idx2 - idx1
            # if np.any(np.abs(offset) > 4*np.array(patchshape)/5):
            if np.any(np.abs(offset) > np.array(patchshape)):
                continue
            pf2 = selected_patch_foregrounds[r2]

            if kwargs['removeIntersection']:
                # intersection is known has very good consensus
                # because each individual patch does, so remove:
                intersection = pf1 & pf2
                # don't remove all of it because this may leave very close
                # pixels unconnected:
                intersection -= set(random.sample(intersection,
                                                     min(len(intersection), 5)))
                pf1 = pf1 - intersection
                pf2 = pf2 - intersection

            if len(pf1) < 1 or len(pf2) < 1:
                continue
            pf1 = [np.array(p) for p in pf1]
            pf2 = [np.array(p) for p in pf2]

            pf1pf2 = np.transpose([
                np.concatenate([p, q - p + patchshape - 1])
                for p in pf1
                for q in pf2
                if np.all(np.abs(p - q) < patchshape) and np.any(p - q != 0)])
            if len(pf1pf2) > 0:
                offsets_bases = lookup[tuple(pf1pf2)]
                offsets_bases = tuple(np.transpose(offsets_bases))

                weight = np.sum(consensus_vote_array[offsets_bases])

                affinity_graph.add_edge(tuple(idx1), tuple(idx2), aff=weight)

                # debug:
                if np.any([np.all(idx1[1:] == didx) for didx in debugidx]) and \
                        np.any([np.all(idx2[1:] == didx) for didx in debugidx]):
                    logger.info("edge between idx1, idx2; len1, len2 %s %s %s %s",
                                idx1, idx2, len(pf1), len(pf2))
                    logger.info("pf1 %s", pf1)
                    logger.info("pf2 %s", pf2)
                    logger.info("votes %s", consensus_vote_array[offsets_bases])

    return affinity_graph
