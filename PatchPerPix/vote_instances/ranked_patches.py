import logging
import os
import pickle

import h5py
import numpy as np
import re
import zlib

if __package__ is None or __package__ == '':
    from utilVoteInstances import *
    from cuda_code import *
else:
    from .utilVoteInstances import *
    from .cuda_code import *


logger = logging.getLogger(__name__)

def rank_patches_by_score(all_patches_idx, rank_scores):
    logger.info("sort patches by score")

    ranked_patches = []
    for idx in range(len(all_patches_idx)):
        ranked_patches.append((all_patches_idx[idx],
                               rank_scores[tuple(all_patches_idx[idx])]))
    ranked_patches = sorted(ranked_patches, key=lambda x: x[1], reverse=True)
    logger.info("done compute patch ranking.")
    return ranked_patches


def rank_patches_cuda(labels, consensus_vote_array, patchshape, neighshape,
                      overlap_mask, **kwargs):
    if kwargs.get("flip_cons_arr_axes", False):
        cuda_fn = "cuda/rankPatches6.cu"
    else:
        cuda_fn = "cuda/rankPatches.cu"
    code = loadKernelFromFile(cuda_fn, labels.shape,
                              patchshape, neighshape,
                              kwargs['patch_threshold'])

    build_options = setKernelBuildOptions(step="rank", **kwargs)

    kernels = make_kernel(code, options=build_options)

    datazsize = labels.shape[1]
    dataysize = labels.shape[2]
    dataxsize = labels.shape[3]
    outScore = alloc_zero_array((datazsize, dataysize, dataxsize),
                                np.float32)

    if kwargs.get('overlapping_inst'):
        overlap = alloc_zero_array(overlap_mask.shape, np.bool)
        overlap[:] = overlap_mask

    kernel = kernels.get_function("rankPatches")
    with kwargs['mutex']:
        fargs = [labels, consensus_vote_array]
        if kwargs.get('overlapping_inst'):
            fargs.append(overlap)
        fargs.append(outScore)
        kernel(*fargs,
               block=get_block_shape(datazsize, dataysize, dataxsize),
               grid=get_grid_shape(datazsize, dataysize, dataxsize))

        sync(kwargs['context'])

    if kwargs.get('overlapping_inst'):
        overlap.base.free()

    logger.info("scores patches min/max %s, %s",
                np.min(outScore), np.max(outScore))
    return outScore

def rank_patches(offsets_bases_ff, offsets_bases_fb, all_patches_idx,
                 consensus_vote_array):
    logger.info("compute patch ranking")

    ranked_patches = []

    numi = len(offsets_bases_ff)
    print(numi, "=", len(all_patches_idx), "=", len(offsets_bases_fb))

    for iobff, compressed_obff in enumerate(offsets_bases_ff):
        if iobff % 100 == 0:
            print("compute patch ranking at {} of {}".format(iobff, numi))
        patch_score = 0
        obff = pickle.loads(zlib.decompress(compressed_obff))
        obff = tuple(obff)
        if len(obff) > 0:
            patch_score += np.sum(consensus_vote_array[obff] > 0)
            patch_score -= np.sum(consensus_vote_array[obff] <= 0)

        compressed_obfb = offsets_bases_fb[iobff]
        obfb = pickle.loads(zlib.decompress(compressed_obfb))
        obfb = tuple(obfb)
        if len(obfb) > 0:
            patch_score += np.sum(consensus_vote_array[obfb] < 0)
            patch_score -= np.sum(consensus_vote_array[obfb] >= 0)

        ranked_patches.append((all_patches_idx[iobff], patch_score))
    ranked_patches = sorted(ranked_patches, key=lambda x: x[1], reverse=True)
    print("done compute patch ranking.")
    return ranked_patches


def loadOrComputePatchRanking(labels=None,
                              consensus_vote_array=None,
                              offsets_bases_ff=None,
                              offsets_bases_fb=None,
                              overlap_mask=None,
                              all_patches=None,
                              patchshape=None,
                              neighshape=None,
                              **kwargs
                              ):
    scores_array = None
    if kwargs.get('scores') is not None and \
       os.path.exists(kwargs['scores']):
        scores = loadFromFile(kwargs['scores'], key=kwargs['scores_key'])
        ranked_patches = rank_patches_by_score(all_patches, scores)
    elif kwargs.get('ranked_patches') is not None and \
            os.path.exists(kwargs['ranked_patches']):
        ranked_patches = loadFromFile(kwargs['ranked_patches'])
    elif kwargs['cuda']:
        scores = rank_patches_cuda(
            labels, consensus_vote_array, patchshape, neighshape,
            overlap_mask,
            **kwargs)
        ranked_patches = rank_patches_by_score(all_patches, scores)
        scores_array = np.array(scores)
        if kwargs.get('store_selected_hdf', False):
            with h5py.File("scores_test.hdf", 'w') as f:
                f.create_dataset(
                    "scores",
                    data=scores,
                    compression='gzip'
                )
        scores.base.free()
    else:
        print("1st patch idx: ", all_patches[0])
        ranked_patches = rank_patches(offsets_bases_ff, offsets_bases_fb,
                                      all_patches, consensus_vote_array)
        if not kwargs['save_no_intermediates']:
            fn = os.path.join(kwargs['result_folder'], "ranking.pickle")
            print("writing {}".format(fn))
            f = open(fn, 'wb')
            pickle.dump(ranked_patches, f, protocol=4)
        del offsets_bases_ff[:]
        del offsets_bases_fb[:]

    logger.info("best/worst score: %s %s",
                ranked_patches[0][1], ranked_patches[-1][1])
    return ranked_patches, scores_array
