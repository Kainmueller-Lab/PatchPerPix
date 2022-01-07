import argparse
import glob
import logging
import os
import sys

import h5py
import numpy as np
from scipy import ndimage
from skimage import io
from skimage.morphology import skeletonize_3d

# import pycuda
# import pycuda.compiler
# from pycuda.autoinit import context


if __package__ is None or __package__ == '':
    from consensus_array import loadOrComputeConsensus
    from ranked_patches import loadOrComputePatchRanking
    from utilVoteInstances import getResKey, loadAffinities, fillLookup
    from foreground_cover import *
    from graph_to_labeling import affGraphToInstancesT, affGraphToInstances
    from aff_patch_graph import *
    from isbi_hacks import sparsifyPatches, filterInstanceBoundariesFromFG
    from cuda_code import *
else:
    from .consensus_array import loadOrComputeConsensus
    from .ranked_patches import loadOrComputePatchRanking
    from .utilVoteInstances import getResKey, loadAffinities, fillLookup
    from .foreground_cover import *
    from .graph_to_labeling import affGraphToInstancesT, affGraphToInstances
    from .aff_patch_graph import *
    from .isbi_hacks import sparsifyPatches, filterInstanceBoundariesFromFG
    from .cuda_code import *

np.seterr(over='raise')

logger = logging.getLogger(__name__)    # todo: make use of logger


def replace(array, old_values, new_values):
    values_map = np.arange(int(array.max() + 1), dtype=new_values.dtype)
    values_map[old_values] = new_values

    return values_map[array]


def merge_dicts(sink, source):
    if not isinstance(sink, dict) or not isinstance(source, dict):
        raise TypeError('Args to merge_dicts should be dicts')

    for k, v in source.items():
        if isinstance(source[k], dict) and isinstance(sink.get(k), dict):
            sink[k] = merge_dicts(sink[k], v)
        else:
            sink[k] = v

    return sink


def get_arguments(check_required=True):
    parser = argparse.ArgumentParser()

    # standard input / output paths
    parser.add_argument('--affinities', type=str, help='affinities',
                        required=False)
    parser.add_argument('--affinities_key', type=str, help='hdf key',
                        required=False, default="images/pred_affs")
    parser.add_argument('--basedir', type=str, help='basedir',
                        required=False)
    parser.add_argument('--mode', type=str, help='mode', required=False)
    parser.add_argument('--result_folder', type=str,
                        help='folder to save result segmentations as hdf5',
                        required=check_required)
    parser.add_argument('--checkpoint', type=str, help='checkpoint',
                        required=False)
    parser.add_argument("--debug", help="",
                        action="store_true")

    # PatchPerPix-specific parameters
    parser.add_argument('--patch_threshold', type=float, default=0.9)
    parser.add_argument('--fc_threshold', type=float, default=0.5)
    parser.add_argument('-p', '--patchshape', type=int,
                        action='append', required=check_required)

    # intermediate results: consensus, scores, ranked_patches, ...
    parser.add_argument('--consensus', type=str,
                        help='path to consensus array')
    parser.add_argument('--consensus_key', type=str, help='hdf key',
                        required=False, default="images/consensus")
    parser.add_argument('--scores', type=str,
                        help='path to scores array')
    parser.add_argument('--ranked_patches', type=str,
                        help='path to ranked_patches')
    # parser.add_argument('--scores_key', type=str, help='hdf key',
    #                     required=False, default="images/scores")
    parser.add_argument('--aff_graph', type=str, help='path to affgraph',
                        required=False)
    parser.add_argument('--selected_patches', type=str,
                        help='selected_patches',
                        required=False)
    parser.add_argument('--selected_patch_pairs', type=str,
                        help='selected_patch_pairs',
                        required=False)
    parser.add_argument('--select_patches_for_sparse_data', help="",
                        action="store_true")

    # control flags
    parser.add_argument("--cuda", help="",
                        action="store_true")
    parser.add_argument("--skipLookup", help="",
                        action="store_true")
    parser.add_argument("--skipThinCover", help="",
                        action="store_true")
    parser.add_argument("--skipRanking", help="",
                        action="store_true")
    parser.add_argument("--skipConsensus", help="",
                        action="store_true")
    parser.add_argument("--termAfterThinCover", help="",
                        action="store_true")
    parser.add_argument("--graphToInst", help="",
                        action="store_true")
    parser.add_argument("--mws", help="",
                        action="store_true")
    parser.add_argument("--includeSinglePatchCCS", help="",
                        action="store_true")
    parser.add_argument("--removeIntersection", help="",
                        action="store_true")
    parser.add_argument("--isbiHack", help="",
                        action="store_true")
    parser.add_argument("--mask_fg_border", help="",
                        action="store_true")
    parser.add_argument("--parallel", help="",
                        action="store_true")
    parser.add_argument("--save_no_intermediates", help="",
                        action="store_true")
    parser.add_argument('--crop_x_s', type=int, default=0)
    parser.add_argument('--crop_x_e', type=int, default=None)
    parser.add_argument('--crop_y_s', type=int, default=0)
    parser.add_argument('--crop_y_e', type=int, default=None)
    parser.add_argument('--crop_z_s', type=int, default=0)
    parser.add_argument('--crop_z_e', type=int, default=None)

    args, unknown = parser.parse_known_args()

    return args


def to_instance_seg(
        pred_affs,
        foreground,
        mask_to_cover,
        numinst,
        patchshape,
        **kwargs
):
    logger.info("pred_affs shape: %s", pred_affs.shape)
    logger.info("numinst shape: %s", numinst.shape)
    logger.info("foreground shape: %s", foreground.shape)
    logger.info("mask shape: %s", mask_to_cover.shape)

    rad = np.array([p // 2 for p in patchshape])
    mid = int(np.prod(patchshape) / 2)

    if kwargs.get("pad_with_ps", False):
        assert not kwargs['blockwise'], "can only pad whole volumes"

        pred_affs = np.pad(pred_affs,
                        ((0, 0),
                         (rad[0], rad[0]),
                         (rad[1], rad[1]),
                         (rad[2], rad[2])),
                        mode='constant')
        foreground = np.pad(foreground,
                               ((rad[0], rad[0]),
                                (rad[1], rad[1]),
                                (rad[2], rad[2])),
                               mode='constant')
        mask_to_cover = np.pad(mask_to_cover,
                               ((rad[0], rad[0]),
                                (rad[1], rad[1]),
                                (rad[2], rad[2])),
                               mode='constant')
        numinst = np.pad(numinst,
                         ((rad[0], rad[0]),
                          (rad[1], rad[1]),
                          (rad[2], rad[2])),
                         mode='constant')
        logger.info("shape after padding %s", pred_affs.shape)


    if kwargs['cuda']:
        if pred_affs.dtype != np.float32:
            pred_affs = pred_affs.astype(np.float32)
        tmp = alloc_zero_array(pred_affs.shape, np.float32)
        # tmp = pycuda.driver.managed_zeros_like(
        #     pred_affs, mem_flags=pycuda.driver.mem_attach_flags.GLOBAL)
        tmp[:] = pred_affs
        pred_affs = tmp

    if not kwargs['debug']:
        debug_output1 = None
        debug_output2 = None

    radslice = tuple([slice(rad[i], foreground.shape[i] - rad[i])
                      for i in range(len(rad))])
    logger.info("input image shape: {}".format(foreground.shape))
    logger.info("Number fg pixel: {}".format(np.count_nonzero(foreground)))

    overlap_mask = 1 * (numinst > 1)
    logger.info("overlap mask: %s %s", overlap_mask.shape,
                np.count_nonzero(overlap_mask))

    # isbi2012: hack for one-slice:
    if kwargs['isbiHack'] and (foreground.shape[0] > 1):
        foreground[0] = 0
        foreground[2:] = 0

    if kwargs.get('blockwise', False) == False:
        if kwargs.get('skeletonize_foreground'):
            mask_to_cover = skeletonize_3d(mask_to_cover) > 0
            logger.info("Number fg pixel after skeletonization: {}".format(
                np.count_nonzero(mask_to_cover)))

    mask_to_cover[overlap_mask > 0] = 0
    logger.info("Number overlapping pixel: %s", np.count_nonzero(overlap_mask))
    logger.info("Number pixel mask_to_cover: %s", np.count_nonzero(mask_to_cover))

    instances = (0 * foreground).astype(np.uint16)

    if np.count_nonzero(mask_to_cover[radslice]) == 0:
        logger.info("no fg found, returning...")
        if kwargs.get('return_intermediates', False):
            return None, None
        else:
            return instances.astype(np.uint16), foreground.astype(np.uint8)

    # [patchshape[0],2*patchshape[1],2*patchshape[2]]
    # TODO: check
    neighshape = patchshape.copy()
    if neighshape[0] > 1:
        neighshape *= 2
    else:
        neighshape[1:] *= 2

    # (0) short cut
    # load previously computed graph and compute labeling
    # e.g. graph computed externally using cuda
    if kwargs['graphToInst']:
        fn = os.path.splitext(os.path.basename(kwargs['affinities']))[0]
        kwargs['affgraph'] = os.path.join(kwargs['result_folder'],
                                          fn + "_aff_graph.npy")
        kwargs['selected_patch_pairs'] = os.path.join(
            kwargs['result_folder'],
            fn + "_selected_patch_pairs.npy")
        return affGraphToInstancesT(pred_affs, patchshape, rad,
                                    debug_output1, debug_output2,
                                    instances, foreground,
                                    **kwargs)

    if kwargs['debug']:
        debug_output1 = np.zeros(tuple(np.array(patchshape) * instances.shape),
                                 dtype=np.float32)
        debug_output2 = np.zeros(tuple(np.array(patchshape) * instances.shape),
                                 dtype=np.float32)

    all_patches = np.transpose(np.where(foreground))
    logger.info("num foreground pixels: %s", len(all_patches))
    logger.info("num foreground pixels: %s (without border)",
                np.count_nonzero(foreground[radslice]))

    if not kwargs['cuda'] and not kwargs['skipLookup']:
        lookup = fillLookup(foreground, patchshape, neighshape, all_patches)
    else:
        lookup = None

    all_patches = [p for p in all_patches
                   if np.all(p >= rad) and np.all(p < foreground.shape - rad)]
    logger.info("num foreground pixels excluding boundary region: %s",
                len(all_patches))

    if len(all_patches) == 0:
        logger.info("no patches found, returning...")
        if kwargs.get('return_intermediates', False):
            return None, None
        else:
            return instances.astype(np.uint16), foreground.astype(np.uint8)

    # isbi2012:
    if kwargs['isbiHack']:
        all_patches = sparsifyPatches(all_patches)

    # (1) for each pair of foreground pixels within reach,
    # count: positive votes, negative votes, total votes
    # mws on consensus votes?? --> same problems as before?
    if not kwargs['skipConsensus']:
        consensus_vote_array, offsets_bases_ff, offsets_bases_fb = \
            loadOrComputeConsensus(instances, patchshape,
                                   neighshape, all_patches, pred_affs, rad,
                                   foreground, lookup,
                                   overlap_mask, **kwargs)
    else:
        consensus_vote_array = None
        offsets_bases_ff = None
        offsets_bases_fb = None

    if kwargs.get('save_consensus', False):
        return None, None

    # (2) get back to instance patch selection, rate as a whole:
    # for each pair in foreground, is patch consistent with "consensus vote"?
    # --> count consistent vs inconsistent
    # (weighted by strength of consensus?)
    # --> ranked list of patches
    if not kwargs['skipRanking']:
        ranked_patches_list, scores_array = \
            loadOrComputePatchRanking(
                pred_affs=pred_affs,
                consensus_vote_array=consensus_vote_array,
                offsets_bases_ff=offsets_bases_ff,
                offsets_bases_fb=offsets_bases_fb,
                overlap_mask=overlap_mask,
                all_patches=all_patches,
                patchshape=patchshape,
                neighshape=neighshape,
                rad=rad,
                **kwargs,
            )
        logger.info("num ranked patches %s ", len(ranked_patches_list))

    # debug_output output for *all* patches:
    if kwargs['debug']:
        it = np.nditer(foreground, flags=['multi_index'])
        while not it.finished:
            idx = np.array(it.multi_index)
            it.iternext()
            # if not foreground[tuple(idx)]: continue
            start = idx - rad
            stop = idx + rad + 1
            if np.any(start < 0) or np.any(stop > foreground.shape):
                continue
            affslice = tuple([slice(0, pred_affs.shape[0])] +
                               [idx[i] for i in range(len(idx))])

            patch = pred_affs[affslice]
            patch = np.reshape(patch, patchshape)
            debugstart = np.array(idx) * patchshape
            debugstop = debugstart + patchshape
            debugslice = tuple([slice(debugstart[i], debugstop[i])
                                for i in range(len(debugstart))])
            debug_output1[debugslice] = patch

    # isbi2012: patch-based foreground excluding boundaries:
    if kwargs['isbiHack'] and (foreground.shape[0] > 1):
        mask_to_cover = filterInstanceBoundariesFromFG(
            pred_affs, foreground, all_patches,
            patchshape, rad, mask_to_cover, **kwargs)

    if kwargs.get('aff_graph') is None:
        if kwargs.get('selected_patches') is not None:
            # already set by e.g. stitch_patch_graph
            selected_patches_list = list(kwargs.get('selected_patches'))
            selected_patches_list = [(coord, 1.0) for coord in
                                     selected_patches_list]
            logger.info("%s", selected_patches_list)
            num_selected = len(selected_patches_list)
        # (3) pick patches from ranked list until foreground is covered
        elif kwargs.get('skipSelection', False):
            selected_patches_list = ranked_patches_list
            num_selected = len(all_patches)
        else:
            logger.info("compute foreground cover")
            selected_patches_list, num_selected = \
                computeForegroundCover(
                    overlap_mask, mask_to_cover,
                    patchshape, ranked_patches_list,
                    radslice,
                    pred_affs, rad, debug_output1, scores_array, **kwargs)
            logger.info("done compute foreground cover")

        # (4) thin out selection with greedy set cover algorithm:
        if not kwargs['skipThinCover'] and num_selected > 0:
            logger.info("compute thin out foreground cover")
            selected_patches_list, num_selected = \
                thinOutForegroundCover(mask_to_cover,
                                       selected_patches_list,
                                       radslice, pred_affs,
                                       rad, patchshape, **kwargs)
            logger.info("done thin out foreground cover")

        if kwargs.get('selected_patch_pairs') is not None:
            # already set by e.g. stitch_patch_graph
            selected_patch_pairsIDs = np.array(kwargs.get(
                'selected_patch_pairs'), dtype=np.uint32)
            tmp = alloc_zero_array((len(selected_patch_pairsIDs), 6), np.uint32)
            tmp[:] = selected_patch_pairsIDs
            selected_patch_pairsIDs = tmp
        else:
            selected_patch_pairsIDs = \
                computeAndStorePatchPairs(selected_patches_list, patchshape,
                                          **kwargs)

        # no pairs selected after pruning
        if selected_patch_pairsIDs is None:
            if kwargs.get('return_intermediates', False):
                return None, None
            else:
                return instances.astype(np.uint16), foreground.astype(np.uint8)

        if kwargs['termAfterThinCover']:
            exit(0)

        # (5) compute graph over patches
        logger.info("compute patch graph")
        affinity_graph = computePatchGraph(
                selected_patches_list,
                num_selected,
                selected_patch_pairsIDs,
                pred_affs, mask_to_cover,
                patchshape, neighshape,
                rad, overlap_mask, lookup,
                consensus_vote_array, **kwargs
            )
        logger.info("done compute patch graph")

        if kwargs.get('return_intermediates'):
            assert kwargs.get('cuda'), \
                "only works with cuda, otherwise graph is built directly"
            # if flag is set, matrix form is returned
            affinity_graph_mat = affinity_graph
            return selected_patch_pairsIDs, affinity_graph_mat
        if kwargs.get('termAfterPatchGraph', False):
            return None, None
    else:
        affinity_graph = loadAffgraph(
            kwargs['aff_graph'],
            kwargs['selected_patch_pairs']
        )

    # (6) label pixels according to graph connected components
    return affGraphToInstances(affinity_graph, pred_affs, patchshape, rad,
                               debug_output1, debug_output2,
                               instances, foreground, **kwargs)


def do_block(
        block,
        foreground,
        mask,
        numinst,
        **kwargs
):
    patchshape = kwargs['patchshape']
    del kwargs['patchshape']
    if type(patchshape) != np.ndarray:
        patchshape = np.array(patchshape)

    res = to_instance_seg(
        block,
        foreground,
        mask,
        numinst,
        patchshape,
        **kwargs
    )
    if kwargs.get('return_intermediates'):
        return res
    else:
        instances, _ = res
        rad = np.array([p // 2 for p in patchshape])
        slices = [slice(r, d - r) for r, d in zip(rad, instances.shape)]
        instances = instances[slices]

        return instances


def do_all(
        aff_file,
        patchshape=np.array([1, 25, 25]),
        **kwargs
):
    logger.info("processing %s into %s",
                aff_file, kwargs['result_folder'])

    if type(patchshape) is not np.ndarray:
        patchshape = np.array(patchshape)

    add_suffix = kwargs.get('add_suffix', False)
    if add_suffix:
        res_ext = getResKey(**kwargs)
    else:
        res_ext = ''

    affinities, numinst, foreground = \
        loadAffinities(aff_file, res_ext,
                       patchshape=patchshape,
                       **kwargs)
    mask = np.copy(foreground)
    if numinst is None:
        numinst = np.copy(foreground)

    kwargs['aff_file'] = aff_file
    res = to_instance_seg(
        affinities,
        foreground,
        mask,
        numinst,
        patchshape,
        **kwargs
    )
    res_key = kwargs.get('res_key', 'vote_instances')
    if kwargs['debug']:
        instances, foreground, debug, debug2 = res
        foreground = foreground.astype(np.uint8)
        results = [instances, foreground, debug, debug2]
        result_names = [res_key, 'vote_foreground',
                        'vote_debug', 'vote_debug2']
    else:
        instances, foreground = res
        if instances is None and foreground is None:
            return
        foreground = foreground.astype(np.uint8)
        results = [instances, foreground]
        result_names = [res_key, 'vote_foreground']

    if kwargs.get('crop_to_foreground', False):
        if kwargs.get('one_instance_per_channel', False):
            for i in range(instances.shape[0]):
                instances[i][foreground == 0] = 0
        else:
            instances[foreground == 0] = 0

    fn = os.path.splitext(os.path.basename(aff_file))[0]
    with h5py.File(os.path.join(kwargs['result_folder'],
                                fn + ".hdf"), 'w') as f2:
        for i, dataset_basename in enumerate(result_names):
            key = dataset_basename + res_ext
            dataset = key
            result = results[i]
            f2.create_dataset(
                dataset,
                data=result,
                compression='gzip')
            f2[dataset].attrs['offset'] = (0, 0, 0)
            f2[dataset].attrs['resolution'] = (1, 1, 1)


def main(**kwargs):
    # todo: unique identifier for intermediate results
    if 'check_required' in kwargs:
        args = get_arguments(check_required=kwargs['check_required'])
    else:
        args = get_arguments()

    # convert command line args to dict and overwrite with kwargs if given
    args = vars(args)

    if len(kwargs) > 0:
        args = merge_dicts(args, kwargs)

    if 'check_required' in kwargs:
        assert type(args['patchshape']) in [np.ndarray, tuple, list], \
            "Please check type of patchshape {}".format(
                type(args['patchshape']))
        assert type(args['result_folder']) == str, \
            "Please check type of result_folder {}".format(
                type(args['result_folder']))

    if args.get('cuda') and not args.get('graphToInst', False):
        args['context'] = init_cuda()
    os.makedirs(args['result_folder'], exist_ok=True)

    affinities = args['affinities']
    if affinities is not None:
        if affinities.endswith(".zarr") or os.path.isfile(affinities):
            do_all(affinities, **args)
            return
        elif os.path.isdir(affinities):
            aff_files = glob.glob(os.path.join(affinities, "*.hdf"))
        else:
            raise RuntimeError("affinities (%s) should be file or dir",
                               affinities)
    else:
        if args['mode'] is not None and args['checkpoint'] is not None:
            aff_files = glob.glob(os.path.join(args['basedir'], args['mode'],
                                               "processed", args['checkpoint'],
                                               "*.hdf"))

    if args['parallel']:
        raise NotImplementedError
    else:
        for fl in aff_files:
            do_all(fl, **args)

    delete_cuda(args['context'])


if __name__ == "__main__":
    main()
