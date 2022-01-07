import logging
import os
import pickle

import h5py
import numpy as np
import random
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


def rank_patches_cuda(pred_affs, consensus_vote_array, patchshape, neighshape,
                      overlap_mask, **kwargs):
    if kwargs.get("flip_cons_arr_axes", False):
        cuda_fn = "cuda/rankPatches6.cu"
    else:
        cuda_fn = "cuda/rankPatches.cu"
    code = loadKernelFromFile(cuda_fn, pred_affs.shape,
                              patchshape, neighshape,
                              kwargs['patch_threshold'])

    build_options = setKernelBuildOptions(step="rank", **kwargs)

    kernels = make_kernel(code, options=build_options)

    datazsize = pred_affs.shape[1]
    dataysize = pred_affs.shape[2]
    dataxsize = pred_affs.shape[3]
    outScore = alloc_zero_array((datazsize, dataysize, dataxsize),
                                np.float32)

    if kwargs.get('overlapping_inst'):
        overlap = alloc_zero_array(overlap_mask.shape, np.bool)
        overlap[:] = overlap_mask

    kernel = kernels.get_function("rankPatches")
    with kwargs['mutex']:
        fargs = [pred_affs, consensus_vote_array]
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


def loadOrComputePatchRanking(pred_affs=None,
                              consensus_vote_array=None,
                              offsets_bases_ff=None,
                              offsets_bases_fb=None,
                              overlap_mask=None,
                              all_patches=None,
                              patchshape=None,
                              neighshape=None,
                              rad=None,
                              **kwargs
                              ):
    scores_array = None
    if kwargs.get('use_score_oracle'):
        logger.info("using oracle scores")
        thresh_key = str(round(kwargs['patch_threshold'], 2)
                             ).replace('.', '_')
        def_scores_key = "volumes/{}/IOU".format(thresh_key)
        scores_key = kwargs.get('scores_key', def_scores_key)
        scores = loadFromFile(kwargs['aff_file'], key=scores_key)
        if len(scores.shape) == 2:
            scores = np.reshape(scores, (1,) + scores.shape)
        if kwargs.get("pad_with_ps", False):
            scores = np.pad(scores,
                            ((rad[0], rad[0]),
                             (rad[1], rad[1]),
                             (rad[2], rad[2])),
                            mode='constant')
        scores_array = np.array(scores)
        ranked_patches = rank_patches_by_score(all_patches, scores)
    elif kwargs.get('ranked_patches') is not None and \
            os.path.exists(kwargs['ranked_patches']):
        ranked_patches = loadFromFile(kwargs['ranked_patches'])
    elif kwargs['cuda']:
        scores = rank_patches_cuda(
            pred_affs, consensus_vote_array, patchshape, neighshape,
            overlap_mask,
            **kwargs)
        ranked_patches = rank_patches_by_score(all_patches, scores)
        scores_array = np.array(scores)
        if kwargs.get('store_scores', False):
            pred_fn = kwargs['aff_file']
            if pred_fn.endswith("zarr"):
                outfl = zarr.open(pred_fn, 'a')
            elif pred_fn.endswith("hdf"):
                outfl = h5py.File(pred_fn, 'a')
            else:
                raise RuntimeError("invalid file format")
            thresh_key = str(round(kwargs['patch_threshold'], 2)
                             ).replace('.', '_')
            scores_key = "volumes/{}/scores".format(thresh_key)
            try:
                del outfl[scores_key]
            except KeyError:
                pass
            scores_array_tmp = np.copy(scores_array)
            logger.info("scores padded %s %s", np.min(scores_array_tmp), np.max(scores_array_tmp))
            scores_array_tmp2 = \
                    scores_array_tmp[rad[0]:scores_array.shape[0]-rad[0],
                                     rad[1]:scores_array.shape[1]-rad[1],
                                     rad[2]:scores_array.shape[2]-rad[2]]
            logger.info("scores not padded %s %s", np.min(scores_array_tmp2), np.max(scores_array_tmp2))
            for idx in range(len(all_patches)):
                scores_array_tmp[tuple(all_patches[idx])] += 100
            if kwargs.get("pad_with_ps", False):
                scores_array_tmp = \
                    scores_array_tmp[rad[0]:scores_array.shape[0]-rad[0],
                                     rad[1]:scores_array.shape[1]-rad[1],
                                     rad[2]:scores_array.shape[2]-rad[2]]
            outfl.create_dataset(
                scores_key,
                data=np.squeeze(scores_array_tmp),
                compression='gzip'
            )
            if pred_fn.endswith("hdf"):
                outfl.close()
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

    if kwargs.get("shuffle_patches"):
        for i in range(10):
            logger.info("%f", ranked_patches[i][1])
        logger.info("stat %f %f %f",
                    np.mean(scores_array_tmp[scores_array_tmp != 0.0]),
                    np.std(scores_array_tmp[scores_array_tmp != 0.0]),
                    np.median(scores_array_tmp[scores_array_tmp != 0.0]))
        logger.info("hist %s",
                    np.histogram(scores_array_tmp[scores_array_tmp != 0.0],
                                 bins=40, range=(-1, 1)))
        logger.info("shuffling")
        random.shuffle(ranked_patches)
        for i in range(10):
            logger.info("%f", ranked_patches[i][1])
    return ranked_patches, scores_array
