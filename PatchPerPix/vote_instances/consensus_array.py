import logging
import os
import numpy as np
import zlib
import pickle
import re

if __package__ is None or __package__ == '':
    from utilVoteInstances import *
    from cuda_code import *
else:
    from .utilVoteInstances import *
    from .cuda_code import *


logger = logging.getLogger(__name__)

def create_consensus_array(all_patches_fg, all_patches_bg,
                           shape, patchshape, neighshape, lookup):
    logger.info("creating consensus vote graph array")
    neighsize = np.prod(neighshape)
    consensus_vote_array = np.zeros(shape=(neighsize,) + tuple(shape),
                                    dtype=np.int16)

    numFGPatches = len(all_patches_fg)
    offsets_bases_ff = []
    offsets_bases_fb = []

    for ipf, pf in enumerate(all_patches_fg):
        pf = [np.array(idx) for idx in pf]
        pb = [np.array(idx) for idx in all_patches_bg[ipf]]

        if ipf % 100 == 0:
            logger.info("create_consensus_array at %s of %s",
                        ipf, numFGPatches)

        if len(pf) > 1:
            offsets_bases = np.transpose(np.concatenate([
                lookup[tuple(idx1) +
                       tuple(np.transpose(pf[i1 + 1:] - idx1 + patchshape - 1))]
                for i1, idx1 in enumerate(pf[:-1])]))
            if len(offsets_bases):
                consensus_vote_array[tuple(offsets_bases)] += 1
        else:
            offsets_bases = np.array([])

        if not type(offsets_bases) is np.ndarray:
            print(ipf, offsets_bases)
        compressed_offsets_bases = zlib.compress(pickle.dumps(offsets_bases))

        offsets_bases_ff.append(compressed_offsets_bases)

        if len(pf) > 0 and len(pb) > 0:
            offsets_bases = np.transpose(np.concatenate([
                lookup[tuple(idx1) +
                       tuple(np.transpose(pb - idx1 + patchshape - 1))]
                for idx1 in pf]))
            if len(offsets_bases):
                consensus_vote_array[tuple(offsets_bases)] -= 1
        else:
            offsets_bases = np.array([])

        compressed_offsets_bases = zlib.compress(pickle.dumps(offsets_bases))

        offsets_bases_fb.append(compressed_offsets_bases)

    logger.info("done creating consensus vote graph array")
    return consensus_vote_array, offsets_bases_ff, offsets_bases_fb


def create_consensus_array_cuda(labels, foreground, overlap_mask,
                                patchshape, neighshape,
                                **kwargs):
    # copy foreground mask to gpu
    # mask = foreground.astype(np.bool)
    # tmp_mask = alloc_zero_array(mask.shape, np.bool)
    # tmp_mask[:] = mask
    # mask = tmp_mask

    if kwargs.get("flip_cons_arr_axes", False):
        logger.info("flipping cuda array axes")
        cuda_fn = "cuda/fillConsensusArray6.cu"
    else:
        cuda_fn = "cuda/fillConsensusArray.cu"
    code = loadKernelFromFile(cuda_fn, labels.shape,
                              patchshape, neighshape,
                              kwargs['patch_threshold'])

    build_options = setKernelBuildOptions(step="consensus", **kwargs)
    if kwargs.get("consensus_interleaved_cnt", True):
        logger.info("compute affs and cntr at the same time")
        build_options.append("-DOUTPUT_BOTH")
    else:
        logger.info("compute affs and cntr separately")

    kernels = make_kernel(code, options=build_options)

    datazsize = labels.shape[1]
    dataysize = labels.shape[2]
    dataxsize = labels.shape[3]
    nsz = int(neighshape[0])
    nsy = int(neighshape[1])
    nsx = int(neighshape[2])

    if kwargs.get("flip_cons_arr_axes", False):
        outCons = alloc_zero_array((datazsize, dataysize, dataxsize,
                                    nsz, nsy, nsx),
                                   np.float32)
    else:
        outCons = alloc_zero_array((nsz, nsy, nsx,
                                    datazsize, dataysize, dataxsize),
                                   np.float32)

    logger.info("consensus array shape %s", outCons.shape)

    if kwargs.get("consensus_norm_aff", True):
        if kwargs.get("flip_cons_arr_axes", False):
            outConsCnt = alloc_zero_array((datazsize, dataysize, dataxsize,
                                           nsz, nsy, nsx),
                                          np.float32)
        else:
            outConsCnt = alloc_zero_array((nsz, nsy, nsx,
                                           datazsize, dataysize, dataxsize),
                                          np.float32)

    if kwargs.get('overlapping_inst'):
        overlap = alloc_zero_array(overlap_mask.shape, np.bool)
        overlap[:] = overlap_mask

    kernel = kernels.get_function("fillConsensusArray_allPatches")
    with kwargs['mutex']:
        logger.info("creating consensus array %s", kwargs.get('affinities'))
        fargs = [labels]
        if kwargs.get('overlapping_inst'):
            fargs.append(overlap)
        fargs.append(outCons)
        if kwargs.get("consensus_interleaved_cnt", True):
            assert kwargs.get("consensus_norm_aff", True), \
                "consensus aff not normalized so no computation required"
            fargs.append(outConsCnt)
        kernel(*fargs,
               block=get_block_shape(datazsize, dataysize, dataxsize),
               grid=get_grid_shape(datazsize, dataysize, dataxsize))
        sync(kwargs['context'])
        logger.info("done")

    if logger.isEnabledFor(logging.DEBUG):
        logger.info("consensus_vote_array sum/min/max %s %s %s",
                    "too slow", np.min(outCons),
                    np.max(outCons))

    if not kwargs.get("consensus_interleaved_cnt", True) and \
       kwargs.get("consensus_norm_aff", True):
        build_options = setKernelBuildOptions(step="consensus", **kwargs)
        build_options.append('-DOUTPUT_CNT')

        kernels = make_kernel(code, options=build_options)

        kernel = kernels.get_function("fillConsensusArray_allPatches")
        with kwargs['mutex']:
            logger.info("creating consensus array cnt %s", kwargs.get('affinities'))
            fargs = [labels]
            if kwargs.get('overlapping_inst'):
                fargs.append(overlap)
            fargs.append(outConsCnt)
            kernel(*fargs,
                   block=get_block_shape(datazsize, dataysize, dataxsize),
                   grid=get_grid_shape(datazsize, dataysize, dataxsize))
            sync(kwargs['context'])
            logger.info("done")

        if logger.isEnabledFor(logging.DEBUG):
            logger.info("consensus_vote_array cnt sum/min/max %s %s %s",
                        "too slow", np.min(outConsCnt),
                        np.max(outConsCnt))

    if kwargs.get("consensus_norm_aff", True):
        if kwargs.get("flip_cons_arr_axes", False):
            cuda_fn = "cuda/normConsensusArray6.cu"
        else:
            cuda_fn = "cuda/normConsensusArray.cu"
        code = loadKernelFromFile(cuda_fn, labels.shape,
                                  patchshape, neighshape,
                                  kwargs['patch_threshold'])

        kernels = make_kernel(code, options=None)
        kernel = kernels.get_function("normConsensusArray")
        with kwargs['mutex']:
            logger.info("normalizing consensus array %s", kwargs.get('affinities'))
            kernel(labels, outCons, outConsCnt,
                   block=get_block_shape(datazsize, dataysize, dataxsize),
                   grid=get_grid_shape(datazsize, dataysize, dataxsize))
            sync(kwargs['context'])
            logger.info("done")

        logger.info("consensus_vote_array after norm sum/min/max %s %s %s",
                    "too slow", np.min(outCons),
                    np.max(outCons))
        outConsCnt.base.free()
    else:
        logger.info("consensus affs not normalized")

    if kwargs.get('overlapping_inst'):
        overlap.base.free()

    if kwargs.get("save_consensus", False):
        fn = os.path.splitext(os.path.basename(kwargs['affinities']))[0]
        logger.info("saving consensus %s", fn)
        np.save(os.path.join(kwargs['result_folder'],
                             fn + "_consensus.npy"), outCons)

    return outCons


def loadOrComputeConsensus(instances, patchshape, neighshape,
                           all_patches, labels, rad, foreground, lookup,
                           overlap_mask, **kwargs
                           ):
    path_to_consensus = kwargs.get('consensus', None)
    if path_to_consensus is not None and os.path.exists(path_to_consensus):
        consensus_vote_array, offsets_bases_ff, offsets_bases_fb = \
            loadFromFile(path_to_consensus,
                         shape=tuple(patchshape) + foreground.shape,
                         key=kwargs['consensus_key'])
    elif kwargs['cuda']:
        consensus_vote_array = create_consensus_array_cuda(
            labels, foreground, overlap_mask, patchshape, neighshape,
            **kwargs)
        offsets_bases_ff = None
        offsets_bases_fb = None
    else:
        all_patches_fgs, all_patches_bgs = computeFGBGsets(
            foreground,
            all_patches,
            labels,
            patchshape,
            rad,
            **kwargs
        )
        consensus_vote_array, offsets_bases_ff, offsets_bases_fb = \
            create_consensus_array(all_patches_fgs, all_patches_bgs,
                                   instances.shape, patchshape, neighshape,
                                   lookup)
        if not kwargs['save_no_intermediates']:
            fn = os.path.join(kwargs['result_folder'], "consensus.pickle")
            logger.info("writing %s", fn)
            f = open(fn, 'wb')
            pickle.dump([consensus_vote_array, offsets_bases_ff,
                         offsets_bases_fb],
                        f, protocol=4)
            f.close()
    return consensus_vote_array, offsets_bases_ff, offsets_bases_fb
