import numpy as np
import zarr
from scipy import ndimage, spatial
from skimage import io
from skimage.morphology import skeletonize_3d
import networkx as nx
import multiprocessing
import h5py
import sys
import logging
import time


from multiprocessing import Process, Lock, Pool
import functools
import os
from numcodecs import Blosc

if __package__ is None or __package__ == '':
    from vote_instances import do_block
    from PatchPerPix.util import zarr2hdf
    from PatchPerPix.util import remove_small_components, relabel, color
    from io_hdflike import IoZarr
    from vote_instances import affGraphToInstances
else:
    from .vote_instances import do_block
    from ..util import zarr2hdf, remove_small_components, relabel, color
    from .io_hdflike import IoZarr
    from .graph_to_labeling import affGraphToInstances

logger = logging.getLogger(__name__)


compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)


def replace(array, old_values, new_values):
    values_map = np.arange(int(array.max() + 1), dtype=new_values.dtype)
    values_map[old_values] = new_values

    return values_map[array]


def clean_mask(mask, structure, size):
    labeled = ndimage.label(mask, structure)[0]
    labels, counts = np.unique(labeled, return_counts=True)
    labels = labels[counts <= size]
    labeled = replace(
        labeled,
        np.array(labels),
        np.array([0] * len(labels))
    )
    logger.info('removing %i of small components.' % len(labels))
    mask = labeled > 0
    return mask


def update_graph(affgraph, affs, pairs):
    for idx, p in enumerate(affs):
        if p != 0:
            affgraph.add_edge(
                tuple(pairs[idx, :3]),
                tuple(pairs[idx, 3:6]),
                aff=p
            )
    return affgraph


def get_offset_str(offset):
    return "_".join(str(off) for off in offset)


def remove_pairs(pairs, points, patchshape):
    to_delete = []

    for idx, p in enumerate(pairs):
        if np.any(np.abs(
            points[p[0]].astype(np.float32) - points[p[1]].astype(
                np.float32)
        ) > patchshape + 1):
            to_delete.append(p)

    for dlt in to_delete:
        pairs.remove(dlt)

    return pairs


def remove_intra_block_pairs(pairs, points, blocks):
    to_delete = []

    for idx, p in enumerate(pairs):
        in_block = []
        for block in blocks:
            in_block.append(
                np.any(np.all(block == points[p[0]], axis=1)) and
                np.any(np.all(block == points[p[1]], axis=1))
            )
        if np.any(np.array(in_block)):
            to_delete.append(p)

    for dlt in to_delete:
        pairs.remove(dlt)

    return pairs


def stitch_vote_instances(out_path, in_key, out_key, output_shape,
                          bb_offset, bb_shape, pred_file, pred_keys,
                          channel_order, **kwargs):
    logger.info("start stitching...")
    # get parameters
    chunksize = np.asarray(kwargs['chunksize'])
    patchshape = np.asarray(kwargs['patchshape'])
    logger.info('chunksize / bb_offset: %s %s', chunksize, bb_offset)

    zf = zarr.open(out_path, mode='a')
    result = np.zeros(output_shape, np.uint32)
    affgraph = None
    global_patches = []
    global_patch_pairs = []
    global_patch_affs = []
    neighborhood = [
        [-1, 0, 0], [0, -1, 0], [0, 0, -1],
        [1, 0, 0], [0, 1, 0], [0, 0, 1]
    ]

    # prepare kwargs for vote instances
    kwargs['skipRanking'] = True
    kwargs['skipThinCover'] = True
    kwargs['return_intermediates'] = True
    kwargs['mutex'] = multiprocessing.Lock()
    if kwargs['cuda']:
        import pycuda
        import pycuda.compiler
        from pycuda.autoinit import context
        kwargs['context'] = context
    io_in = IoZarr(pred_file, pred_keys, channel_order=channel_order)

    block_offsets = get_offsets(bb_shape, chunksize)
    block_offsets = [off + bb_offset for off in block_offsets]

    # iterate through blocks
    for block_id, offset in enumerate(block_offsets):
        # load block and add coordinate
        block_key = in_key + '/' + get_offset_str(offset)
        if block_key + '/patch_pairs' not in zf:
            # block was empty
            global_patches.append(None)
            global_patch_pairs.append(None)
            global_patch_affs.append(None)
            continue

        # read patch pairs and patch affinities from block
        # add global offset to patch pairs
        patch_pairs = np.array(zf[block_key + '/patch_pairs'])
        aff_graph_mat = np.array(zf[block_key + '/aff_graph_mat'])
        patch_pairs = patch_pairs + np.array(list(offset) * 2)
        logger.info("min / max aff graph mat block: %s %s", np.min(aff_graph_mat), np.max(
            aff_graph_mat))

        # get unique list of selected patches
        num_patches, dim_points = patch_pairs.shape
        selected_patches = np.reshape(
            patch_pairs,
            (num_patches * 2, int(dim_points / 2)))
        selected_patches = np.unique(selected_patches, axis=0)

        # add patches to global lists
        global_patches.append(selected_patches)
        global_patch_pairs.append(patch_pairs)
        global_patch_affs.append(aff_graph_mat)

        if affgraph is None:
            # create global patch affinity graph
            affgraph = nx.Graph()
            affgraph = update_graph(affgraph, aff_graph_mat, patch_pairs)
        else:
            # update graph with existing pairs
            affgraph = update_graph(affgraph, aff_graph_mat, patch_pairs)

            # iterate through neighboring blocks
            # call vote instances for adjacent region
            for neighbor in neighborhood:
                neighbor_offset = offset + neighbor * chunksize
                # check all already processed blocks
                for neighbor_id in range(0, block_id):
                    if np.all(block_offsets[neighbor_id] == neighbor_offset):
                        logger.debug("neighbor: {}, {}, {}".format(
                            neighbor, offset, neighbor_offset))
                        if global_patches[neighbor_id] is None:
                            continue
                        neighbor_patches = global_patches[neighbor_id]

                        # get patches in overlapping region
                        dim = np.argmax(np.array(neighbor) != 0)
                        if neighbor[dim] > 0:
                            current_candidates = selected_patches[
                                selected_patches[:, dim] >= offset[dim] -
                                patchshape[dim]]
                            neighbor_candidates = neighbor_patches[
                                neighbor_patches[:, dim] <= offset[dim] +
                                patchshape[dim]]
                        else:
                            current_candidates = selected_patches[
                                selected_patches[:, dim] <= offset[dim] +
                                patchshape[dim]]
                            neighbor_candidates = neighbor_patches[
                                neighbor_patches[:, dim] >= offset[dim] -
                                patchshape[dim]]
                        logger.debug("current and neighbor candidates: {}, "
                                     "{}".format(len(current_candidates),
                                                 len(neighbor_candidates)))

                        # continue if no selected patches in neighboring
                        # region for one of the blocks
                        if len(current_candidates) == 0 or len(
                            neighbor_candidates) == 0:
                            continue

                        # get possible pairs with distance of patchshape
                        candidates = np.concatenate(
                            [current_candidates, neighbor_candidates]
                        )
                        tree = spatial.cKDTree(candidates, leafsize=4)
                        pairs = tree.query_pairs(1 * np.sum(patchshape + 1),
                                                 p=1)
                        pairs = remove_pairs(pairs, candidates, patchshape)

                        # remove intra-block pairs
                        pairs = remove_intra_block_pairs(
                            pairs, candidates,
                            [current_candidates, neighbor_candidates]
                        )
                        if len(pairs) == 0:
                            continue

                        # remove non-used candidates
                        candidate_idx = []
                        for p in pairs:
                            candidate_idx.append(p[0])
                            candidate_idx.append(p[1])
                        candidates_cleaned = candidates[np.unique(
                            candidate_idx)]

                        # get overlapping area
                        bb_start = np.maximum(
                            np.min(candidates_cleaned, axis=0) - patchshape,
                            [0, 0, 0])
                        bb_stop = np.minimum(
                            np.max(candidates_cleaned, axis=0) + patchshape,
                            output_shape)
                        logger.debug('overlapping bb: {}, {}'.format(
                            bb_start, bb_stop))

                        # load neighboring region as input
                        margin = list(np.array(patchshape) // 2)
                        overlap = [0, 0, 0]
                        bb_size = np.maximum(bb_stop - bb_start, [1, 1, 1])
                        block, _ = load_input(
                            io_in, pred_keys[0], bb_start, margin, overlap,
                            bb_size, padding=False)
                        block = block.astype(np.float32)
                        if kwargs['overlapping_inst']:
                            numinst, _ = load_input(
                                io_in, pred_keys[1], bb_start, margin, overlap,
                                bb_size, padding=False)
                            numinst = numinst.astype(np.float32)
                        else:
                            numinst = None

                        # call vote instances, pass selected patches and
                        # patch pairs
                        overlapping_pairs = np.zeros((len(pairs), 6),
                                                     dtype=np.uint32)
                        for i, p in enumerate(pairs):
                            overlapping_pairs[i, :3] = candidates[p[0]]
                            overlapping_pairs[i, 3:] = candidates[p[1]]
                        candidates_relative = np.copy(candidates_cleaned)
                        candidates_relative -= (bb_start - margin)
                        pairs_relative = np.copy(overlapping_pairs)
                        pairs_relative -= np.array(list(bb_start - margin)*2,
                                                   dtype=np.uint32)

                        _, aff_graph_mat = do_block(
                            block, numinst, None,
                            selected_patches=candidates_relative,
                            selected_patch_pairs=pairs_relative,
                            **kwargs
                        )
                        logger.debug("patch affinities: {}".format(
                            aff_graph_mat))
                        logger.debug("min / max aff graph mat: {}, {}".format(
                            np.min(aff_graph_mat), np.max(aff_graph_mat)))

                        # update global patch graph
                        affgraph = update_graph(affgraph, aff_graph_mat,
                                                overlapping_pairs)

    # partition patch affinity graph by connected components analysis
    in_f = zarr.open(pred_file, mode='r')
    # list with selected patches (complete patches, not only their id)
    labels = {}
    rad = np.array([p // 2 for p in patchshape])
    mid = int(np.prod(patchshape) / 2)
    foreground_to_cover = np.array(in_f[pred_keys[0]][mid])
    foreground_to_cover = foreground_to_cover >= kwargs['patch_threshold']
    sz = np.prod(in_f[pred_keys[0]].shape)/1024/1024/1024
    logger.info("%s", sz)
    coords = list(affgraph.nodes)
    if sz < 20:
        pred_affs = np.array(in_f[pred_keys[0]])
    else:
        pred_affs = in_f[pred_keys[0]]
        coords = sorted(coords)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('min / max coords: {}, {}'.format(
            np.min(np.array(coords), axis=0), np.max(np.array(coords), axis=0)))
    for coord in coords:
        coord_key = get_offset_str(coord)
        z, y, x = coord
        labels[coord_key] = np.array(pred_affs[:, z, y, x])
    result, _ = affGraphToInstances(
        affgraph, labels,
        rad=rad,
        debug_output1=False,
        debug_output2=False,
        instances=result,
        foreground_to_cover=foreground_to_cover,
        sparse_labels=True,
        **kwargs)

    return result


# this returns the offsets for the given output blocks.
# blocks are padded on the fly during inference if necessary
def get_chessboard_offsets(total_shape, chunksize):
    offsets = []
    if len(total_shape) == 2:
        for yi, y in enumerate(range(0, total_shape[0], chunksize[0])):
            ymod = yi % 2
            for xi, x in enumerate(range(0, total_shape[1], chunksize[1])):
                xmod = xi % 2
                offsets.append([2 * ymod + xmod, y, x])

    if len(total_shape) == 3:
        for zi, z in enumerate(range(0, total_shape[0], chunksize[0])):
            zmod = zi % 2
            for yi, y in enumerate(range(0, total_shape[1], chunksize[1])):
                ymod = yi % 2
                for xi, x in enumerate(range(0, total_shape[2], chunksize[2])):
                    xmod = xi % 2
                    offsets.append([4 * zmod + 2 * ymod + xmod, z, y, x])

    return offsets


def get_offsets(total_shape, chunksize):
    # get block offsets
    block_offsets = []
    if len(total_shape) == 2:
        for yi, y in enumerate(range(0, total_shape[0], chunksize[0])):
            for xi, x in enumerate(range(0, total_shape[1], chunksize[1])):
                block_offsets.append(np.array([y, x]))

    elif len(total_shape) == 3:
        for zi, z in enumerate(range(0, total_shape[0], chunksize[0])):
            for yi, y in enumerate(range(0, total_shape[1], chunksize[1])):
                for xi, x in enumerate(range(0, total_shape[2], chunksize[2])):
                    block_offsets.append(np.array([z, y, x]))
    else:
        raise NotImplementedError
    return block_offsets


def load_input(io, key, offset, context, overlap, output_shape, padding=True,
               padding_mode='constant'):
    starts = [off - context[i] - overlap[i] for i, off in enumerate(offset)]
    stops = [off + output_shape[i] + overlap[i] + context[i]
             for i, off in enumerate(offset)]
    if io.channel_order is not None:
        shape = io.shape[1:]
    else:
        shape = io.shape
    if len(shape) == 2:
        unsqueezed = True
        shape = (1,) + tuple(shape)
    else:
        unsqueezed = False

    # we pad the input volume if necessary
    pad_left = None
    pad_right = None

    padded = np.array(context) + np.array(overlap)
    logger.info('padded: %s', padded)
    if np.any(np.array(starts) < 0):
        padded[np.array(starts) < 0] = 0

    # check for padding to the left
    # heads up: changed it, no padding
    if padding:
        if any(start < 0 for start in starts):
            pad_left = tuple(abs(start) if start < 0 else 0 for start in starts)
            starts = [max(0, start) for start in starts]
    else:
        starts = np.maximum([0, 0, 0], starts)

    # check for padding to the right
    if padding:
        if any(stop > shape[i] for i, stop in enumerate(stops)):
            pad_right = tuple(
                stop - shape[i] if stop > shape[i] else 0 for i, stop in
                enumerate(stops))
            stops = [min(shape[i], stop) for i, stop in enumerate(stops)]
    else:
        stops = np.minimum(shape, stops)

    if unsqueezed:
        del starts[0]
        del stops[0]
    if io.channel_order is not None:
        bb = tuple([io.channel_order[io.keys.index(key)]]) + tuple(slice(
            start, stop) for start, stop in zip(starts, stops))
    else:
        bb = tuple(slice(start, stop) for start, stop in zip(starts, stops))
    data = io.read(bb, key)

    if unsqueezed:
        data = np.expand_dims(data, axis=1)

    # pad if necessary
    if pad_left is not None or pad_right is not None:
        pad_left = (0, 0, 0) if pad_left is None else pad_left
        pad_right = (0, 0, 0) if pad_right is None else pad_right
        pad_width = tuple((pl, pr) for pl, pr in zip(pad_left, pad_right))
        if io.channel_order is not None:
            data = np.pad(data, ((0, 0),) + pad_width, mode=padding_mode)
        else:
            data = np.pad(data, pad_width, mode=padding_mode)
        logger.info('load input with padding: %s', pad_width)

    return data, padded


def verify_shape(offset, output, shape, chunksize):
    # crop if necessary
    tmp_channel = offset[0]
    offset = offset[1:]
    actual_output_shape = np.array(output.shape)
    block_output_shape = np.array(chunksize)
    offset = np.array(offset)
    overlap = ((actual_output_shape - block_output_shape) / 2).astype(int)
    starts = (offset - overlap).astype(int)

    if np.any(starts < 0):
        bb = tuple(slice(start, stop)
                   for start, stop in zip(
            np.abs(np.minimum(np.zeros(len(offset), dtype=int), starts)),
            actual_output_shape))
        output = output[bb]

    stops = offset + block_output_shape + overlap

    if np.any(stops > np.array(shape)):
        bb = tuple(slice(0, dim_size - off if stop > dim_size else None)
                   for stop, dim_size, off in zip(stops, shape, offset))
        output = output[bb]

    starts = np.maximum(np.zeros(len(offset), dtype=int), starts)
    output_bounding_box = (slice(tmp_channel, tmp_channel + 1),) + tuple(
        slice(start, start + outs)
        for start, outs in zip(starts, output.shape))
    output = np.reshape(output, (1,) + output.shape)

    return output, output_bounding_box


def write_output(io_out, output, output_bounding_box):
    io_out.write(output, output_bounding_box)


def blockwise_vote_instances(
    pred_file, pred_keys, res_file, res_key, shape, channel_order,
    bb_offset, kwargs, offset):
    logger.info("start new block: {}".format(offset))
    if kwargs['cuda']:
        import pycuda
        import pycuda.compiler
        from pycuda.autoinit import context
        kwargs['context'] = context
        if kwargs['num_parallel_blocks'] > 1:
            kwargs['mutex'] = mutex

    # open result file
    # move io_in to main?
    io_in = IoZarr(pred_file, pred_keys, channel_order=channel_order)
    if not os.path.exists(res_file):
        io_out = zarr.open(res_file, mode='w')
    else:
        io_out = zarr.open(res_file, mode='a')

    # load block input
    patchshape = kwargs['patchshape']
    # chunksize = kwargs['chunksize']
    chunksize = np.minimum(kwargs['chunksize'], shape)
    margin = list(np.array(patchshape) // 2)
    overlap = np.array([0, 0, 0])
    in_offset = offset + np.array(bb_offset)
    if patchshape[0] == 1:
        overlap[0] = 0

    ds_name = res_key + '/' + get_offset_str(in_offset) + '/patch_pairs'
    if ds_name in io_out:
        logger.info("{} already processed.".format(ds_name))
        return

    # if small comps should be ignored, load mask
    if kwargs.get('ignore_small_comps', 0) > 0 and len(pred_keys) > 2:
        mask, _ = load_input(
            io_in, pred_keys[2], in_offset, margin, overlap, chunksize,
            padding=False
        )
        mask = np.squeeze(mask).astype(np.uint8)
        if np.sum(mask > kwargs['patch_threshold']) == 0:
            logger.info("skipping block {}, because no foreground is "
                        "contained.".format(offset))
            return
    else:
        mask = None

    # load affinities
    block, padded = load_input(
        io_in, pred_keys[0], in_offset, margin, overlap, chunksize,
        padding=False
    )
    block_shape = block.shape[1:]
    # load numinst
    if kwargs['overlapping_inst']:
        numinst, _ = load_input(
            io_in, pred_keys[1], in_offset, margin, overlap, chunksize,
            padding=False)
        numinst = numinst.astype(np.float32)
    else:
        numinst = None

    # call vote instances
    patch_pairs, aff_graph_mat = do_block(
        block.astype(np.float32),
        numinst,
        mask,
        **kwargs
    )

    # save patch pairs and patch affinities
    if patch_pairs is not None:
        patch_pairs -= np.array(list(padded) * 2, dtype=np.uint32)
        io_out.create_dataset(
            ds_name,
            data=patch_pairs,
            shape=patch_pairs.shape,
            compressor=compressor,
            dtype=np.uint32,
            overwrite=True,
        )
        io_out[ds_name].attrs['block_shape'] = block_shape

        ds_name = res_key + '/' + get_offset_str(in_offset) + '/aff_graph_mat'
        io_out.create_dataset(
            ds_name,
            data=aff_graph_mat,
            shape=aff_graph_mat.shape,
            compressor=compressor,
            dtype=np.float32,
            overwrite=True,
        )


def main(pred_file, result_folder='.', **kwargs):
    """Calls vote_instances blockwise and stitches them afterwards.

    Args:

        pred_file (``string``):

            Filename of prediction. Should be zarr.

        result_folder (``string``):

            Path to result folder.

        **kwargs (``dict``):

            All arguments needed for vote_instances and stitching. In the
            following only stitching arguments are listed: (maybe they should be
            renamed?)

            aff_key
            fg_key
            res_key
            overlapping_inst
            only_bb
            patchshape
            patch_threshold
            ignore_small_comps
            skeletonize_foreground
            chunksize
            num_parallel_blocks

            OUTPUT / POSTPROCESSING:
            save_mip
            remove_small_comps
            dilate_instances

    """
    assert os.path.exists(pred_file), \
        'Prediction file {} does not exist. Please check!'.format(pred_file)

    sample = os.path.basename(pred_file).split('.')[0]
    result_file = os.path.join(result_folder, sample + '.zarr')
    kwargs['result_folder'] = result_folder

    kwargs['return_intermediates'] = True

    # dataset keys
    aff_key = kwargs['aff_key']
    fg_key = kwargs.get('fg_key')
    res_key = kwargs.get('res_key', 'vote_instances')
    cleaned_mask_key = 'volumes/foreground_cleaned'
    tmp_key = 'volumes/blocks'

    # read input shape
    if pred_file.endswith('.zarr'):
        in_f = zarr.open(pred_file, mode='a')
    else:
        raise NotImplementedError
    aff_shape = in_f[aff_key].shape
    channel_order = [slice(0, aff_shape[0])]
    pred_keys = [aff_key]

    if kwargs['overlapping_inst']:
        numinst_shape = in_f[fg_key].shape
        channel_order.append(slice(0, numinst_shape[0]))
        pred_keys += [fg_key]
        assert aff_shape[1:] == numinst_shape[1:], \
            'Please check: affinity and numinst shape do not match!'
    input_shape = aff_shape[1:]     # without first channel dimension

    # get bounding box for foreground
    if kwargs.get('only_bb'):
        mid = np.prod(kwargs['patchshape']) // 2
        if cleaned_mask_key in in_f:
            pred_keys += [cleaned_mask_key]
            channel_order.append(slice(0, 1))
            shape = np.array(in_f[cleaned_mask_key].attrs['fg_shape'])
            bb_offset = np.array(in_f[cleaned_mask_key].attrs['offset'])
        else:
            mask = np.array(in_f[aff_key][mid])
            mask = mask > kwargs['patch_threshold']
            if np.sum(mask) == 0:
                logger.info('Volume has no foreground voxel, returning...')
                return
            # remove small components
            if kwargs.get('ignore_small_comps', 0) > 0:
                mask = clean_mask(mask, np.ones([3] * mask.ndim),
                                  kwargs.get('ignore_small_comps'))
                mask = mask.astype(np.uint8)
            # skeletonize mask (flylight specific)
            if kwargs.get('skeletonize_foreground'):
                mask = skeletonize_3d(mask) > 0
                mask = mask.astype(np.uint8)
            # save processed mask to input file
            if kwargs.get('ignore_small_comps', 0) > 0 or kwargs.get(
                'skeletonize_foreground'):
                in_f.create_dataset(
                    cleaned_mask_key,
                    data=np.reshape(mask, (1,) + mask.shape),
                    shape=(1,) + mask.shape,
                    compressor=compressor,
                    dtype=np.uint8,
                    overwrite=True
                )
            min = np.min(np.transpose(np.nonzero(mask)), axis=0)
            max = np.max(np.transpose(np.nonzero(mask)), axis=0)
            shape = max - min + 1
            bb_offset = min
            if kwargs.get('ignore_small_comps', 0) > 0 or kwargs.get(
                'skeletonize_foreground'):
                in_f[cleaned_mask_key].attrs['offset'] = [int(off)
                                                          for off in bb_offset]
                in_f[cleaned_mask_key].attrs['fg_shape'] = [int(dim)
                                                            for dim in shape]
    else:
        shape = input_shape
        bb_offset = [0] * len(shape)
    if len(shape) == 2:
        shape = (1,) + tuple(shape)
    logger.info("input shape: {}, bb cropped shape: {}, offset: {}".format(
        input_shape, shape, bb_offset))

    # create offset lists
    offsets = get_offsets(shape, kwargs['chunksize'])
    # offsets = [offset + bb_offset for offset in offsets]
    logger.info("processing {} blocks".format(len(offsets)))
    logger.debug("blocks: {}".format(offsets))

    def init(l):
        global mutex
        mutex = l

    mutex = Lock()
    if kwargs['num_parallel_blocks'] > 1:
        pool = Pool(
            processes=kwargs['num_parallel_blocks'],
            initializer=init,
            initargs=(mutex,)
            )
        pool.map(functools.partial(
            blockwise_vote_instances,
            pred_file, pred_keys, result_file, tmp_key,
            shape, channel_order, bb_offset,
            kwargs),
                 offsets)
        pool.close()
        pool.join()
    else:
        kwargs['mutex'] = mutex
        for idx, offset in enumerate(offsets):
            logger.info("start block idx: %s/%s (file %s)",
                        idx, len(offsets), sample)
            blockwise_vote_instances(pred_file, pred_keys, result_file, tmp_key,
                                     shape, channel_order, bb_offset, kwargs,
                                     offset)

    # stitch blocks
    #child_pid = os.fork()
    #if child_pid == 0:
    # child process
    instances = stitch_vote_instances(
        result_file, tmp_key, res_key, input_shape,
        bb_offset, shape, pred_file, pred_keys, channel_order, **kwargs
    )
    # save mip
    if kwargs.get('save_mip', False):
        colored = color(np.max(instances, axis=0))
        io.imsave(
            os.path.join(result_folder, sample + '.png'),
            colored.astype(np.uint8)
        )

    # remove small components
    remove_small_comps = kwargs.get('remove_small_comps', 0)
    if remove_small_comps > 0:

        instances = remove_small_components(instances, remove_small_comps)
        instances = relabel(instances)
        io.imsave(
            os.path.join(result_folder, sample + '.tif'),
            instances.astype(np.uint16),
            plugin='tifffile'
        )
        if kwargs.get('save_mip', False):
            colored = color(np.max(instances, axis=0))
            io.imsave(
                os.path.join(result_folder, sample + '_cleaned.png'),
                colored.astype(np.uint8)
            )

    if kwargs['output_format'] == 'hdf':
        hf = h5py.File(os.path.join(result_folder, sample + '.hdf'), 'w')
        hf.create_dataset(
            res_key,
            data=instances.astype(np.uint16),
            dtype=np.uint16,
            compression='gzip'
        )
        if kwargs.get("dilate_instances", False):
            logger.info("dilating")
            instdil = np.copy(instances)
            for lbl in np.unique(instances):
                if lbl == 0:
                    continue
                label_mask = instdil == lbl
                dilated_label_mask = ndimage.binary_dilation(label_mask,
                                                             iterations=1)
                instdil[dilated_label_mask] = lbl
            hf.create_dataset(
                res_key + "_dil_1",
                data=instdil.astype(np.uint16),
                dtype=np.uint16,
                compression='gzip'
            )

        #sys.exit(0)

    #pid, status = os.waitpid(child_pid, 0)


if __name__ == '__main__':

    pred_file = '/home/maisl/workspace/ppp/flylight/setup04_190812_00/val' \
                '/processed/100000/BJD_117B11_AE_01-20171013_65_B6.zarr'
    result_folder = '/home/maisl/workspace/ppp/flylight/setup04_190812_00/val' \
                    '/instanced/100000'

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    ex_kwargs = {'patch_threshold': 0.7,
                 'fc_threshold': 0.5,
                 'cuda': True,
                 'blockwise': True,
                 'num_parallel_samples': 1,
                 'num_workers': 1,
                 'chunksize': [184, 92, 92],
                 'debug': False,
                 'select_patches_for_sparse_data': True,
                 'save_no_intermediates': True,
                 'output_format': 'hdf',
                 'includeSinglePatchCCS': True,
                 'sample': 1.0,
                 'removeIntersection': True,
                 'mws': False,
                 'isbiHack': False,
                 'mask_fg_border': False,
                 'graphToInst': False,
                 'skipLookup': False,
                 'skipConsensus': False,
                 'skipRanking': False,
                 'skipThinCover': False,
                 'termAfterThinCover': False,
                 'check_required': False,
                 'train_net_name': 'train_net',
                 'test_net_name': 'test_net',
                 'train_input_shape': [140, 140, 140],
                 'test_input_shape': [180, 180, 180],
                 'patchshape': [7, 7, 7],
                 'factory': 'ppp_unet',
                 'num_fmaps': 12,
                 'num_output_fmaps': 343,
                 'fmap_inc_factors': [3, 3, 3],
                 'fmap_dec_factors': [0.8, 1.0, 2.0],
                 'downsample_factors': [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
                 'overlapping_inst': True}

    main(pred_file, result_folder, **ex_kwargs)
