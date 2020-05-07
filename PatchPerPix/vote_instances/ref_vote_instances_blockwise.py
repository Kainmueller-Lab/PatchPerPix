import logging
import numpy as np
import zarr
from scipy import ndimage
from skimage import io
import h5py
import time

from multiprocessing import Process, Lock, Pool
import functools
import os
from numcodecs import Blosc

if __package__ is None or __package__ == '':
    from vote_instances import do_block
    from PatchPerPix.util import zarr2hdf
    from PatchPerPix.util import remove_small_components, relabel, color
    from io_hdflike import IoZarr, IoHDF5
else:
    from .vote_instances import do_block
    from ..util import zarr2hdf, remove_small_components, relabel, color
    from .io_hdflike import IoZarr, IoHDF5


logger = logging.getLogger(__name__)

def replace(array, old_values, new_values):
    values_map = np.arange(int(array.max() + 1), dtype=new_values.dtype)
    values_map[old_values] = new_values

    return values_map[array]


def stitch_vote_instances(out_path, in_key, out_key, output_shape,
                          bb_offset, **kwargs):
    logger.info("start stitching")

    zf = zarr.open(out_path, mode='a')
    instances = zf[in_key]
    tmp_shape = np.asarray(instances.shape[1:])
    chunksize = np.asarray(kwargs['chunksize'])
    logger.info('chunksize %s / bb_offset: %s', chunksize, bb_offset)

    block_offset = []
    block_worker = []

    if len(tmp_shape) == 2:
        for yi, y in enumerate(range(0, tmp_shape[0], chunksize[0])):
            ymod = yi % 2
            for xi, x in enumerate(range(0, tmp_shape[1], chunksize[1])):
                xmod = xi % 2
                block_worker.append(2 * ymod + xmod)
                block_offset.append(np.array([y, x]))

    elif len(tmp_shape) == 3:
        for zi, z in enumerate(range(0, tmp_shape[0], chunksize[0])):
            zmod = zi % 2
            for yi, y in enumerate(range(0, tmp_shape[1], chunksize[1])):
                ymod = yi % 2
                for xi, x in enumerate(range(0, tmp_shape[2], chunksize[2])):
                    xmod = xi % 2
                    block_worker.append(4 * zmod + 2 * ymod + xmod)
                    block_offset.append(np.array([z, y, x]))
    else:
        raise NotImplementedError

    result = np.zeros(output_shape, dtype=np.uint16)
    max_label = 0
    overlap = np.array(kwargs.get('overlap', kwargs['patchshape']))
    logger.info('overlap: %s', overlap)

    logger.info("block offsets %s, block workers %s",
                block_offset, block_worker)
    for offset, worker in zip(block_offset, block_worker):

        # load current block into memory
        starts = np.maximum(np.array([0] * len(tmp_shape)), (offset - overlap))
        stops = np.minimum((offset + chunksize + overlap), tmp_shape)
        idx = tuple(slice(start, stop)
                    for start, stop in zip(starts, stops))
        result_idx = tuple(slice(start + off, stop + off)
                           for start, stop, off in
                           zip(starts, stops, bb_offset))
        block = np.array(instances[(slice(worker, worker + 1),) + idx])
        # continue if block is empty
        if np.max(block) == 0:
            continue

        # preprocess block
        block = np.reshape(block, block.shape[1:])
        block = relabel(block.astype(np.uint16), start=max_label + 1)
        print('block shape: ', block.shape)
        max_block_label = np.max(block)

        # copy block without checks to result if it is first one with instances
        if max_label == 0 and max_block_label > 0:
            result[result_idx] = block
            max_label = max_block_label
            continue

        # get mask for overlapping area
        radslice = tuple([slice(overlap[i], block.shape[i] - overlap[i])
                          for i in range(len(overlap))])
        mask = np.ones(block.shape, np.bool)
        mask[radslice] = 0

        overlapping_area = result[result_idx]
        print('overlapping_area: ', overlapping_area.shape)

        # TODO fix this, very hacky, sometimes there is some off by one error
        try:
            np.logical_and(result[result_idx] > 0, block > 0)
        except:
            block = block[...,:-1]
        # check if overlapping labels exist for current block
        if np.sum(np.logical_and(result[result_idx] > 0, block > 0)) > 0:

            logger.info('number of overlapping pixels: %s',
                        np.sum(np.logical_and(overlapping_area > 0, block > 0)))

            # overlay labels:
            # [0] block,
            # [1] already consolidated overlapping areas from other blocks
            overlapping_labels = np.array(
                [block.flatten(), overlapping_area.flatten()])
            overlapping_labels, counts = np.unique(
                overlapping_labels, return_counts=True, axis=1)

            block_labels = np.unique(overlapping_labels[0])
            if 0 in block_labels:
                block_labels = np.delete(block_labels, 0)

            if kwargs.get('stitch_only_largest_overlap'):
                # check for n-m-relation, clean up to make 1-n-relation
                result_labels = np.unique(overlapping_labels[1])
                for result_label in result_labels:
                    if result_label == 0:
                        continue
                    result_label_idx = overlapping_labels[1] == result_label
                    n2m = overlapping_labels[0, result_label_idx]
                    if np.sum(n2m > 0) <= 1:
                        continue

                    # get largest overlap
                    block_label_counts = counts[result_label_idx]
                    cmax = max(block_label_counts[n2m > 0])
                    for l, c in zip(n2m, block_label_counts):
                        if l != 0 and c < cmax:
                            i = np.logical_and(overlapping_labels[0] == l,
                                               overlapping_labels[
                                                   1] == result_label)
                            overlapping_labels = np.delete(overlapping_labels,
                                                           np.where(i)[0],
                                                           axis=1)
                            counts = np.delete(counts, np.where(i)[0], axis=0)

                for label in block_labels:
                    label_idx = overlapping_labels[0] == label

                    if np.sum(label_idx) == 1:
                        nonzero = block == label
                        if overlapping_labels[1, label_idx] == 0:

                            max_label += 1
                            result[result_idx][nonzero] = max_label

                    else:
                        merge = overlapping_labels[1, label_idx]
                        if np.sum(merge > 0) == 0:
                            continue
                        merge_label = np.min(merge[merge > 0])

                        merge_overlaps = merge[merge > 0]
                        if len(merge_overlaps) > 1:
                            result = replace(result, merge_overlaps, np.array(
                                [merge_label] * len(merge_overlaps)))

                        if 0 in merge:
                            nonzero = block == label
                            result[result_idx][nonzero] = merge_label

            else:
                print('doing the new stitching')
                thresh = kwargs.get('overlap_thresh', 0.8)
                block_merge = {}
                dst_merge = {}
                for block_label in block_labels:
                    if block_label == 0:
                        continue
                    # get overlapping dst labels
                    dst_labels = overlapping_labels[1][overlapping_labels[0] ==
                                                       block_label]
                    for dst_label in dst_labels:
                        if dst_label == 0:
                            continue
                        # check overlap
                        block_mask = block == block_label
                        block_mask[mask] = 0
                        dst_mask = overlapping_area == dst_label
                        union = float(np.sum(block_mask * dst_mask))
                        if ((np.sum(dst_mask) / union) > thresh) or (
                            (np.sum(dst_mask) / union) > thresh):
                            if block_label not in block_merge:
                                block_merge[block_label] = dst_label
                            elif dst_label not in dst_merge:
                                dst_merge[dst_label] = block_merge[block_label]
                            else:
                                # find start label
                                # we want to merge dst label and block label
                                if block_merge[block_label] != dst_merge[dst_label]:
                                    print("dst label already in dst merge!")
                                    # relabel existing one's
                                    old_label = block_merge[block_label]
                                    new_label = dst_merge[dst_label]
                                    for k in block_merge:
                                        if block_merge[k] == old_label:
                                            block_merge[k] = new_label
                                    for k in dst_merge:
                                        if dst_merge[k] == old_label:
                                            dst_merge[k] = new_label

                # relabel block
                if len(block_merge.keys()) > 0:
                    old_values = np.array([k for k in block_merge])
                    new_values = np.array([block_merge[k] for k in block_merge])
                    block = replace(block, old_values, new_values)

                # relabel result
                if len(dst_merge.keys()) > 0:
                    old_values = np.array([k for k in dst_merge])
                    new_values = np.array([dst_merge[k] for k in dst_merge])
                    print(old_values, new_values)
                    result = replace(result, old_values, new_values)

                nonzero = block > 0
                result[result_idx][nonzero] = block[nonzero]
                max_label = result.max()

        else:
            # no overlaps, relabel current block and insert into existing result
            nonzero = block > 0
            result[result_idx][nonzero] = block[nonzero]
            max_label = block.max()

    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
    zf.create_dataset(
        out_key,
        data=result.astype(np.uint16),
        shape=[int(s) for s in output_shape],
        compressor=compressor,
        dtype=np.uint16,
        chunks=[int(c) for c in chunksize],
        overwrite=True,
    )

    return result.astype(np.uint16)


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
    print('padded: ', padded)
    if np.any(np.array(starts) < 0):
        padded[np.array(starts) < 0] = 0
    if np.any(np.array(stops) > shape):
        padded[np.array(stops) > shape] = 0

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
        print('load input with padding: ', pad_width)

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
    logger.info('start new block: %s', offset)
    if kwargs['cuda']:
        import pycuda
        import pycuda.compiler
        from pycuda.autoinit import context
        kwargs['context'] = context
        if kwargs['num_parallel_blocks'] > 1:
            kwargs['mutex'] = mutex

    # open result file
    # move io_in to main?
    if pred_file.endswith("zarr"):
        io_in = IoZarr(pred_file, pred_keys, channel_order=channel_order)
    else:
        io_in = IoHDF5(pred_file, pred_keys, channel_order=channel_order)
    io_out = IoZarr(res_file, res_key)

    # load block input
    patchshape = kwargs['patchshape']
    # chunksize = kwargs['chunksize']
    chunksize = np.minimum(kwargs['chunksize'], shape)
    margin = list(np.array(patchshape) // 2)
    # overlap = np.array(kwargs.get('overlap', patchshape))
    overlap = np.array([0, 0, 5])
    in_offset = offset[1:] + np.array(bb_offset)
    logger.info('start new block: %s (with bb_offset)', in_offset)
    if patchshape[0] == 1:
        overlap[0] = 0
    block, _ = load_input(
        io_in, pred_keys[0], in_offset, margin, overlap, chunksize, padding=False)
    block = block.astype(np.float32)

    if kwargs.get("use_pred_fg", False):
        logger.info("loading pred_fg")
        mask, _ = load_input(
            io_in, kwargs['fg_key'], in_offset, margin,
            overlap, chunksize, padding=False)
        mask = mask.astype(np.float32)
        fg_thresh = kwargs['fg_thresh_vi']
        mask = mask > fg_thresh
    else:
        mask = None
    if kwargs['overlapping_inst']:
        numinst, _ = load_input(
            io_in, pred_keys[1], in_offset, margin, overlap, chunksize, padding=False)
        numinst = numinst.astype(np.float32)
    else:
        numinst = None

    if pred_file.endswith('.hdf'):
        io_in.close()

    # call vote instances
    output = do_block(block, numinst, mask, **kwargs)

    # verify output shape and crop if necessary
    output, output_bounding_box = verify_shape(offset, output, shape,
                                               chunksize)
    write_output(io_out, output, output_bounding_box)


def main(pred_file, result_folder='.', **kwargs):
    assert os.path.exists(pred_file), \
        'Prediction file {} does not exists. Please check!'.format(pred_file)

    sample = os.path.basename(pred_file).split('.')[0]
    if sample == 'GMR_38F04_AE_01-20181005_63_G5':
        return
    if sample == 'BJD_127B01_AE_01-20171124_64_H5':
        return
    result_file = os.path.join(result_folder, sample + '.zarr')
    aff_key = kwargs['aff_key']
    fg_key = kwargs.get('fg_key')

    # read input shape
    if pred_file.endswith('.zarr'):
        in_f = zarr.open(pred_file, mode='r')
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
    input_shape = aff_shape[1:]  # without first channel dimension

    # check if blocks should only be within bounding box
    if kwargs.get('only_bb'):
        mid = np.prod(kwargs['patchshape']) // 2
        mask = np.array(in_f[aff_key][mid])
        mask = mask > kwargs['patch_threshold']
        if np.sum(mask) == 0:
            logger.warning("bb empty")
            return
        if kwargs.get('ignore_small_comps', 0) > 0:
            labeled = ndimage.label(mask, np.ones([3] * len(input_shape)))[0]
            labels, counts = np.unique(labeled, return_counts=True)
            labels = labels[counts <= kwargs.get('ignore_small_comps')]
            labeled = replace(labeled, np.array(labels), np.array([0] * len(
                labels)))
            print('num small comps: ', len(labels))
            # for label in labels:
            #     mask[labeled == label] = 0
            mask = labeled > 0
        min = np.min(np.transpose(np.nonzero(mask)), axis=0)
        max = np.max(np.transpose(np.nonzero(mask)), axis=0)
        # TODO for l1 data
        # min = np.array([31, 31, 0])
        # max = np.array([input_shape[0]-30, input_shape[1]-30, input_shape[2]])
        shape = max - min + 1
        bb_offset = min
    else:
        shape = input_shape
        bb_offset = [0] * len(shape)
    if len(shape) == 2:
        shape = (1,) + tuple(shape)
        bb_offset = [0] * len(shape)
    logger.info("input shape: %s, bb cropped shape: %s, offset: %s",
                input_shape, shape, bb_offset)

    if pred_file.endswith('.hdf'):
        in_f.close()

    # create offset lists
    offsets = get_chessboard_offsets(shape, kwargs['chunksize'])
    #offsets = [offset + bb_offset for offset in offsets]
    logger.info('num blocks: %s', len(offsets))
    logger.info("%s", offsets)

    # create temporary zarr dataset for blocks (2^dim x shape)
    tmp_key = 'volumes/tmp_worker'
    skip_tmp_worker = False
    if not os.path.exists(result_file):
        compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
        f = zarr.open(result_file, mode='w')
        f.create_dataset(
            tmp_key,
            shape=(2 ** len(shape),) + tuple(shape),
            compressor=compressor,
            dtype=np.uint32,
            chunks=(1,) + tuple(kwargs['chunksize'])
        )
    else:
        f = zarr.open(result_file, mode='r')
        if tmp_key in f:
            skip_tmp_worker = True

    def init(l):
        global mutex
        mutex = l

    if not skip_tmp_worker:
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
                # if idx < 7 or idx > 8:
                #     continue
                logger.info("start block idx: %s/%s (file %s)",
                            idx, len(offsets), sample)
                blockwise_vote_instances(pred_file, pred_keys, result_file, tmp_key,
                                         shape, channel_order, bb_offset, kwargs,
                                         offset)
    else:
        logger.info("skipping tmp_worker (blocks already exist?)")

    # stitch blocks
    res_key = kwargs.get('res_key', 'vote_instances')
    logger.info("%s", kwargs)
    instances = stitch_vote_instances(
        result_file, tmp_key, res_key, input_shape,
        bb_offset, **kwargs
    )

    # save mip
    save_mip = kwargs.get('save_mip', False)
    if save_mip:
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
        if save_mip:
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
