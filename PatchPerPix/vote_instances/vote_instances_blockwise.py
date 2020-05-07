# credits to Saalfeld Lab
# copied and modified from https://github.com/saalfeldlab/simpleference
import os
import time
import json
import logging
import numpy as np
import zarr
import h5py
from numcodecs import Blosc
from datetime import datetime
from glob import glob
from skimage import io
import argparse
from concurrent.futures import ProcessPoolExecutor

if __package__ is None or __package__ == '':
    from offsets import get_offset_lists, get_chessboard_offset_lists
    from parallelize_task import run_task_zarr
    from vote_instances import do_block
    from PatchPerPix.util import zarr2hdf, remove_small_components, relabel, color
else:
    #from .offsets import get_offset_lists, get_chessboard_offset_lists
    from .parallelize_task import run_task_zarr
    from .vote_instances import do_block
    from ..util import zarr2hdf, remove_small_components, relabel, color


def replace(array, old_values, new_values):
    values_map = np.arange(int(array.max() + 1), dtype=new_values.dtype)
    values_map[old_values] = new_values

    return values_map[array]


logger = logging.getLogger(__name__)

def stitch_vote_instances(out_path, in_key, out_key, chunksize, patchshape):

    zf = zarr.open(out_path, mode='a')
    instances = zf[in_key]
    input_shape = instances.shape
    shape = np.asarray(input_shape[1:])
    chunksize = np.asarray(chunksize)

    block_offset = []
    block_worker = []

    if len(shape) == 2:
        for yi, y in enumerate(range(0, shape[0], chunksize[0])):
            ymod = yi % 2
            for xi, x in enumerate(range(0, shape[1], chunksize[1])):
                xmod = xi % 2

                block_worker.append(2 * ymod + xmod)
                block_offset.append(np.array([y, x]))

    elif len(shape) == 3:
        for zi, z in enumerate(range(0, shape[0], chunksize[0])):
            zmod = zi % 2
            for yi, y in enumerate(range(0, shape[1], chunksize[1])):
                ymod = yi % 2
                for xi, x in enumerate(range(0, shape[2], chunksize[2])):
                    xmod = xi % 2

                    block_worker.append(4 * zmod + 2 * ymod + xmod)
                    block_offset.append(np.array([z, y, x]))
    else:
        raise NotImplementedError

    result = np.zeros(shape, dtype=np.uint16)
    max_label = 0
    overlap = np.array(patchshape)

    for offset, worker in zip(block_offset, block_worker):

        # load current block into memory
        starts = np.maximum(np.array([0] * len(shape)), (offset - overlap))
        stops = np.minimum((offset + chunksize + overlap), shape)
        idx = tuple(slice(start, stop) for start, stop in zip(starts, stops))
        block = np.array(instances[(slice(worker, worker + 1),) + idx])
        block = np.reshape(block, block.shape[1:])
        max_block_label = np.max(block)

        # continue if block is empty
        if max_block_label == 0:
            continue

        # copy block without checks to result if it is first one with instances
        if max_label == 0 and max_block_label > 0:
            result[idx] = block.astype(np.uint16)
            max_label = max_block_label
            continue

        block_labels = np.unique(block)
        if 0 in block_labels:
            block_labels = np.delete(block_labels, 0)

        overlapping_area = result[idx]

        # check if overlapping labels exist for current block
        if np.sum(np.logical_and(result[idx] > 0, block > 0)) > 0:

            logger.info('number of overlapping pixels: %s',
                        np.sum(np.logical_and(overlapping_area > 0, block > 0)))

            # overlay labels:
            # [0] block,
            # [1] already consolidated overlapping areas from other blocks
            overlapping_labels = np.array(
                [block.flatten(), overlapping_area.flatten()])
            overlapping_labels, counts = np.unique(
                overlapping_labels, return_counts=True, axis=1)

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
                                                       np.where(i)[0], axis=1)
                        counts = np.delete(counts, np.where(i)[0], axis=0)

            for label in block_labels:
                label_idx = overlapping_labels[0] == label

                if np.sum(label_idx) == 1:
                    nonzero = block == label
                    if overlapping_labels[1, label_idx] == 0:

                        max_label += 1
                        result[idx][nonzero] = max_label

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
                        result[idx][nonzero] = merge_label

        else:

            # no overlaps, relabel current block and insert into existing result
            block = replace(block, block_labels, np.array(
                range(max_label + 1, max_label + 1 + len(block_labels))))
            nonzero = block > 0
            result[idx][nonzero] = block[nonzero]
            max_label = block.max()

    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
    zf.create_dataset(
        out_key,
        data=result.astype(np.uint16),
        shape=shape,
        compressor=compressor,
        dtype=np.uint16,
        chunks=chunksize,
        overwrite=True,
    )

    return result.astype(np.uint16)


def run_worker(
        pred_file,
        out_file,
        in_keys,
        out_key,
        worker_id,
        offset_list,
        input_shape,
        output_shape,
        chunksize,
        patchshape,
        channel_order,
        **kwargs
):

    if worker_id != 6:
        return
    # assume all preparations are done by either single_worker
    # or non_overlapping_chessboard_worker
    run_task_zarr(
        do_block,
        pred_file,
        out_file,
        offset_list,
        input_shape=output_shape,
        output_shape=chunksize,
        input_keys=in_keys,
        target_keys=out_key,
        num_cpus=1,
        channel_order=channel_order,
        overlap=patchshape,
        worker_id=worker_id,
        context=list(np.array(patchshape) // 2),
        **kwargs, **{'patchshape': patchshape}
    )


def single_worker(pred_file, out_file, worker, chunksize, patchshape, **kwargs):
    # todo: sorry not updated yet

    assert os.path.exists(
        pred_file), 'Prediction file {} does not. Please check!'.format(
        pred_file)

    # open prediction file
    pf = zarr.open(pred_file, mode='r')
    input_shape = pf['volumes/pred_affs'].shape
    output_shape = input_shape[1:]

    # create offset list
    offset_folder = os.path.join(os.path.dirname(out_file), 'offsets')
    get_offset_lists(output_shape, range(worker), offset_folder,
                     output_shape=chunksize)
    offset_file = os.path.join(offset_folder, 'offset_list.json')
    with open(offset_file, 'r') as f:
        offset_list = json.load(f)

    t_1 = time.time()
    run_worker(pred_file, out_file, worker, offset_list, input_shape,
               output_shape, chunksize, patchshape)
    t_process = time.time() - t_1
    logger.info("Running vote instances in %f s", t_process)


def non_overlapping_chessboard_worker(
        pred_file,
        out_file,
        pred_keys=['volumes/pred_affs'],
        out_key='volumes/tmp_worker',
        num_workers=8,
        chunksize=None,
        patchshape=None,
        **kwargs
):
    assert os.path.exists(pred_file), \
        'Prediction file {} does not exists. Please check!'.format(pred_file)

    # open prediction file
    # heads up: actually not working for hdf
    if pred_file.endswith('.zarr'):
        in_f = zarr.open(pred_file, mode='r')
    elif pred_file.endswith('.hdf'):
        in_f = h5py.File(pred_file, 'r')
    else:
        raise NotImplementedError

    # assuming pred_affs being first key
    input_shape = in_f[pred_keys[0]].shape
    output_shape = input_shape[1:]  # without affinity dimension
    if pred_file.endswith('.hdf'):
        in_f.close()

    input_shape_numinst = in_f[pred_keys[1]].shape
    channel_order = [slice(0, input_shape[0]), slice(0, input_shape_numinst[0])]

    # create offset lists
    sample = os.path.basename(out_file).split('.')[0]
    offset_folder = os.path.join(os.path.dirname(out_file), sample + '_offsets')
    get_chessboard_offset_lists(output_shape,
                                range(num_workers),
                                offset_folder,
                                output_shape=chunksize)

    # create zarr file with size (worker x output_shape)
    if not os.path.exists(out_file):
        compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
        f = zarr.open(out_file, mode='w')
        f.create_dataset(
            out_key,
            shape=(num_workers,) + output_shape,
            compressor=compressor,
            dtype='uint32',
            chunks=(1,) + tuple(chunksize)
        )

    with ProcessPoolExecutor(max_workers=num_workers) as pp:

        tasks = []
        for worker_id in range(num_workers):
            offset_file = os.path.join(
                offset_folder, 'offset_list_%i.json' % worker_id)
            with open(offset_file, 'r') as f:
                offset_list = json.load(f)
            tasks.append(
                pp.submit(run_worker, pred_file, out_file,
                          pred_keys, out_key, worker_id,
                          offset_list, input_shape, output_shape,
                          chunksize, patchshape, channel_order, **kwargs)
            )

        result = [t.result() for t in tasks]

    # check worker status and return
    if all(result):
        logger.info("All workers finished properly.")
        return 0
    else:
        logger.info("WARNING: at least one process didn't finish properly.")
        return -1


def main(
        pred_file,
        result_folder='.',
        pred_keys=['volumes/pred_affs'],
        result_key='volumes/instances',
        patchshape=[7, 7, 7],
        chunksize=[92, 92, 92],
        num_workers=8,
        remove_small_comps=0,
        save_mip=False,
        **kwargs
):
    sample = os.path.basename(pred_file).split('.')[0]
    result_file = os.path.join(result_folder, sample + '.zarr')
    tmp_key = 'volumes/tmp_worker'

    if num_workers == 1:
        single_worker(pred_file, result_file, num_workers, **kwargs)

    else:
        non_overlapping_chessboard_worker(
            pred_file, result_file, pred_keys, tmp_key, num_workers,
            chunksize, patchshape, **kwargs, **{'result_folder': result_folder}
        )
        # stitch blocks
        instances = stitch_vote_instances(
            result_file, tmp_key, result_key, chunksize, patchshape)
        # convert result to hdf
        if kwargs['output_format'] == 'hdf':
            zarr2hdf(
                result_file,
                hdf_file=os.path.join(result_folder, sample + '.hdf'),
                zarr_key=['volumes/instances'],
                hdf_key=['volumes/instances']
            )

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
                os.path.join(result_folder, sample + '.png'),
                colored.astype(np.uint8)
            )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str,
                        help='input file or folder', required=True)
    parser.add_argument('--out_file', type=str,
                        help='output file or folder', required=True)
    parser.add_argument('--in_key', type=str, default='volumes/pred_affs',
                        help='input file or folder')
    parser.add_argument('--out_key', type=str, default='volumes/instances',
                        help='output file or folder')
    parser.add_argument('--num_workers', type=int,
                        default=8, help='number of workers')
    parser.add_argument('-p', '--patchshape', type=int,
                        nargs='+', required=True)
    parser.add_argument('-c', '--chunksize', type=int,
                        nargs='+', required=True)
    parser.add_argument('--remove_small_comps', type=int,
                        default=0,
                        help='remove components smaller than this value')
    parser.add_argument("--save_mip", help="",
                        action="store_true")

    args = parser.parse_args()

    tmp_key = 'volumes/tmp_worker'
    kwargs = {'check_required': False,
              'debug': False,
              'select_patches_for_sparse_data': True,
              'save_no_intermediates': True
              }

    if args.in_file.endswith('.zarr'):
        pred_files = [args.in_file]
    elif os.path.isdir(args.in_file):
        pred_files = glob(args.in_file + '/*.zarr')
    else:
        pred_files = [args.in_file]

    for pred_file in pred_files:

        sample = os.path.basename(pred_file).split('.')[0]
        logger.info('Processing sample... %s', sample)
        if args.out_file.endswith('.zarr'):
            out_file = args.out_file
            kwargs['result_folder'] = os.path.dirname(args.out_file)
        else:
            # if zarr is not in filename, assume it is folder
            out_file = os.path.join(args.out_file, sample + '.zarr')
            kwargs['result_folder'] = args.out_file

        if os.path.exists(out_file):
            continue

        t1 = datetime.now()
        if args.num_workers == 1:
            single_worker(pred_file, out_file, args.num_workers)

        else:
            non_overlapping_chessboard_worker(
                pred_file, out_file, args.in_key, tmp_key, args.num_workers,
                args.chunksize, args.patchshape, **kwargs
            )

            instances = stitch_vote_instances(
                out_file, tmp_key, args.out_key, args.chunksize,
                args.patchshape)

            if args.remove_small_comps > 0:
                cleaned = remove_small_components(instances,
                                                  args.remove_small_comps)
                relabeled = relabel(cleaned)
                io.imsave(
                    out_file.replace('.zarr', '.tif'),
                    relabeled.astype(np.uint16),
                    plugin='tifffile'
                )

            if args.save_mip:
                colored = color(np.max(relabeled, axis=0))
                io.imsave(
                    out_file.replace('.zarr', '_mip.png'),
                    colored.astype(np.uint8)
                )

            zarr2hdf(
                out_file,
                hdf_file=out_file.replace('.zarr', '.hdf'),
                zarr_key=['volumes/instances'],
                hdf_key=['volumes/instances']
            )

        logger.info("%s", datetime.now() - t1)
