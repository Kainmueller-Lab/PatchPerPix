# credits to Saalfeld Lab
# copied and modified from https://github.com/saalfeldlab/simpleference
from __future__ import print_function
import os
import json

import numpy as np
import dask
import toolz as tz

if __package__ is None or __package__ == '':
    from io_hdflike import IoZarr, IoHDF5
else:
    from .io_hdflike import IoZarr, IoHDF5


def load_input(io, key, offset, context, overlap, output_shape,
               padding_mode='reflect'):

    starts = [off - context[i] - overlap[i] for i, off in enumerate(offset)]
    stops = [off + output_shape[i] + overlap[i] + context[i] for i, off in enumerate(offset)]
    if io.channel_order is not None:
        shape = io.shape[1:]
    else:
        shape = io.shape

    # we pad the input volume if necessary
    pad_left = None
    pad_right = None

    # check for padding to the left
    if any(start < 0 for start in starts):
        pad_left = tuple(abs(start) if start < 0 else 0 for start in starts)
        starts = [max(0, start) for start in starts]

    # check for padding to the right
    if any(stop > shape[i] for i, stop in enumerate(stops)):
        pad_right = tuple(stop - shape[i] if stop > shape[i] else 0 for i, stop in enumerate(stops))
        stops = [min(shape[i], stop) for i, stop in enumerate(stops)]

    if io.channel_order is not None:
        bb = tuple([io.channel_order[io.keys.index(key)]]) + tuple(slice(
            start, stop) for start, stop in zip(starts, stops))
    else:
        bb = tuple(slice(start, stop) for start, stop in zip(starts, stops))
    data = io.read(bb, key)

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

    return data


def run_task_zarr(process,
                  raw_path,
                  save_file,
                  offset_list,
                  input_shape,
                  output_shape,
                  input_keys,
                  target_keys,
                  padding_mode='reflect',
                  num_cpus=1,
                  log_processed=None,
                  channel_order=None,
                  overlap=[0, 0, 0],
                  worker_id=None,
                  context=[0, 0, 0],
                  **kwargs):

    assert os.path.exists(raw_path), \
        'Prediction file {} does not. Please check!'.format(raw_path)
    assert os.path.exists(save_file), \
        'Output file {} does not. Please check!'.format(save_file)

    if raw_path.endswith('.zarr'):
        io_in = IoZarr(raw_path, input_keys, channel_order=channel_order)
    elif raw_path.endswith('.hdf'):
        io_in = IoHDF5(raw_path, input_keys, channel_order=channel_order)
    io_out = IoZarr(save_file, target_keys)

    run_task(process, io_in, input_keys, io_out,
             offset_list, input_shape, output_shape,
             padding_mode=padding_mode,
             num_cpus=num_cpus,
             log_processed=log_processed,
             worker_id=worker_id,
             overlap=overlap,
             context=context,
             **kwargs
             )


def run_task(process,
             io_in,
             in_keys,
             io_out,
             offset_list,
             input_shape,
             output_shape,
             padding_mode='reflect',
             num_cpus=1,
             log_processed=None,
             overlap=[0, 0, 0],
             worker_id=None,
             context=[0, 0, 0],
             **kwargs
             ):

    assert callable(process), 'Task to process is not callable. Please check!'
    assert len(output_shape) == len(input_shape)

    n_blocks = len(offset_list)
    print("Starting vote instances...")
    print("For %i number of blocks" % n_blocks)

    shape = io_in.shape
    #overlapping_output_shape = np.array(output_shape) + np.array(overlap)
    #print('input shape: ', shape, 'overlapping shape: ', overlapping_output_shape)
    # todo: determine context depending on overlap and patchsize half

    @dask.delayed
    def load_offset(offset):
        return load_input(io_in, in_keys[0], offset, context, overlap,
                          output_shape, padding_mode=padding_mode)

    @dask.delayed
    def load_numinst(offset):
        return load_input(io_in, in_keys[1], offset, context, overlap,
                          output_shape, padding_mode=padding_mode)

    @dask.delayed
    def call_process(block, numinst, **kwargs):
        return process(block, numinst, **kwargs)

    @dask.delayed(nout=2)
    def verify_shape(offset, output):

        # crop if necessary
        actual_output_shape = np.array(output.shape)
        block_output_shape = np.array(output_shape)
        offset = np.array(offset)
        overlap = ((actual_output_shape - block_output_shape) / 2).astype(int)
        starts = (offset - overlap).astype(int)

        if np.any(starts < 0):
            bb = tuple(slice(start, stop)
                       for start, stop in zip(np.abs(np.minimum(np.zeros(len(offset), dtype=int), starts)), actual_output_shape))
            output = output[bb]

        stops = offset + block_output_shape + overlap

        if np.any(stops > np.array(input_shape)):
            bb = tuple(slice(0, dim_size - off if stop > dim_size else None)
                        for stop, dim_size, off in zip(stops, input_shape, offset))
            output = output[bb]

        starts = np.maximum(np.zeros(len(offset), dtype=int), starts)
        output_bounding_box = (slice(worker_id, worker_id + 1),) + tuple(slice(start, start + outs)
                                    for start, outs in zip(starts, output.shape))
        output = np.reshape(output, (1,)+output.shape)

        return output, output_bounding_box

    @dask.delayed
    def write_output(output, output_bounding_box):
        io_out.write(output, output_bounding_box)
        return 1    # why return 1?

    @dask.delayed
    def log(off):
        if log_processed is not None:
            with open(log_processed, 'a') as log_f:
                log_f.write(json.dumps(off) + '\n')
        return off

    # iterate over all the offsets, get the input data and predict
    results = []
    for offset in offset_list:
        in_block = tz.pipe(offset, log, load_offset)
        in_numinst = tz.pipe(offset, log, load_numinst)
        output = call_process(in_block, in_numinst, **kwargs)
        output_crop, output_bounding_box = verify_shape(offset, output)
        result = write_output(output_crop, output_bounding_box)
        results.append(result)

    # NOTE: Because dask.compute doesn't take an argument, but rather an
    # arbitrary number of arguments, computing each in turn, the output of
    # dask.compute(results) is a tuple of length 1, with its only element
    # being the results list. If instead we pass the results list as *args,
    # we get the desired container of results at the end.
    success = dask.compute(*results,
                           scheduler='processes',
                           num_workers=num_cpus
                           )   # scheduler='threads'
    print('Ran {0:} jobs'.format(sum(success)))
