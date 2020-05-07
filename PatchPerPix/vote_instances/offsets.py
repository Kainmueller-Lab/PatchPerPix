# credits to Saalfeld Lab
# copied and modified from https://github.com/saalfeldlab/simpleference
from __future__ import print_function
import os
import json

from random import shuffle


def _offset_list(shape, output_shape):
    in_list = []
    for z in range(0, shape[0], output_shape[0]):
        for y in range(0, shape[1], output_shape[1]):
            for x in range(0, shape[2], output_shape[2]):
                in_list.append([z, y, x])
    return in_list


# NOTE this will not cover the whole volume
def _offset_list_with_shift(shape, output_shape, shift):
    in_list = []
    for z in range(0, shape[0], output_shape[0]):
        for y in range(0, shape[1], output_shape[1]):
            for x in range(0, shape[2], output_shape[2]):
                in_list.append([min(z + shift[0], shape[0]),
                                min(y + shift[1], shape[1]),
                                min(x + shift[2], shape[2])])
    return in_list


# this returns the offsets for the given output blocks.
# blocks are padded on the fly during inference if necessary
def get_offset_lists(shape,
                     gpu_list,
                     save_folder,
                     output_shape,
                     randomize=False,
                     shift=None):
    in_list = _offset_list(shape, output_shape) if shift is None else\
            _offset_list_with_shift(shape, output_shape, shift)
    if randomize:
        shuffle(in_list)

    n_splits = len(gpu_list)
    out_list = [in_list[i::n_splits] for i in range(n_splits)]

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    for ii, olist in enumerate(out_list):
        list_name = os.path.join(save_folder, 'list_gpu_%i.json' % gpu_list[ii])
        with open(list_name, 'w') as f:
            json.dump(olist, f)





# this returns the offsets for the given output blocks and bounding box.
# blocks are padded on the fly during inference if necessary
def get_offset_lists_with_bb(shape,
                             gpu_list,
                             save_folder,
                             output_shape,
                             bb_start,
                             bb_stop,
                             randomize=False):

    # zap the bounding box to grid defined by out_blocks
    bb_start_c = [(bbs // outs) * outs for bbs, outs in zip(bb_start, output_shape)]
    bb_stop_c = [(bbs // outs + 1) * outs for bbs, outs in zip(bb_stop, output_shape)]

    in_list = []
    for z in range(bb_start_c[0], bb_stop_c[0], output_shape[0]):
        for y in range(bb_start_c[1], bb_stop_c[1], output_shape[1]):
            for x in range(bb_start_c[2], bb_stop_c[2], output_shape[2]):
                in_list.append([z, y, x])

    if randomize:
        shuffle(in_list)

    n_splits = len(gpu_list)
    out_list = [in_list[i::n_splits] for i in range(n_splits)]

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    for ii, olist in enumerate(out_list):
        list_name = os.path.join(save_folder, 'list_gpu_%i.json' % gpu_list[ii])
        with open(list_name, 'w') as f:
            json.dump(olist, f)


# this returns the offsets for the given output blocks.
# blocks are padded on the fly in the inference if necessary
def offset_list_from_precomputed(input_list,
                                 gpu_list,
                                 save_folder,
                                 list_name_extension='',
                                 randomize=False):

    if isinstance(input_list, str):
        with open(input_list, 'r') as f:
            input_list = json.load(f)
    else:
        assert isinstance(input_list, list)

    if randomize:
        shuffle(input_list)

    n_splits = len(gpu_list)
    out_list = [input_list[i::n_splits] for i in range(n_splits)]

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    print("Original len", len(input_list))
    for ii, olist in enumerate(out_list):
        list_name = os.path.join(save_folder, 'list_gpu_{0:}{1:}.json'.format(gpu_list[ii], list_name_extension))
        print("Dumping list number", ii, "of len", len(olist))
        with open(list_name, 'w') as f:
            json.dump(olist, f)
