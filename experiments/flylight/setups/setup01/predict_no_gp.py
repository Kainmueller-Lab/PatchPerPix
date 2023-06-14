"""Script for a prediction worker process
"""
import argparse
import logging
import json
import os
import sys

import h5py
import numpy as np
import scipy.special
import torch
import zarr
from glob import glob
from numcodecs import Blosc

import gunpowder as gp

from . import torch_model
from PatchPerPix.util import (
    normalize)

logger = logging.getLogger(__name__)


def predict(**config):
    """Predict function used by a prediction worker process

    Args
    ----
    config
    """
    overlapping_inst = config.get('overlapping_inst')
    train_code = config.get("train_code")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = torch_model.UnetModelWrapper(
        config, device, config["checkpoint"], for_inference=True)
    if config.get("network_style", "unet").lower() == "unet":
        model.set_padding(config['val_padding'])
    try:
        model = model.to(device)
    except RuntimeError as e:
        raise RuntimeError(
            "Failed to move model to device. If you are using a child process "
            "to run your model, maybe you already initialized CUDA by sending "
            "your model to device in the main process."
        ) from e

    model.eval()
    if config.get("network_style", "unet").lower() == "unet":
        model.set_padding(config['val_padding'])
        if config['val_padding'] == 'valid':
            test_input_shape = config["test_input_shape_valid"]
        else:
            test_input_shape = config["test_input_shape_same"]
        input_shape, output_shape = model.inout_shapes(
            test_input_shape, "test_net", training=False)
        print("input/output shape val: ", input_shape, output_shape)
    else:
        input_shape = config["test_input_shape_same"]
        output_shape = config["test_input_shape_same"]
    model.eval()

    checkpoint = torch.load(config["checkpoint"], map_location=device)
    if config.get("use_swa"):
        logger.info("loading swa checkpoint")
        model = torch.optim.swa_utils.AveragedModel(model)
        model.load_state_dict(checkpoint["swa_model_state_dict"])
    else: # "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    # else:
    #     model.load_state_dict()


    logger.debug("Model: %s", model)

    # with open(os.path.join(
    #         config['input_folder'], "..", "train",
    #         config['test_net_name'] + "_config.json"),
    #           'r') as f:
    #     s = json.load(f)
    #     input_shape = s["input_shape"]
    #     output_shape = s["output_shape"]
    logger.info("input/output shape from json: %s/%s",
                 input_shape, output_shape)

    context = [(i - o) // 2 for i, o in zip(input_shape, output_shape)]
    logger.info("context: %s", context)
    assert np.all([
        ins % 2 == 0 and ous % 2 == 0
        for ins, ous in zip(input_shape, output_shape)
    ]), "input and output shape have to be even!"
    chunksize = [int(c) for c in np.asarray(output_shape) // 2]
    logger.info("chunk size %s", chunksize)

    keys_channels = {}
    raw_key = config.get('raw_key', 'volumes/raw')
    if train_code:
        code_aff_key = config.get('code_key', 'volumes/pred_code')
        keys_channels[code_aff_key] = int(config['code_units'])
    else:
        code_aff_key = config.get('aff_key', 'volumes/pred_affs')
        keys_channels[code_aff_key] = int(np.prod(config['patchshape']))
    if overlapping_inst:
        fgbg_numinst_key = config.get('numinst_key', 'volumes/pred_numinst')
        keys_channels[fgbg_numinst_key] = int(config['max_num_inst']) + 1
    else:
        fgbg_numinst_key = config.get('fg_key', 'volumes/pred_fgbg')
        keys_channels[fgbg_numinst_key] = 1

    if config['input_format'] != "hdf" and config['input_format'] != "zarr":
        raise NotImplementedError("predict node for %s not implemented yet",
                                  config['input_format'])
    # if config['input_format'] == "hdf":
    #     with h5py.File(os.path.join(config['data_folder'],
    #                                 config['sample'] + ".hdf"), 'r') as f:
    #         shape = f[raw_key].shape[1:]
    # else: # config['input_format'] == "zarr":
    #     f = zarr.open(os.path.join(config['data_folder'],
    #                                config['sample'] + ".zarr"), 'r')
    #     shape = f[raw_key].shape[1:]

    if config['output_format'] != "zarr":
        raise NotImplementedError("Please use zarr as prediction output")

    # add_partly_val = config.get("add_partly_val", False)
    # if add_partly_val:
    #     samples_partly = glob(
    #             config["val_data_partly"] + "/*." + config["input_format"])
    #     samples_partly = [
    #         os.path.splitext(os.path.basename(s))[0] for s in samples_partly]
    # else:
    #     samples_partly = []

    for sample in config['samples']:
        logger.info("Sample: %s", sample)
        fn = os.path.join(
            config['data_folder'],
            sample + "." + config['input_format'])
        if not os.path.exists(fn) and config.get("add_partly_val", False):
            fn = os.path.join(
                config['data_folder'].replace("complete", "partly"),
                sample + "." + config['input_format'])

        # if sample in samples_partly:
        #     data_folder = config["val_data_partly"]
        # else:
        #     data_folder = config["data_folder"]

        if config['input_format'] == "hdf":
            with h5py.File(fn, 'r') as in_f:
                raw = np.array(in_f[raw_key])
        else: # config['input_format'] == "zarr":
            in_f = zarr.open(fn, 'r')
            raw = in_f[raw_key]

        if raw.shape[0] != 1 and config["num_channels"] == 1:
            raw = np.reshape(raw, (1,) + raw.shape)
        shape = raw.shape[1:]
        shape_padded = [(s//os + 1) * os + 2 * c
                        for s, os, c in zip(shape, output_shape, context)]
        logger.info(f"Shape: {shape}, padded shape: {shape_padded}")

        assert np.all(
            [(s-2*c) % st == 0
             for s, st, c in zip(shape_padded, output_shape, context)]), (
            "to avoid inconsistencies in last ROI, "
            "padded shape - 2*context has to be divisible by stride/"
            "output_shape, please pad first! "
            f"(shape: {shape}, stride: {output_shape}, context: {context})")
        shifts = enumerate_shifts(shape_padded, input_shape, output_shape)
        out_zf = create_zarr_outputs(
            config, sample, shape, chunksize, keys_channels)

        raw_padded = np.pad(
            raw,
            [[0, 0]] + [[c, sp-s-c]
                        for c, s, sp in zip(context, shape, shape_padded)],
            mode='constant', constant_values=0)
        logger.info(
            "raw shape: %s, raw padded shape: %s",
            raw.shape, raw_padded.shape)

        print(shifts)
        print("batch_size: ", config["batch_size"])
        for idx in range(0, len(shifts), config['batch_size']):
            chunks_batch = []
            slices_batch = []
            for bidx in range(config['batch_size']):
                if idx+bidx >= len(shifts):
                    break
                print(idx, bidx)
                shift = shifts[idx+bidx]
                slices = tuple([slice(None)] +
                    [slice(shift[i], shift[i]+input_shape[i])
                     for i in range(len(shift))])
                logger.info(
                    "(%s/%s) | Shift: %s, Slices: %s",
                    idx, len(shifts), shift, slices)
                slices_batch.append(slices)
                raw_b = raw_padded[slices]
                # gp.IntensityScaleShift(raw, 2, -1)
                if config.get("shift_intensity", False):
                    raw_b = raw_b * 2 - 1
                chunks_batch.append(raw_b)

            logger.info("Predicting..")
            pred_code_affs, pred_fgbg_numinst = \
                model(raw=torch.as_tensor(np.stack(chunks_batch, axis=0), device=device))

            for b in range(len(chunks_batch)):
                code_affs = pred_code_affs[b].cpu().detach().numpy()
                fgbg_numinst = pred_fgbg_numinst[b].cpu().detach().numpy()

                slices = slices_batch[b]
                slices_global = [slices[0]]
                slices_local = [slice(None)]
                for sidx, slc in enumerate(slices[-len(shape):]):
                    slc_s = slc.start + context[sidx] - context[sidx]
                    slc_e = min(shape[sidx] + context[sidx],
                                slc.stop - context[sidx]) - context[sidx]
                    slices_global.append(slice(slc_s, slc_e))
                    slices_local.append(slice(0, slc_e - slc_s))

                logger.info("B%d: Slices global: %s, Slices local: %s",
                            b, slices_global, slices_local)

                out_zf[code_aff_key][tuple(slices_global)] = \
                    code_affs[tuple(slices_local)]
                out_zf[fgbg_numinst_key][tuple(slices_global)] = \
                    fgbg_numinst[tuple(slices_local)]


    # if config.get("normalization"):
    #     print("normalize input")
    #     source = normalize(
    #         source, config["normalization"]["type"],
    #         raw, config["normalization"])


def create_zarr_outputs(config, sample, shape, chunksize, keys_channels):
    # open zarr file
    fn = os.path.join(config['output_folder'], sample + '.zarr')
    logger.info("Creating output file: %s", fn)
    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
    zf = zarr.open(fn, mode='w')
    for key, ch in keys_channels.items():
        zf.create(key,
                  shape=[ch] + list(shape),
                  chunks=[ch] + list(chunksize),
                  compressor=compressor,
                  dtype=np.float16)
        zf[key].attrs['offset'] = [0, 0]
        zf[key].attrs['resolution'] = config['voxel_size']
    return zf


def enumerate_shifts(shape, input_shape, stride):
    '''Produces a sequence of shift coordinates starting at the beginning,
    progressing with ``stride``. The maximum shift coordinate in any
    dimension will be the last point inside the shift roi this dimension.'''

    logger.info(
        "enumerating possible shifts of %s in %s", stride, shape)

    shape = np.array(shape) - np.array(input_shape)
    min_shift = np.array([0] * len(shape))
    max_shift = np.array(
        [max(ms, s) for s, ms in zip(shape, min_shift)])

    shift = np.array(min_shift)
    shifts = []

    print(shape, min_shift, max_shift)
    dims = len(min_shift)

    while True:

        logger.info("adding %s", shift)
        shifts.append(np.copy(shift))

        if (shift == max_shift).all():
            break

        # count up dimensions
        for d in range(dims):

            if shift[d] >= max_shift[d]:
                if d == dims - 1:
                    break
                shift[d] = min_shift[d]
            else:
                shift[d] += stride[d]
                # snap to last possible shift, don't overshoot
                if shift[d] > max_shift[d]:
                    shift[d] = max_shift[d]
                break

    return shifts
