import logging
import os
import toml
import zarr

import h5py
import numpy as np
import torch

from . import torch_model
from PatchPerPix.visualize import visualize_patches

logger = logging.getLogger(__name__)


def decode_sample(config, model, sample, device):
    batch_size = config['decode_batch_size']
    code_units = config['code_units']
    patchshape = config['patchshape']
    if type(patchshape) != np.ndarray:
        patchshape = np.array(patchshape)
    patchshape = patchshape[patchshape > 1]

    # load data depending on prediction.output_format and prediction.aff_key
    if "zarr" in config['output_format']:
        pred_code = np.array(zarr.open(sample, 'r')[config['code_key']])
        pred_numinstfg = np.array(zarr.open(sample, 'r')[config.get('numinst_key', config.get('fg_key'))])
    else:
        raise NotImplementedError("invalid input format")

    # check if fg is numinst with one channel per number instances [0,1,..]
    # heads up: assuming probabilities for numinst [0, 1, 2] in this order!
    if pred_numinstfg.shape[0] > 1:
        pred_fg = np.array(pred_numinstfg[0] < 0.1).astype(np.uint8)
    else:
        pred_fg = (pred_numinstfg >= config['fg_thresh']).astype(np.uint8)
        pred_fg = np.squeeze(pred_fg)

    fg_coords = np.transpose(np.nonzero(pred_fg))
    num_batches = int(np.ceil(fg_coords.shape[0] / float(batch_size)))
    logger.info("processing %i batches", num_batches)

    output = np.zeros((np.prod(patchshape),) + pred_fg.shape, dtype=np.float32)

    for idx, b in enumerate(range(0, len(fg_coords), batch_size)):
        fg_coords_batched = fg_coords[b:b + batch_size]
        fg_coords_batched = [(slice(None),) + tuple(
            [slice(i, i + 1) for i in fg_coord])
                             for fg_coord in fg_coords_batched]
        pred_code_batched = [pred_code[fg_coord].reshape((1, code_units))
                             for fg_coord in fg_coords_batched]
        logger.info(
            '%s/%s: in decode sample: %s',
            idx, num_batches, pred_code_batched[0].shape)
        predictions = model.decoder(
            torch.as_tensor(
                np.stack(pred_code_batched, axis=0).astype(dtype=np.float32),
                device=device))

        logger.info("%s %s", predictions.size(), len(fg_coords_batched))
        for idx, fg_coord in enumerate(fg_coords_batched):
            prediction = predictions[idx].cpu().detach().numpy()
            output[fg_coord] = np.reshape(
                prediction, (np.prod(prediction.shape), 1, 1, 1)
            )
    return output


def decode(**config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = torch_model.UnetModelWrapper(config, device, 0)
    model.eval()
    try:
        model = model.to(device)
    except RuntimeError as e:
        raise RuntimeError(
            "Failed to move model to device. If you are using a child process "
            "to run your model, maybe you already initialized CUDA by sending "
            "your model to device in the main process."
        ) from e

    checkpoint = torch.load(config["checkpoint_file"], map_location=device)
    if config.get("use_swa"):
        logger.info("loading swa checkpoint")
        model = torch.optim.swa_utils.AveragedModel(model)
        model.load_state_dict(checkpoint["swa_model_state_dict"])
    else: # "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    for idx, sample in enumerate(config['samples']):
        logger.info("decoding sample: %s (%s/%s)", sample, idx, len(config['samples']))
        # decode each sample
        prediction = decode_sample(config, model, sample, device)

        # save prediction
        sample_name = os.path.basename(sample).split('.')[0]
        outfn = os.path.join(config['output_folder'],
                             sample_name + '.' + config['output_format'])
        mode = 'a' if os.path.exists(outfn) else 'w'
        if config['output_format'] == 'zarr':
            zf = zarr.open(outfn, mode=mode)
            zf.create(config['aff_key'],
                      shape=prediction.shape,
                      dtype=np.float16)
            zf[config['aff_key']].attrs['offset'] = [0] * len(config['voxel_size'])
            zf[config['aff_key']].attrs['resolution'] = config['voxel_size']
            zf[config['aff_key']][:] = prediction

        elif config['output_format'] == 'hdf':
            outf = h5py.File(outfn, mode)
            outf.create_dataset(
                config['aff_key'],
                data=prediction,
                compression='gzip'
            )
        else:
            raise NotImplementedError

        # visualize patches if given
        if config.get('show_patches'):
            if sample_name in config.get('samples_to_visualize', []):
                outfn_patched = os.path.join(config['output_folder'], "vis",
                                             sample_name + '.hdf')
                os.makedirs(os.path.dirname(outfn_patched), exist_ok=True)
                out_key = config['aff_key'] + '_patched'
                _ = visualize_patches(prediction, config['patchshape'],
                                      out_file=outfn_patched,
                                      out_key=out_key)
