import numpy as np
import tensorflow as tf
import h5py
import zarr
import argparse
import toml
import os
from skimage import io
import logging

from PatchPerPix.models import autoencoder, FastPredict
from PatchPerPix.visualize import visualize_patches

logger = logging.getLogger(__name__)


def predict_input_fn(generator, input_shape):
    def _inner_input_fn():
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=tf.float32,
            output_shapes=(tf.TensorShape(input_shape))).batch(1)
        return dataset

    return _inner_input_fn


def decoder_model_fn(features, labels, mode, params):
    if mode != tf.estimator.ModeKeys.PREDICT:
        raise RuntimeError("invalid tf estimator mode %s", mode)

    logger.info("feature tensor: %s", features)
    logger.info("label tensor: %s", labels)
    # read original autoencoder config
    autoencoder_config = toml.load(params['config'])

    is_training = False
    code = tf.reshape(features, (-1,) + params['input_shape'])
    dummy_in = tf.placeholder(
        tf.float32, [None, ] + autoencoder_config['model']['patchshape'])
    input_shape = tuple(p for p in autoencoder_config['model']['patchshape']
                        if p > 1)
    logits, _, _ = autoencoder(
        code,
        is_training=is_training,
        input_shape_squeezed=input_shape,
        only_decode=True,
        dummy_in=dummy_in,
        **autoencoder_config['model']
    )
    pred_affs = tf.sigmoid(logits, name="affinities")

    predictions = {
        "affinities": pred_affs,
    }
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


def decode_sample(code, mask, checkpoint, config_file, output_folder,
                  output_file):
    config = toml.load(config_file)
    patchshape = np.array(config['model']['patchshape'])
    batch_size = config['training']['batch_size']
    code_units = config['model']['code_units']

    # create decoder config
    params = {
        'config': config_file,
        'input_shape': tuple([config['model']['code_units']]),
    }

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    config = tf.estimator.RunConfig(
        model_dir=output_folder,
        session_config=sess_config)

    # init tf estimator and fast predict
    decoder = tf.estimator.Estimator(
        model_fn=decoder_model_fn, params=params, config=config)
    decoder = FastPredict(decoder, predict_input_fn, checkpoint, params)
    # create output array
    decoded = np.zeros((np.prod(patchshape),) + mask.shape, dtype=np.float32)

    fg_coords = np.transpose(np.nonzero(mask))
    num_batches = int(np.ceil(fg_coords.shape[0] / float(batch_size)))
    logger.info("processing %i batches", num_batches)

    for b in range(0, len(fg_coords), batch_size):
        fg_coords_batched = fg_coords[b:b + batch_size]
        fg_coords_batched = [(slice(None),) + tuple(
            [slice(i, i + 1) for i in fg_coord])
                             for fg_coord in fg_coords_batched]
        pred_code_batched = [code[fg_coord].reshape((1, code_units))
                             for fg_coord in fg_coords_batched]
        if len(pred_code_batched) < batch_size:
            pred_code_batched = pred_code_batched + ([np.zeros(
                (1, code_units))] * (batch_size - len(pred_code_batched)))
        print('in decode sample: ', pred_code_batched[0].shape)
        predictions = decoder.predict(pred_code_batched)

        for idx, fg_coord in enumerate(fg_coords_batched):
            prediction = predictions[idx]
            decoded[fg_coord] = np.reshape(
                prediction['affinities'],
                (np.prod(prediction['affinities'].shape), 1, 1)
            )
    _ = visualize_patches(decoded, patchshape, out_file=output_file,
                          out_key='volumes/pred_affs_decoded_patched')


def decode(
    code_fn,
    checkpoint,
    config,
    mask_fn=None,
    in_key=None,
    mask_key=None,
    out_file=None,
    out_key=None
):
    """Decode and visualize code for given filename and given autoencoder
    checkpoint. Additionally, foreground mask must be provided --> only
    mask_key if it is in same file as code.
    """
    # get code and mask
    assert (mask_fn is not None or mask_key is not None), \
        'Please provide fg mask or where to read it from!'
    if code_fn.endswith('.zarr'):
        inf = zarr.open(code_fn, mode='r')
    elif code_fn.endswith('.hdf'):
        inf = h5py.File(code_fn, 'r')
    else:
        raise NotImplementedError
    assert (in_key is not None), 'Please provide a code key!'
    code = np.squeeze(np.array(inf[in_key]))
    if mask_fn is None or mask_fn == code_fn:
        mask = np.squeeze(np.array(inf[mask_key]))
        if code_fn.endswith('.hdf'):
            inf.close()
    elif mask_fn is not None and code_fn != code:
        if mask_fn.endswith('.zarr'):
            inf = zarr.open(mask_fn, mode='r')
        elif mask_fn.endswith('.hdf'):
            inf = h5py.File(mask_fn, 'r')
        else:
            raise NotImplementedError
        mask = np.squeeze(np.array(inf[mask_key]))
        if code_fn.endswith('.hdf'):
            inf.close()
    else:
        # sorry, probably I messed sth up
        raise NotImplementedError

    # get output folder
    if out_file is not None:
        output_folder = os.path.dirname(out_file)
        outfn = out_file
    else:
        output_folder = os.path.dirname(code_fn)
        outfn = os.path.join(
            os.path.dirname(code_fn),
            os.path.basename(code_fn).split('.')[0] + '_patched.tif'
        )

    # todo: set fg thresholds
    # heads up: if mask.shape[0] == 3 assuming numinst with [0, 1, 2]
    logger.info('mask: ', mask.shape, mask.dtype, np.max(mask))
    if mask.shape[0] == 3:
        mask = np.any(np.array([mask[i] >= 0.5
            for i in range(1, mask.shape[0])
        ]), axis=0).astype(np.uint8)
    elif mask.dtype in [np.float16, np.float32, np.float64]:
        mask = (mask > 0.9).astype(np.uint8)
    else:
        mask = (mask == 1).astype(np.uint8)

    # decode
    decode_sample(code, mask, checkpoint, config, output_folder, outfn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file', type=str, dest='in_file',
                        help='input file', required=True)
    parser.add_argument('--in-key', type=str, dest='in_key',
                        help='input key', required=True)
    parser.add_argument('--mask-key', type=str, dest='mask_key',
                        help='mask key', required=True)
    parser.add_argument('--checkpoint', type=str, dest='checkpoint',
                        help='autoencoder checkpooint', required=True)
    parser.add_argument('-c', '--config', type=str, dest='config',
                        help='autoencoder config', required=True)
    parser.add_argument('--out-file', type=str, default=None,
                        dest='out_file', help='output file')
    parser.add_argument('--out-key', type=str, default=None,
                        dest='out_key', help='output key')
    args = parser.parse_args()

    decode(args.in_file, args.checkpoint, args.config, None,
           args.in_key, args.mask_key, args.out_file, args.out_key)


if __name__ == "__main__":
    main()
