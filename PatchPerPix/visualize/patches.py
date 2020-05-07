import numpy as np
import h5py
import zarr
import argparse
from skimage import io
import logging
import scipy.special

logger = logging.getLogger(__name__)


def reshape_affinities(in_array, patchshape, selected=None):
    if type(patchshape) != np.ndarray:
        patchshape = np.array(patchshape)
    patchshape = patchshape[patchshape > 1]

    if selected is not None:
        with h5py.File(selected, 'r') as f:
            selected = np.array(f['selected'])

    if patchshape.shape[0] == 2:
        py, px = patchshape
        logger.debug("%s %s %s %s", in_array.dtype,
                     np.min(in_array), np.max(in_array), in_array.shape)
        if selected is not None:
            patched = np.zeros(
                (3, in_array.shape[1] * py, in_array.shape[2] * px),
                dtype=in_array.dtype)
        else:
            patched = np.zeros(
                (1, in_array.shape[1] * py, in_array.shape[2] * px),
                dtype=in_array.dtype)
        logger.info('transforming affs shape %s into %s',
                    in_array.shape, patched.shape)
        for y in range(in_array.shape[1]):
            logger.debug('processing %i / %i', y, in_array.shape[1])
            for x in range(in_array.shape[2]):
                p = in_array[:, y, x]
                p.shape = patchshape
                p[0, :] = 1
                p[:, 0] = 1
                if selected is None:
                    patched[0,
                            y * py:(y + 1) * py,
                            x * px:(x + 1) * px] = p
                elif selected.shape[0] == 1:
                    if selected[0, y, x]:
                        patched[0,
                                y * py:(y + 1) * py,
                                x * px:(x + 1) * px] = p
                    else:
                        patched[2,
                                y * py:(y + 1) * py,
                                x * px:(x + 1) * px] = p
                else:
                    if selected[0, y, x]:
                        patched[0,
                                y * py:(y + 1) * py,
                                x * px:(x + 1) * px] = p
                    elif selected[1, y, x]:
                        patched[1,
                                y * py:(y + 1) * py,
                                x * px:(x + 1) * px] = p
                    elif selected[2, y, x]:
                        patched[2,
                                y * py:(y + 1) * py,
                                x * px:(x + 1) * px] = p
    else:
        raise NotImplementedError
    return patched


def reshape_affinities_3d(in_array, patchshape):
    if type(patchshape) != np.ndarray:
        patchshape = np.array(patchshape)
    patchshape = patchshape[patchshape > 1]

    pz, py, px = patchshape
    # create patched with mip in z
    patched = np.zeros(
        (in_array.shape[1], in_array.shape[2] * py, in_array.shape[3] * px),
        dtype=in_array.dtype
    )
    logger.info('transforming affs shape %s into %s',
                in_array.shape, patched.shape)
    for z in range(in_array.shape[1]):
        logger.debug('processing %i / %i', z, in_array.shape[1])
        for y in range(in_array.shape[2]):
            logger.debug('processing %i / %i', y, in_array.shape[2])
            for x in range(in_array.shape[3]):
                p = in_array[:, z, y, x]
                p.shape = patchshape
                p[:, 0, :] = 1
                p[:, :, 0] = 1
                patched[z, y * py:(y + 1) * py, x * px:(x + 1) * px] = np.max(
                    p, axis=0)

    return patched


def visualize_patches(
    affinities,
    patchshape,
    in_key=None,
    out_file=None,
    out_key=None,
    threshold=None,
    selected=None,
    sigmoid=False,
    store_int=False
):
    """Visualize sequential affinities by reshaping to patchshape and
    separating them visually. Affinities can be either filename or numpy array.
    """
    if type(affinities) == str:
        if affinities.endswith('.zarr'):
            inf = zarr.open(affinities, mode='r')
        elif affinities.endswith('.hdf'):
            inf = h5py.File(affinities, 'r')
        else:
            raise NotImplementedError
        assert (in_key is not None), 'Please provide a key, ' \
                                     'if affinities are loaded from file.'
    elif type(affinities) == np.ndarray:
        inf = None
    else:
        raise NotImplementedError

    # check if in_key is provided
    if in_key is not None:
        if type(in_key) not in [list, tuple]:
            in_key = list([in_key])
        if out_key is not None and type(out_key) not in [list, tuple]:
            out_key = list([out_key])
        num_to_visualize = len(in_key)
    else:
        num_to_visualize = 1
        labels = np.squeeze(affinities)

    if type(patchshape) != np.ndarray:
        patchshape = np.array(patchshape)
    patchshape = patchshape[patchshape > 1]

    for i in range(num_to_visualize):
        if inf is not None:
            labels = np.squeeze(np.array(inf[in_key[i]]))
        logger.info("file type/min/max %s %s %s", labels.dtype,
                    np.min(labels), np.max(labels))

        # CHW vs HWC
        if patchshape.shape[0] == 2:
            py, px = patchshape
            patchsize = py*px
            if labels.shape[0] != patchsize and labels.shape[-1] == patchsize:
                labels = np.ascontiguousarray(np.moveaxis(labels, -1, 0))
            # with/without sigmoid applied?
            if sigmoid:
                labels = scipy.special.expit(labels)
                logger.info("after sigmoid: file type/min/max %s %s %s",
                            labels.dtype, np.min(labels), np.max(labels))

            # reshape sequential affinities to patchshape
            patched = reshape_affinities(labels, patchshape, selected=selected)

        elif patchshape.shape[0] == 3:
            patched = reshape_affinities_3d(labels, patchshape)

        if threshold is not None:
            patched[patched < threshold] = 0

        if out_file is not None:
            if out_file.endswith('.hdf'):
                if out_key is None:
                    c_out_key = in_key[i] + '_patched'
                else:
                    if type(out_key) not in [tuple, list]:
                        out_key = list([out_key])
                    c_out_key = out_key[i]

                if patched.dtype == np.float16:
                    patched = patched.astype(np.float32)

                if store_int:
                    patched = (patched*255).astype(np.uint8)
                with h5py.File(out_file, 'a') as outf:
                    outf.create_dataset(
                        c_out_key,
                        data=patched,
                        compression='gzip'
                    )
            elif out_file.endswith('.png'):
                patched = (patched * 255).astype(np.uint8)
                io.imsave(out_file, patched)
            elif out_file.endswith('.tif'):
                patched = np.squeeze(patched)
                io.imsave(out_file, patched.astype(np.float32))
            else:
                raise NotImplementedError
        else:
            return patched
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file', type=str, dest='in_file',
                        help='input file', required=True)
    parser.add_argument('--in-key', type=str, dest='in_key',
                        help='input key', nargs='+', required=True)
    parser.add_argument('--patchshape', type=int,
                        help='patchshape', nargs='+', required=True)
    parser.add_argument('--out-file', type=str, default=None,
                        dest='out_file', help='output file')
    parser.add_argument('--out-key', type=str, default=None,
                        dest='out_key', help='output key', nargs='+')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Value to threshold predictions')
    parser.add_argument('--selected', type=str, default=None,
                        dest='selected', help='selected array')
    parser.add_argument("--sigmoid", action="store_true",
                        help='apply sigmoid')
    parser.add_argument("--store_int", action="store_true",
                        help='store patched array as uint8')

    args = parser.parse_args()

    visualize_patches(args.in_file, args.patchshape, args.in_key,
                      args.out_file, args.out_key, args.threshold,
                      selected=args.selected, sigmoid=args.sigmoid,
                      store_int=args.store_int)


if __name__ == "__main__":
    main()
