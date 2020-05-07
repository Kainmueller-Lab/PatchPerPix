import os

import numpy as np
import h5py
import zarr
from numcodecs import Blosc
from skimage import io
import argparse


def zarr2hdf(zarr_file, hdf_file=None, zarr_key=[], hdf_key=[], dtype=None):

    zf = zarr.open(zarr_file, mode='r')

    if hdf_file is None:
        hdf_file = zarr_file[:-5] + '.hdf'

    hf = h5py.File(hdf_file, 'a')

    for i, zk in enumerate(zarr_key):

        array = np.asarray(zf[zk])
        print(len(hdf_key))

        if len(hdf_key) > i:
            hk = hdf_key[i]
        else:
            hk = zk

        # types not handled by fiji: float16
        if dtype is None:
            dtype = array.dtype
        if array.dtype == np.float16:
            dtype = np.float32

        hf.create_dataset(
            hk,
            data=array.astype(dtype),
            dtype=dtype,
            compression='gzip'
        )

    hf.close()


def hdf2zarr(hdf_file, zarr_file=None, hdf_key=[], zarr_key=[], chunksize=None):

    hf = h5py.File(hdf_file, 'r')

    if zarr_file is None:
        zarr_file = hdf_file[:-4] + '.zarr'

    zf = zarr.open(zarr_file, mode='a')
    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

    for i, hk in enumerate(hdf_key):

        array = np.asarray(hf[hk])

        if len(zarr_key) > i:
            zk = zarr_key[i]
        else:
            zk = hk

        zf.create_dataset(
            zk,
            data=array,
            shape=array.shape,
            compressor=compressor,
            dtype=array.dtype,
            chunks=chunksize
        )

    hf.close()


def arr2image(array, image_file):

    if len(array.shape) == 2:
        io.imsave(image_file, array.astype(np.uint32))

    else:
        io.imsave(image_file, np.max(array, axis=0).astype(np.uint32))


def hdf2npy(hdf_file, hdf_key, npy_file=None):
    if npy_file is None:
        npy_file = os.path.splitext(hdf_file)[0] + ".npy"

    with h5py.File(hdf_file, 'r') as f:
        npy_file = os.path.splitext(npy_file)[0]
        for k in hdf_key:
            array = f[k]
            print("{}".format(array.shape))
            np.save(npy_file + k.replace("/","_") + ".npy", array)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file', type=str, dest='in_file',
                        help='input file', required=True)
    parser.add_argument('--out-file', type=str, dest='out_file',
                        help='output file')
    parser.add_argument('--in-key', type=str, dest='in_key',
                        help='input key', action='append')
    parser.add_argument('--out-key', type=str, dest='out_key',
                        help='output key', action='append')
    parser.add_argument('--dtype', type=str, dest='dtype',
                        help='output dtype')
    parser.add_argument('-c', '--chunksize', type=int,
                        help='zarr chunk size', action='append')
    args = parser.parse_args()
    print(args)

    if args.in_file.endswith(".zarr"):
        if args.out_file is None or args.out_file.endswith(".hdf"):
            zarr2hdf(args.in_file, hdf_file=args.out_file,
                     zarr_key=args.in_key, hdf_key=args.out_key,
                     dtype=args.dtype)
        else:
            print("unsupported output file format")
    elif args.in_file.endswith(".hdf"):
        if args.out_file is None:
            print("please specify output file (zarr/npy)")
        elif args.out_file.endswith(".zarr"):
            hdf2zarr(args.in_file, zarr_file=args.out_file,
                     hdf_key=args.in_key, zarr_key=args.out_key,
                     chunksize=args.chunksize)
        elif args.out_file.endswith(".npy"):
            hdf2npy(args.in_file, args.in_key, npy_file=args.out_file)
        else:
            print("unsupported output file format")
    else:
        print("unsupported input file format")


if __name__ == "__main__":
    main()
