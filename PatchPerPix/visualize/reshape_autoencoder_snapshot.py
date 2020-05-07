import argparse
import os
import sys

import h5py
import numpy as np
import zarr


def reshape_snapshot(filename, key, out_file, show_mip=False):
    if filename.endswith("hdf"):
        f = h5py.File(filename, 'r')
    elif filename.endswith("zarr"):
        f = zarr.open(filename, 'r')
    else:
        raise NotImplementedError

    assert out_file.endswith("hdf"), "only hdf output supported atm"

    arr = np.array(f[key])
    print(arr.shape, arr.shape[1], int(np.round(np.cbrt(arr.shape[1]))))
    if len(arr.shape[2:]) == 2:
        arr.shape = (arr.shape[0],
                     int(np.sqrt(arr.shape[1])),
                     int(np.sqrt(arr.shape[1])))
    elif len(arr.shape[2:]) == 3:
        arr.shape = (arr.shape[0],
                     int(np.round(np.cbrt(arr.shape[1]))),
                     int(np.round(np.cbrt(arr.shape[1]))),
                     int(np.round(np.cbrt(arr.shape[1]))),
                     )
        if show_mip:
            arr = np.max(arr, axis=1)
    with h5py.File(out_file, 'w') as f:
        f.create_dataset(
            key,
            data=arr,
            compression='gzip'
        )
    if filename.endswith("hdf"):
        f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file', type=str, dest='in_file',
                        help='input file', required=True)
    parser.add_argument('--key', type=str, default='volumes/pred_affs',
                        help='dataset key')
    parser.add_argument('--out-file', type=str, required=True,
                        dest='out_file', help='output file')
    parser.add_argument('--show-mip', action="store_true",
                        dest='show_mip',
                        help='show maximum intensity projection for 3d '
                             'snapshots.')

    args = parser.parse_args()

    reshape_snapshot(args.in_file, args.key, args.out_file, args.show_mip)


if __name__ == "__main__":
    main()
