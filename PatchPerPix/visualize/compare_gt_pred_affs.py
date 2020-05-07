import argparse
import h5py
import numpy as np
import zarr
from skimage import io


def reshape_snapshot(filename, gt_key, pred_key, out_file, num_patches,
                     raw_key=None, show_mip=False):
    if filename.endswith("hdf"):
        f = h5py.File(filename, 'r')
    elif filename.endswith("zarr"):
        f = zarr.open(filename, 'r')
    else:
        raise NotImplementedError

    # sample num patches
    shape = f[gt_key].shape
    dim = len(shape[2:])
    if dim == 2:
        ps = int(np.sqrt(shape[1]))
    elif dim == 3:
        ps = int(np.round(np.cbrt(shape[1])))
    else:
        raise NotImplementedError
    ids = np.random.choice(range(shape[0]), size=num_patches, replace=False)
    if raw_key is None:
        dst = np.ones(
            (num_patches * ps + (num_patches - 1), 2 * ps + 1, 3),
            dtype=np.float32
        )
    else:
        dst = np.ones(
            (num_patches * ps + (num_patches - 1), 3 * ps + 2, 3),
            dtype=np.float32
        )

    for i, idx in enumerate(ids):
        gt = np.reshape(np.array(f[gt_key][idx]), [ps] * dim)
        pred = np.reshape(np.array(f[pred_key][idx]), [ps] * dim)
        if raw_key is not None:
            raw = np.squeeze(np.array(f[raw_key][idx]))
            # crop raw
            if raw.ndim > gt.ndim:
                start = (np.array(raw.shape[1:]) - np.array(gt.shape)) // 2
                slice_idx = tuple([slice(None), ] + [
                    slice(start[d], start[d] + gt.shape[d])
                    for d in range(len(start))])
                raw = raw[slice_idx]
            else:
                start = (np.array(raw.shape) - np.array(gt.shape)) // 2
                slice_idx = tuple([slice(start[d], start[d] + gt.shape[d])
                                   for d in range(len(start))])
                raw = raw[slice_idx]

        if dim == 3:
            gt = np.max(gt, axis=0)
            pred = np.max(pred, axis=0)
            if raw_key is not None:
                raw = np.max(raw, axis=1)
                print(raw.shape)
                raw = np.moveaxis(raw, 0, -1)
        print('dst / gt: ', dst.shape, gt.shape)
        src = gt.astype(np.float32)
        src = np.stack([src, src, src], axis=-1)
        dst[i * ps + i:i * ps + ps + i, :ps, :] = src
        src = pred.astype(np.float32)
        src = np.stack([src, src, src], axis=-1)
        dst[i * ps + i:i * ps + ps + i, ps + 1:ps * 2 + 1, :] = src

        if raw_key is not None:
            dst[i * ps + i:i * ps + ps + i, ps * 2 + 2:, :] = raw.astype(
                np.float32)

    print(np.min(dst), np.max(dst))
    dst = (dst * 255).astype(np.uint8)

    if out_file.endswith('.tif'):
        io.imsave(out_file, dst)

    elif out_file.endswith('.hdf'):
        with h5py.File(out_file, 'w') as f:
            f.create_dataset(
                'gt_pred_affs',
                data=dst,
                compression='gzip'
            )
    else:
        raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file', type=str, dest='in_file',
                        help='input file', required=True)
    parser.add_argument('--gt-key', type=str, default='volumes/gt_affs',
                        help='dataset gt key', dest='gt_key')
    parser.add_argument('--pred-key', type=str, default='volumes/pred_affs',
                        help='dataset pred key', dest='pred_key')
    parser.add_argument('--raw-key', type=str, default=None,
                        help='dataset raw key', dest='raw_key')
    parser.add_argument('--out-file', type=str, required=True,
                        dest='out_file', help='output file')
    parser.add_argument('--num-patches', type=int, default=6,
                        dest='num_patches',
                        help='number of patches to sample from snapshot batch'
                        )
    parser.add_argument('--show-mip', action="store_true",
                        dest='show_mip',
                        help='show maximum intensity projection for 3d '
                             'snapshots.')

    args = parser.parse_args()

    reshape_snapshot(
        args.in_file,
        args.gt_key,
        args.pred_key,
        args.out_file,
        args.num_patches
    )

    reshape_snapshot(args.in_file, args.gt_key, args.pred_key, args.out_file,
                     args.num_patches, args.raw_key, args.show_mip)


if __name__ == "__main__":
    main()
