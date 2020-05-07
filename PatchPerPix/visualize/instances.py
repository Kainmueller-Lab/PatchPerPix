import numpy as np
import zarr
from scipy import ndimage
from skimage import io
import h5py
import argparse
import colorcet as cc

from PatchPerPix.util import color, relabel


def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    return np.array([int(hex[i:i + 2], 16) for i in (0, 2, 4)])


def visualize_instances(label_fn, label_key, output_file,
                        max_axis=None, show_outline=False,
                        raw_file=None, raw_key=None):
    # read labeling
    if label_fn.endswith('.zarr'):
        inf = zarr.open(label_fn, mode='r')
    elif label_fn.endswith('.hdf'):
        inf = h5py.File(label_fn, 'r')
    else:
        raise NotImplementedError
    label = np.squeeze(np.array(inf[label_key]))
    print(label.shape, label.dtype)
    if label_fn.endswith('.hdf'):
        inf.close()
    if max_axis is not None:
        label = np.max(label, axis=max_axis)

    # read raw if given
    raw = None
    if raw_file is not None and raw_key is not None:
        if raw_file.endswith('.zarr'):
            inf = zarr.open(raw_file, mode='r')
        elif raw_file.endswith('.hdf'):
            inf = h5py.File(raw_file, 'r')
        else:
            raise NotImplementedError
        raw = np.squeeze(np.array(inf[raw_key]))
        if raw_file.endswith('.hdf'):
            raw_file.close()

    if show_outline:
        # label = label.astype(np.uint16)
        # for i in range(label.shape[0]):
        #     label[i] *= (i+1)

        labels, locations = np.unique(label, return_index=True)
        print(labels)
        locations = np.delete(locations, np.where(labels == 0))
        labels = np.delete(labels, np.where(labels == 0))
        # for different colormaps, see https://colorcet.pyviz.org/
        colormap = cc.glasbey_light
        # uncomment to choose randomly from colormap
        # colormap = np.random.choice(colormap, size=len(labels),
        #                             replace=(len(labels)>len(colormap)))

        if raw is None:
            colored = np.zeros(label.shape[1:] + (3,), dtype=np.uint8)
        else:
            # convert raw to np.uint8
            # heads up: assuming normalized raw between [0, 1]
            raw = (raw * 255).astype(np.uint8)
            colored = np.stack([raw] * 3, axis=-1)
        for i, (lbl, loc) in enumerate(zip(labels, locations)):
            if lbl == 0:
                continue
            # heads up: assuming one instance per channel
            c = np.unravel_index(loc, label.shape)[0]
            outline = ndimage.distance_transform_cdt(label[c] == lbl) == 1
            #colored[outline, :] = np.random.randint(0, 255, 3)
            colored[outline, :] = hex_to_rgb(colormap[i])
    else:
        colored = color(label)
    io.imsave(output_file, colored.astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file', type=str, dest='in_file',
                        help='input file', required=True)
    parser.add_argument('--in-key', type=str, dest='in_key',
                        help='input key', required=True)
    parser.add_argument('--out-file', type=str, dest='out_file',
                        help='output file', required=True)
    parser.add_argument('--show-outline', action="store_true",
                        dest='show_outline',
                        help='show only outline/contour of instances')
    parser.add_argument('--raw-file', type=str, dest='raw_file',
                        help='show raw file with labeling')
    parser.add_argument('--raw-key', type=str, dest='raw_key',
                        help='raw key')

    args = parser.parse_args()

    visualize_instances(args.in_file, args.in_key, args.out_file,
                        show_outline=args.show_outline,
                        raw_file=args.raw_file,
                        raw_key=args.raw_key
                        )


if __name__ == "__main__":
    main()
