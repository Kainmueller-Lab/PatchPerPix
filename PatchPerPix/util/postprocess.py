import numpy as np
import argparse
import colorcet as cc
from scipy import ndimage
from skimage.morphology import skeletonize_3d
import zarr
import h5py
import os
import nrrd


def replace(array, old_values, new_values):
    values_map = np.arange(int(array.max() + 1), dtype=new_values.dtype)
    values_map[old_values] = new_values

    return values_map[array]


def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    return np.array([int(hex[i:i + 2], 16) for i in (0, 2, 4)])


def remove_small_components(array, compsize=5):

    labels, counts = np.unique(array, return_counts=True)
    print('labels: ', labels)
    print('counts: ', counts)
    small_labels = labels[counts <= compsize]

    array = replace(
        array,
        np.array(small_labels),
        np.array([0] * len(small_labels))
    )
    return array


def relabel(array, start=None):

    labels = np.unique(array)
    relabeled = np.zeros_like(array)
    cnt = start if start is not None else 1
    print('start relabeling at ', cnt)

    for label in labels:
        if label == 0:
            continue
        relabeled[array == label] = cnt
        cnt += 1

    return relabeled


def color(src, colormap=None):

    labels = np.unique(src)
    colored = np.stack(
        [np.zeros_like(src), np.zeros_like(src), np.zeros_like(src)],
        axis=-1)

    print(labels)

    for i, label in enumerate(labels):
        if label == 0:
            continue
        if colormap == 'glasbey':
            label_color = hex_to_rgb(cc.glasbey_light[i])
        else:
            label_color = np.random.randint(0, 255, 3)
        idx = src == label
        colored[idx, :] = label_color

    return colored


def postprocess_instances(samples, output_folder, **kwargs):
    comp_thresh = kwargs['remove_small_comps']

    for sample in samples:
        inf = h5py.File(sample, "a")

        inst = np.array(inf[kwargs['res_key']])
        labels, counts = np.unique(inst, return_counts=True)
        small_labels = labels[counts <= comp_thresh]
        inst_cleaned = replace(
            inst,
            np.array(small_labels),
            np.array([0] * len(small_labels))
        )
        inst_cleaned = relabel(inst_cleaned)
        if np.max(inst_cleaned) < 65535:
            dtype = np.uint16
        else:
            dtype = np.uint32

        new_key = kwargs['res_key'] + ("_rm_%s" % comp_thresh)
        inf.create_dataset(
            new_key,
            data=inst_cleaned.astype(dtype),
            dtype=dtype,
            compression='gzip'
        )

        if kwargs.get('export_skeleton_nrrds', False):
            sample_name = os.path.basename(sample).split(".")[0]
            labels = np.unique(inst_cleaned)
            labels = labels[labels > 0]

            for label in labels:
                mask = inst_cleaned == label
                mask = (skeletonize_3d(mask) > 0).astype(np.uint8)
                # check if transpose necessary?
                mask = np.transpose(mask, (2, 1, 0))
                nrrd.write(
                    os.path.join(
                        output_folder, sample_name + ("_%i.nrrd" % label)),
                    mask
                )


def postprocess_fg(samples, output_folder, **kwargs):

    comp_thresh = kwargs['remove_small_comps']
    for sample in samples:
        outfn = os.path.join(output_folder, os.path.basename(
            sample)).replace('.zarr', '.hdf')
        zin = zarr.open(sample, mode='r')
        hout = h5py.File(outfn, 'w')

        # get foreground mask
        pred = np.array(zin[kwargs['fg_key']])
        pred = pred > kwargs['fg_threshold']
        pred, _ = ndimage.label(pred, np.ones((3, 3, 3)))

        if kwargs.get('max_distance_to_fg', 0) > 0:

            # check how far small components are from big ones
            labels, counts = np.unique(pred, return_counts=True)
            print('labels: ', labels)
            print('counts: ', counts)
            small_labels = labels[counts <= comp_thresh]
            big_labels = labels[counts > comp_thresh]
            pred_cleaned = replace(
                pred,
                np.array(small_labels),
                np.array([0] * len(small_labels))
            )
            small_label_mask = replace(
                pred,
                np.array(big_labels),
                np.array([0] * len(big_labels))
            )
            dist = ndimage.distance_transform_edt(np.logical_not(pred_cleaned))
            idx_dist = (dist[small_label_mask > 0] >= kwargs[
                'max_distance_to_fg']).astype(np.uint16)
            idx_label = small_label_mask[small_label_mask > 0].astype(np.uint16)
            idx = np.stack([idx_dist, idx_label], axis=0)

            rm_label = []
            for small_label in small_labels:
                if np.all(idx[0][idx[1] == small_label]):
                    rm_label.append(small_label)

            pred = replace(
                pred,
                np.array(rm_label),
                np.array([0] * len(rm_label))
            )

        else:
            pred = remove_small_components(pred, kwargs['remove_small_comps'])

        pred = pred > 0

        hout.create_dataset(
            'volumes/fg',
            data=pred.astype(np.uint8),
            dtype=np.uint8,
            compression='gzip',
        )

        # create instances from connected components
        if kwargs.get('cc_instances', False):
            pred, _ = ndimage.label(pred, np.ones((3, 3, 3)))

            hout.create_dataset(
                'volumes/instances',
                data=pred.astype(np.uint16),
                dtype=np.uint16,
                compression='gzip',
            )


if __name__ == '__main__':

    import h5py
    import zarr
    from skimage import io
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str,
                        help='input file', required=True)
    parser.add_argument('--in_key', type=str, help='input key')
    parser.add_argument('--small-comps', type=int,
                        default=None, dest='small_comps',
                        help='remove components smaller than this value')
    parser.add_argument('--save-relabeled',
                        action='store_true',
                        dest='save_relabeled',
                        default=False,
                        help='If relabeled array should be stored in in_file.')
    parser.add_argument('--save-mip',
                        default=False,
                        dest='save_mip',
                        action='store_true',
                        help='If colored mip should be saved.')

    args = parser.parse_args()

    # open file
    if args.in_file.endswith(".zarr"):
        zf = zarr.open(args.in_file, mode='r')
        array = np.asarray(zf[args.in_key])
    elif args.in_file.endswith(".hdf"):
        hf = h5py.File(args.in_file, 'r')
        array = np.asarray(hf[args.in_key])
        hf.close()
    else:
        print("unsupported input file format")
        raise NotImplementedError

    # remove small components
    if args.small_comps is not None:
        array = remove_small_components(array, args.small_comps)

    relabeled = relabel(array)
    if args.save_relabeled:

        if args.in_file.endswith(".zarr"):
            zf = zarr.open(args.in_file, mode='r')
            zf.create_dataset(
                args.in_key + '_rlbd_rm_' + str(args.small_comps),
                data=relabeled,
                compression='gzip',
            )

        elif args.in_file.endswith(".hdf"):
            hf = h5py.File(args.in_file, 'a')
            hf.create_dataset(
                args.in_key + '_rlbd_rm_' + str(args.small_comps),
                data=relabeled,
                compression='gzip',
            )

    if args.save_mip:

        colored = color(np.max(array, axis=0))
        name = os.path.join(
            os.path.dirname(args.in_file),
            os.path.basename(args.in_file).split('.')[0] + '_' + str(
                args.small_comps) + '.png')
        print('Saving ', name)
        io.imsave(name, colored.astype(np.uint8))

    """
    # if gt...
    with h5py.File(gt, 'r') as hdf:

        array = np.asarray(hdf['BJD_101E06_AE_01-20171128_61_E3/gt'])
        array = np.max(array, axis=0)
        colored = color(np.max(array, axis=0))
        io.imsave(sample[:-4] + '_gt.png', colored.astype(np.uint8))
    """
