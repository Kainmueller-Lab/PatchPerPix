import numpy as np
import zarr
import h5py
import argparse
import os
from glob import glob
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from skimage import io
from skimage.morphology import skeletonize_3d
from scipy import ndimage

import neurolight.gunpowder as nl

if __package__ is None or __package__ == '':
    from PatchPerPix.util import remove_small_components
else:
    from ..util import remove_small_components


def get_affinity_function(shape, overlapping_inst):
    if overlapping_inst and len(shape) == 3:
        seg_to_aff_fun = nl.add_affinities.seg_to_affgraph_2d_multi
    elif len(shape) == 2:
        seg_to_aff_fun = nl.add_affinities.seg_to_affgraph_2d
    elif overlapping_inst and len(shape) == 4:
        seg_to_aff_fun = nl.add_affinities.seg_to_affgraph_3d_multi
    else:
        seg_to_aff_fun = nl.add_affinities.seg_to_affgraph

    return seg_to_aff_fun


def evaluate_patch(
    prediction_fn,
    label_fn,
    **kwargs
):
    # read prediction affinities
    if prediction_fn.endswith('zarr'):
        inf = zarr.open(prediction_fn, mode='r')
        pred_affs = np.squeeze(np.array(inf[kwargs['aff_key']]))
    elif prediction_fn.endswith('hdf'):
        with h5py.File(prediction_fn, 'r') as inf:
            pred_affs = np.squeeze(np.array(inf[kwargs['aff_key']]))
    else:
        raise NotImplementedError

    # read gt labeling
    if label_fn.endswith('zarr'):
        inf = zarr.open(label_fn, mode='r')
        labels = np.squeeze(np.array(inf[kwargs['sample_gt_key']]))
    elif label_fn.endswith('hdf'):
        with h5py.File(label_fn, 'r') as inf:
            labels = np.squeeze(np.array(inf[kwargs['sample_gt_key']]))
    else:
        raise NotImplementedError

    # create neighborhood
    neighborhood = []
    psH = np.array(kwargs['patchshape']) // 2
    if len(pred_affs.shape) == 3:
        for i in range(-psH[1], psH[1] + 1):
            for j in range(-psH[2], psH[2] + 1):
                neighborhood.append([i, j])
    elif len(pred_affs.shape) == 4:
        for z in range(-psH[0], psH[0] + 1):
            for y in range(-psH[1], psH[1] + 1):
                for x in range(-psH[2], psH[2] + 1):
                    neighborhood.append([z, y, x])
    else:
        raise NotImplementedError
    neighborhood = np.array(neighborhood)

    # create gt affinities
    aff_fun = get_affinity_function(labels.shape, kwargs['overlapping_inst'])
    gt_affs = aff_fun(labels, neighborhood)
    numinst = np.sum(labels > 0, axis=0)
    pred_affs[:, numinst > 1] = 0
    gt_affs[:, numinst > 1] = 0
    num_gt = np.sum(gt_affs > 0)
    mid = np.prod(kwargs['patchshape']) // 2

    threshs = kwargs['threshs']
    metrics = {}
    for thresh in threshs:
        binarized = pred_affs > thresh
        num_pred = np.sum(binarized)

        if kwargs['return_patches']:
            tp = np.logical_and(binarized, gt_affs)
            print('tp: ', np.sum(tp))
            fp = np.logical_and(binarized, np.logical_not(tp))
            fn = np.logical_and(gt_affs, np.logical_not(tp))
            tp_patch_coords = np.transpose(np.nonzero(tp[mid]))
            print(mid, np.sum(tp[mid]))
            mse = np.abs(gt_affs - pred_affs)
            print(mse.shape, mse.dtype, np.min(mse), np.max(mse))
            metrics[str(round(thresh, 2)).replace('.', '_')] = {
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'mse': mse,
                'tp_patch_coords': tp_patch_coords,
            }

        else:
            tp = np.sum(np.logical_and(binarized, gt_affs))
            precision = tp / float(num_pred)
            recall = tp / float(num_gt)
            f1 = 2 * (precision * recall) / (precision + recall)
            mse = np.mean(np.abs(gt_affs - pred_affs) ** 2)

            metrics[str(round(thresh, 2)).replace('.', '_')] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'mse': mse,
            }

    return metrics


def evaluate_numinst(
    prediction_fn,
    label_fn,
    **kwargs
):
    # read prediction affinities
    if prediction_fn.endswith('zarr'):
        inf = zarr.open(prediction_fn, mode='r')
        pred_numinst = np.squeeze(np.array(inf[kwargs['fg_key']]))
    elif prediction_fn.endswith('hdf'):
        with h5py.File(prediction_fn, 'r') as inf:
            pred_numinst = np.squeeze(np.array(inf[kwargs['fg_key']]))
    else:
        raise NotImplementedError

    # read gt labeling
    if label_fn.endswith('zarr'):
        inf = zarr.open(label_fn, mode='r')
        labels = np.squeeze(np.array(inf[kwargs['sample_gt_key']]))
    elif label_fn.endswith('hdf'):
        with h5py.File(label_fn, 'r') as inf:
            labels = np.squeeze(np.array(inf[kwargs['sample_gt_key']]))
    else:
        raise NotImplementedError

    gt_numinst = np.sum(labels > 0, axis=0).astype(np.uint8)
    pred_numinst = np.argmax(pred_numinst, axis=0).astype(np.uint8)

    metrics = {}
    inst = np.unique(gt_numinst)
    for i in inst:
        gt_mask = gt_numinst == i
        pred_mask = pred_numinst == i
        num_gt = np.sum(gt_mask)
        num_pred = np.sum(pred_mask)

        if kwargs.get('evaluate_skeleton_coverage') and i > 0:
            gt_mask_skeletonized = skeletonize_3d(gt_mask) > 0
            pred_mask_skeletonized = skeletonize_3d(pred_mask) > 0
            num_gt_skeletonized = np.sum(gt_mask_skeletonized)
            num_pred_skeletonized = np.sum(pred_mask_skeletonized)
            tp = float(np.sum(np.logical_and(gt_mask_skeletonized, pred_mask)))

            if num_pred_skeletonized > 0 and tp > 0:
                precision = tp / (tp + float(np.sum(
                    np.logical_and(
                        np.logical_not(gt_mask), pred_mask_skeletonized))))

                recall = tp / (tp + float(np.sum(
                    np.logical_and(
                        gt_mask_skeletonized,
                        np.logical_not(pred_mask)
                    )
                )))
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                precision = 0
                recall = 0
                f1 = 0

        else:
            tp = np.sum(np.logical_and(gt_mask, pred_mask))
            print('numinst: %i, num gt: %i, num pred: %i, num tp: %i'
                  % (i, num_gt, num_pred, tp))

            if num_pred > 0 and tp > 0:
                precision = tp / float(num_pred)
                recall = tp / float(num_gt)
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                precision = 0
                recall = 0
                f1 = 0

        metrics[str(i)] = {
            'num_gt': num_gt,
            'num_pred': num_pred,
            'num_tp': tp,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    if kwargs.get('output_folder', None) is not None:
        outfn = os.path.join(
            kwargs['output_folder'],
            os.path.basename(prediction_fn).split('.')[0] + '.png'
        )
        io.imsave(outfn, np.max(pred_numinst, axis=0).astype(np.uint8))

    return metrics


def evaluate_fg(
    prediction_fn,
    label_fn,
    **kwargs
):
    # read prediction affinities
    if prediction_fn.endswith('zarr'):
        inf = zarr.open(prediction_fn, mode='r')
        pred_numinst = np.squeeze(np.array(inf[kwargs['fg_key']]))
    elif prediction_fn.endswith('hdf'):
        with h5py.File(prediction_fn, 'r') as inf:
            pred_numinst = np.squeeze(np.array(inf[kwargs['fg_key']]))
    else:
        raise NotImplementedError

    # read gt labeling
    if label_fn.endswith('zarr'):
        inf = zarr.open(label_fn, mode='r')
        labels = np.squeeze(np.array(inf[kwargs['sample_gt_key']]))
    elif label_fn.endswith('hdf'):
        with h5py.File(label_fn, 'r') as inf:
            labels = np.squeeze(np.array(inf[kwargs['sample_gt_key']]))
    else:
        raise NotImplementedError

    gt_mask = np.max(labels > 0, axis=0).astype(np.uint8)

    metrics = {}
    threshs = kwargs.get('threshs', [0.9])
    rm_comps = kwargs.get('remove_small_comps', [0])
    rm_comps = sorted(rm_comps)
    print('rm_comps: ', rm_comps)

    for thresh in threshs:
        thresh_key = str(round(thresh, 2)).replace('.', '_')
        metrics[thresh_key] = {}
        pred_mask = (pred_numinst > thresh).astype(np.uint8)
        for rm_comp in rm_comps:
            if rm_comp > 0:
                pred_mask, _ = ndimage.label(pred_mask, np.ones((3, 3, 3)))
                pred_mask = remove_small_components(pred_mask, rm_comp)
                pred_mask = pred_mask > 0
            num_gt = np.sum(gt_mask)
            num_pred = np.sum(pred_mask)

            if kwargs.get('evaluate_skeleton_coverage'):
                gt_mask_skeletonized = skeletonize_3d(gt_mask) > 0
                pred_mask_skeletonized = skeletonize_3d(pred_mask) > 0
                tp = float(np.sum(np.logical_and(gt_mask_skeletonized, pred_mask)))

                if np.sum(pred_mask_skeletonized) > 0 and tp > 0:
                    precision = tp / (tp + float(np.sum(
                        np.logical_and(
                            np.logical_not(gt_mask), pred_mask_skeletonized
                        ))))
                    recall = tp / (tp + float(np.sum(
                        np.logical_and(
                            gt_mask_skeletonized,
                            np.logical_not(pred_mask)
                        ))))
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    precision = 0
                    recall = 0
                    f1 = 0

            else:
                tp = np.sum(np.logical_and(gt_mask, pred_mask))
                print('numinst: num gt: %i, num pred: %i, num tp: %i'
                      % (num_gt, num_pred, tp))

                if num_pred > 0 and tp > 0:
                    precision = tp / float(num_pred)
                    recall = tp / float(num_gt)
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    precision = 0
                    recall = 0
                    f1 = 0

            metrics[thresh_key][str(int(rm_comp))] = {
                'num_gt': num_gt,
                'num_pred': num_pred,
                'num_tp': tp,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            }

        if kwargs.get('output_folder', None) is not None:
            outfn = os.path.join(
                kwargs['output_folder'],
                os.path.basename(prediction_fn).split('.')[0] +
                '_' + str(round(thresh, 2)).replace('.', '_') + '.png'
            )
            io.imsave(outfn, np.max(pred_mask, axis=0).astype(np.uint8) * 255)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-folder', type=str, dest='pred_folder',
                        help='pred folder', required=True)
    parser.add_argument('--aff-key', type=str, dest='aff_key',
                        help='affinity key')
    parser.add_argument('--gt-folder', type=str, dest='gt_folder',
                        help='gt folder', required=True)
    parser.add_argument('--gt-key', type=str, dest='gt_key',
                        help='gt key')
    parser.add_argument('--out-folder', type=str, dest='out_folder',
                        help='output folder')
    parser.add_argument('--patchshape', type=int,
                        help='patchshape', nargs='+',
                        default=[1, 25, 25])
    parser.add_argument('--num-workers', type=int, dest='num_workers',
                        default=1,
                        help='number of workers')

    args = parser.parse_args()

    preds = glob(args.pred_folder + '/*.zarr')
    samples = [os.path.basename(pred).split('.')[0] for pred in preds]
    labels = [glob(args.gt_folder + '/*' + sample + '*.zarr')[0]
              for sample in samples]

    patchshape = np.array(args.patchshape)

    preds = preds[:5]
    samples = samples[:5]
    labels = labels[:5]

    metrics = []
    if args.num_workers > 1:
        metrics = Parallel(n_jobs=args.num_workers, backend='multiprocessing',
                 verbose=0)(
            delayed(evaluate_patch)(p, l,
                                    aff_key=args.aff_key,
                                    sample_gt_key=args.gt_key,
                                    patchshape=patchshape,
                                    overlapping_inst=True,
                                    threshs=[0.9],
                                    return_patches=True,
                                    )
            for p, l in zip(preds, labels))
    else:
        for i, sample in enumerate(samples):
            metrics.append(evaluate_patch(preds[i], labels[i],
                           aff_key=args.aff_key,
                           gt_sample_key=args.gt_key,
                           patchshape=patchshape,
                           overlapping_inst=True,
                           threshs=[0.9],
                           return_patches=True,
                           ))

    patchshape = np.squeeze(patchshape)
    mse = np.zeros(patchshape)
    tp = np.zeros(patchshape)
    fp = np.zeros(patchshape)
    fn = np.zeros(patchshape)
    tp_patch_cnt = 0
    for metric in metrics:
        print(metric)
        metric = metric['0_9']
        for coord in metric['tp_patch_coords']:
            tp_patch_cnt += 1
            mse += np.reshape(metric['mse'][:, coord[0], coord[1]], patchshape)
            #tp += metric['tp'][coord]
            #fp += metric['fp'][coord]
            #fn += metric['fn'][coord]

    mse /= float(tp_patch_cnt)
    # precision = tp / (tp + fp).astype(np.float32)
    # recall = tp / (tp + fn).astype(np.float32)
    # avdsb = tp / (tp + fp + fn).astype(np.float32)

    plt.imshow(mse)
    plt.show()


if __name__ == "__main__":
    main()
