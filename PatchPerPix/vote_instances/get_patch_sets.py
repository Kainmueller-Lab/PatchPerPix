import numpy as np
import random
from skimage.morphology import binary_dilation, ball


def get_boundary_set(idx, labels, foreground, patchshape, rad, pthresh):
    labelslice = tuple([slice(0, labels.shape[0])] +
                       [idx[i]
                        for i in range(len(idx))])
    patchprob = labels[labelslice]
    patchprob = np.reshape(patchprob, patchshape)

    start = idx - rad
    stop = idx + rad + 1
    pbd = set()

    if np.all(start >= 0) and np.all(stop <= foreground.shape):
        startstopslice = tuple([slice(start[i], stop[i])
                                for i in range(len(start))])
        patch = 1 * np.logical_and((patchprob > pthresh),
                                   foreground[startstopslice])
        patch_grow = binary_dilation(patch, selem=ball(2))
        patch = patch_grow - patch

        pbd = start + np.argwhere(np.logical_and(patch > 0,
                                                 foreground[startstopslice]))
        pbd = set(map(tuple, pbd))

    return pbd


def get_foreground_set(idx, labels, foreground, patchshape, rad, pthresh,
                       sample=1.0):
    labelslice = tuple([slice(0, labels.shape[0])] +
                       [idx[i]
                        for i in range(len(idx))])

    patchprob = labels[labelslice]
    patchprob = np.reshape(patchprob, patchshape)

    start = idx - rad
    stop = idx + rad + 1
    pf = set()

    if np.all(start >= 0) and np.all(stop <= foreground.shape):
        startstopslice = tuple([slice(start[i], stop[i])
                                for i in range(len(start))])
        pf = start + np.argwhere(np.logical_and(patchprob > pthresh,
                                                foreground[startstopslice]))
        pf = set(map(tuple, pf))
        if len(pf) > 0 and sample < 1:
            pf = random.sample(pf, max(1, int(sample * len(pf))))

    return pf


def get_background_set(idx, labels, foreground, patchshape, rad, pthresh,
                       sample=1.0):
    labelslice = tuple([slice(0, labels.shape[0])] +
                       [idx[i]
                        for i in range(len(idx))])

    patchprob = labels[labelslice]
    patchprob = np.reshape(patchprob, patchshape)

    start = idx - rad
    stop = idx + rad + 1
    pb = set()

    if np.all(start >= 0) and np.all(stop <= foreground.shape):
        startstopslice = tuple([slice(start[i], stop[i])
                                for i in range(len(start))])
        pb = start + np.argwhere(np.logical_and(patchprob < 1 - pthresh,
                                                foreground[startstopslice]))
        pb = set(map(tuple, pb))
        if len(pb) > 0 and sample < 1:
            pb = random.sample(pb, max(1, int(sample * len(pb))))

    return pb
