import numpy as np
import random

if __package__ is None or __package__ == '':
    from get_patch_sets import get_boundary_set
else:
    from .get_patch_sets import get_boundary_set


def sparsifyPatches(all_patches):
    # heads-up: hack for speed in case of dense foreground
    all_patches = [p for p in all_patches if np.all(p % 20 == 1)]
    print("hack-sparsified num foreground pixels: ", len(all_patches))
    return all_patches


def filterInstanceBoundariesFromFG(labels,
                                   foreground,
                                   all_patches,
                                   patchshape,
                                   rad,
                                   foreground_to_cover,
                                   **kwargs
                                   ):
    if kwargs['sample'] < 1.0:
        subsample = random.sample(
            range(len(all_patches)),
            int(kwargs['sample'] * len(all_patches)))
        subset_patches = [all_patches[i] for i in subsample]
        print("compute boundary from %s patches" % len(subset_patches))
    else:
        subset_patches = all_patches

    all_patches_boundaries = [get_boundary_set(p, labels,
                                               np.ones(foreground.shape),
                                               patchshape, rad,
                                               kwargs['patch_threshold'])
                              for p in subset_patches]

    boundary_count = 0*foreground
    for b in all_patches_boundaries:
        if len(b) > 0:
            boundary_count[tuple(np.transpose(list(b)))] += 1

    # TODO: check product
    foreground_to_cover = np.logical_and(
        foreground_to_cover,
        boundary_count < 0.33 * np.product(patchshape[1:]))
