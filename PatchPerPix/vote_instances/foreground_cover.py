import logging
import h5py
import numpy as np
import scipy.spatial
import scipy.ndimage

if __package__ is None or __package__ == '':
    from get_patch_sets import *
else:
    from .get_patch_sets import *


logger = logging.getLogger(__name__)

def computeForegroundCover(
    overlap_mask,
    foreground_to_cover,
    patchshape,
    ranked_patches_list,
    radslice,
    labels,
    rad,
    debug_output1,
    scores_array,
    silent=False,
    **kwargs
):
    oneworm_foreground_running = foreground_to_cover.copy()

    selected_centers = 0 * oneworm_foreground_running
    rpidx = 0
    selected = np.zeros(len(ranked_patches_list), dtype=np.bool)
    marked = np.zeros_like(oneworm_foreground_running, dtype=np.bool)

    if kwargs['select_patches_for_sparse_data']:
        pixThs = [0]
    else:
        pixThs = [500, 100, 50, 10, 0]
    for pixTh in pixThs:
        if not silent:
            logger.info("compute foreground cover, threshold %s", pixTh)
        computeForegroundCoverLoop(oneworm_foreground_running, radslice,
                                   ranked_patches_list, overlap_mask, labels,
                                   rad, selected, selected_centers,
                                   debug_output1, rpidx, patchshape, pixTh,
                                   silent=silent,
                                   marked=marked,
                                   **kwargs)
        if np.sum(oneworm_foreground_running[radslice]) < 1:
            break

    if kwargs.get('select_patches_overlap_neighborhood', False):
        selected_patches = np.zeros_like(foreground_to_cover)
        selected_patches_list = [rp for rpi, rp in enumerate(ranked_patches_list)
                                 if selected[rpi]]
        for rp in selected_patches_list:
            selected_patches[tuple(rp[0])] = 1

        overlap = overlap_mask.copy()
        overlap_t = scipy.ndimage.binary_dilation(overlap, iterations=2)
        overlap_dil = scipy.ndimage.binary_dilation(overlap,
                                                    iterations=5)
        overlap_not = np.logical_not(overlap_t)
        dil_mask = np.logical_and(overlap_not, overlap_dil)
        fg_dil_mask = np.logical_and(dil_mask, foreground_to_cover)

        rp_list = [rp for rp in ranked_patches_list
                   if not selected_patches[tuple(rp[0])] and
                   fg_dil_mask[tuple(rp[0])]]
        selected_overlap = np.zeros(len(rp_list), dtype=np.bool)
        computeForegroundCoverLoop(fg_dil_mask, radslice,
                                   rp_list, overlap_mask, labels,
                                   rad, selected_overlap, selected_centers,
                                   debug_output1, rpidx, patchshape, pixTh,
                                   silent=silent,
                                   marked=marked,
                                   **kwargs)
        selected_patches_list = [rp for rpi, rp in enumerate(rp_list)
                                 if selected_overlap[rpi]]
        for rp in selected_patches_list:
            selected_patches[tuple(rp[0])] = 1
        selected_patches_list = list(np.argwhere(selected_patches > 0))
        selected_patches_list = [(rp, scores_array[tuple(rp)])
                                 for rp in selected_patches_list]
    else:
        selected_patches_list = [rp for rpi, rp in enumerate(ranked_patches_list)
                                 if selected[rpi]]

    if kwargs.get('store_selected_hdf', False):
        selected_patches = np.zeros_like(foreground_to_cover)
        for t in selected_patches_list:
            selected_patches[tuple([int(round(f)) for f in t[0]])] = 1
            with h5py.File("selected.hdf", 'w') as f:
                f.create_dataset(
                    "selected",
                    data=selected_patches.astype(np.uint8),
                    compression='gzip'
                )
    num_selected = len(selected_patches_list)
    if num_selected > 0 and not silent:
        logger.info("num patches to cover foreground: %s best score: %s, "
                    "worst score: %s at idx %s, uncovered: %s",
                    num_selected, selected_patches_list[0][1],
                    selected_patches_list[-1][1], rpidx,
                    np.sum(oneworm_foreground_running[radslice]))

    return selected_patches_list, num_selected


def computeForegroundCoverLoop(
    oneworm_foreground_running,
    radslice,
    ranked_patches_list,
    overlap_mask,
    labels,
    rad,
    selected,
    selected_centers,
    debug_output1,
    rpidx,
    patchshape,
    pixTh,
    silent=False,
    marked=None,
    **kwargs
):
    while (np.max(oneworm_foreground_running[radslice]) > 0 and
           rpidx < len(ranked_patches_list)):
        rpidx += 1
        r = rpidx - 1

        if selected[r]:
            continue

        if isinstance(kwargs.get('score_threshold', False), float) and \
           ranked_patches_list[r][1] < kwargs['score_threshold']:
            break

        idx = ranked_patches_list[r][0]
        if kwargs.get('mark_close_neighboorhood', False):
            if marked[tuple(idx)]:
                continue
        if overlap_mask[tuple(idx)] > 0:
            continue

        labelslice = tuple([slice(0, labels.shape[0])] +
                           [idx[i] for i in range(len(idx))])

        patch = labels[labelslice]
        patch = np.reshape(patch, patchshape)
        start = idx - rad
        stop = idx + rad + 1
        startstopslice = tuple([slice(start[i], stop[i])
                                for i in range(len(start))])
        if np.count_nonzero(
            oneworm_foreground_running[startstopslice]
            [patch > kwargs['fc_threshold']]) > pixTh:
            selected[r] = True
            selected_centers[tuple(idx)] = 1

            if kwargs.get('mark_close_neighboorhood', False):
                m_rad = np.array([0, 3, 3])
                m_start = idx - m_rad
                m_stop = idx + m_rad + 1
                m_startstopslice = tuple([slice(m_start[i], m_stop[i])
                                     for i in range(len(m_start))])
                marked[m_startstopslice] = True
            # heads up: using 0.5 here to make covering easier
            oneworm_foreground_running[startstopslice][
                patch > kwargs['fc_threshold']] = 0
        if kwargs['debug']:
            debugstart = idx * patchshape
            debugstop = debugstart + patchshape
            debugslice = tuple([slice(debugstart[i], debugstop[i])
                                for i in range(len(debugstart))])
            debug_output1[debugslice] += 1
        if rpidx % 10000 == 0 and not silent:
            logger.info("compute foreground cover: %s pixels left to cover",
                        np.sum(oneworm_foreground_running[radslice]))


def thinOutForegroundCover(
    foreground_to_cover,
    selected_patches_list,
    radslice,
    labels,
    rad,
    patchshape,
    **kwargs
):
    oneworm_foreground_running = foreground_to_cover.copy()
    selected = np.zeros(len(selected_patches_list), dtype=np.bool)
    selected_patch_foregrounds = [get_foreground_set(rp[0], labels,
                                                     foreground_to_cover,
                                                     patchshape, rad,
                                                     kwargs['fc_threshold'],
                                                     sample=kwargs['sample'])
                                  for rp in selected_patches_list]

    if kwargs.get('thin_cover_use_kd', False):
        tree = scipy.spatial.cKDTree([rp[0] for rp in selected_patches_list],
                                     leafsize=4)

    count_to_cover = np.sum(oneworm_foreground_running[radslice])
    count_covered = 0
    while np.max(oneworm_foreground_running[radslice]) > 0:
        selected_patch_foreground_covers = [len(s)
                                            for s in selected_patch_foregrounds]
        best_patch_i = np.argmax(selected_patch_foreground_covers)
        selected[best_patch_i] = True
        best_fg = get_foreground_set(selected_patches_list[best_patch_i][0],
                                     labels, oneworm_foreground_running,
                                     patchshape, rad, kwargs['fc_threshold'],
                                     sample=kwargs['sample'])
        oneworm_foreground_running[tuple(zip(*list(best_fg)))] = 0
        count_covered += len(best_fg)
        count_to_cover -= len(best_fg)
        if kwargs.get('thin_cover_use_kd', False):
            pts = tree.query_ball_point(
                selected_patches_list[best_patch_i][0],
                2*np.sum(patchshape), p=2)
            for i in pts:
                selected_patch_foregrounds[i] = \
                    selected_patch_foregrounds[i] - best_fg
        else:
            selected_patch_foregrounds = [s - best_fg
                                          for s in selected_patch_foregrounds]
        logger.info(
            "thin out foreground cover: %s covered, "
            "%s pixels left to cover",
            count_covered, count_to_cover)

    selected_patches_list = [rp for rpi, rp in enumerate(selected_patches_list)
                             if selected[rpi]]
    num_selected = len(selected_patches_list)
    logger.info('num_selected: %s', num_selected)
    if num_selected > 0:
        logger.info(
            "num patches to cover foreground: {} best score: {}, "
            "worst score: {} uncovered: {}".format(
                num_selected, selected_patches_list[0][1],
                selected_patches_list[-1][1],
                np.sum(oneworm_foreground_running[radslice])))

    if kwargs.get('store_selected_hdf', False):
        selected_patches = np.zeros_like(foreground_to_cover)
        for rp in selected_patches_list:
            selected_patches[tuple(rp[0])] = 1
        with h5py.File("selected2.hdf", 'w') as f:
            f.create_dataset(
                "selected",
                data=selected_patches.astype(np.uint8),
                compression='gzip'
                )
    return selected_patches_list, num_selected
