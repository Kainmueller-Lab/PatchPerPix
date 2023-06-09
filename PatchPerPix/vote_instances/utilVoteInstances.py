import pickle
import os
import logging
import re

import h5py
import numpy as np
import scipy.special
import zarr

if __package__ is None or __package__ == '':
    from get_patch_sets import *
else:
    from .get_patch_sets import *


logger = logging.getLogger(__name__)

def fillLookup(foreground, patchshape, neighshape, all_patches):
    print("fill lookup")
    lookup = np.empty(
        (tuple(foreground.shape) +
         tuple(2 * patchshape - 1) +
         (len(foreground.shape) + 1,)), np.int16)
    print("lookup shape: ", lookup.shape, lookup.nbytes)

    poffsA = np.zeros(2 * patchshape - 1, dtype=np.bool)
    # TODO check
    poffsA[patchshape[0]:] = True
    poffsA[patchshape[0] - 1, patchshape[1]:] = True
    if len(patchshape) > 2:
        poffsA[patchshape[0] - 1,
        patchshape[1] - 1,
        patchshape[2]:] = True
    poffsB = np.logical_not(poffsA)

    argwhereA = np.argwhere(poffsA) - patchshape + 1
    argwhereB = np.argwhere(poffsB) - patchshape + 1

    # todo: make it work for 2d and 3d
    offsA = np.array([[p[0] * neighshape[2] * neighshape[1] +
                       p[1] * neighshape[2] +
                       p[2]]
                      for p in argwhereA])
    offsB = np.array([[-p[0] * neighshape[2] * neighshape[1]
                       - p[1] * neighshape[2]
                       - p[2]]
                      for p in argwhereB])

    for idx1 in all_patches:
        lookup[tuple(idx1)][poffsA] = np.concatenate([
            offsA, np.full((len(offsA), len(idx1)), idx1)], axis=1)
        lookup[tuple(idx1)][poffsB] = np.concatenate([
            offsB, idx1 + argwhereB], axis=1)
    print("done fill lookup")
    return lookup


def computeFGBGsets(foreground,
                    all_patches,
                    pred_affs,
                    patchshape,
                    rad,
                    **kwargs):
    shape = foreground.shape

    # isbi2012 hack:
    if kwargs['isbiHack'] and shape[0] > 1:
        all_patches_fgs = [get_foreground_set(p, pred_affs,
                                              np.ones(shape),
                                              patchshape, rad,
                                              kwargs['patch_threshold'],
                                              sample=kwargs['sample'])
                           for p in all_patches]
        all_patches_bgs = [get_background_set(p, pred_affs,
                                              np.ones(shape),
                                              patchshape, rad,
                                              kwargs['patch_threshold'],
                                              sample=kwargs['sample'])
                           for p in all_patches]
    else:
        all_patches_fgs = [get_foreground_set(p, pred_affs, foreground,
                                              patchshape, rad,
                                              kwargs['patch_threshold'],
                                              sample=kwargs['sample'])
                           for p in all_patches]
        all_patches_bgs = [get_background_set(p, pred_affs, foreground,
                                              patchshape, rad,
                                              kwargs['patch_threshold'],
                                              sample=kwargs['sample'])
                           for p in all_patches]
    return all_patches_fgs, all_patches_bgs


def loadFromFile(filename, shape=None, key=None):
    logger.info("reading %s", filename)
    if filename.endswith("pickle"):
        print("as pickle")
        f = open(filename, 'rb')
        return pickle.load(f)
    elif filename.endswith("hdf"):
        print("as hdf")
        if key is None:
            print("provide hdf key for array")
            exit(-1)
        return np.array(h5py.File(filename, 'r')[key])
    elif filename.endswith("zarr"):
        print("as zarr")
        if key is None:
            print("provide hdf key for array")
            exit(-1)
        print(key)
        dst = np.array(zarr.open(filename, mode='r')[key])
        print(dst.shape)
        return dst
    elif filename.endswith("npy"):
        print("as npy")
        return np.load(filename)
    elif "bin" in filename:
        print("as binary blob")
        with open(filename, 'rb') as f:
            array = np.frombuffer(f.read(), dtype=np.int32)
            try:
                array.shape = shape
            except Exception as e:
                print("array unknown shape, please check")
                print(e)
                exit(-1)
            return array

    else:
        print("invalid file")
        exit(-1)


def loadAffinities(aff_file, res_ext, patchshape=None, **kwargs):
    numinst = None

    # import from hdf or zarr
    if aff_file.endswith((".hdf", ".zarr")):
        if aff_file.endswith(".hdf"):
            f = h5py.File(aff_file, 'r')
        else:
            f = zarr.open(aff_file, 'r')
        logger.info("keys: %s", list(f.keys()))
        if 'vote_instances' + res_ext in f.keys():
            logger.info("%s vote_instances %s already computed",
                        aff_file, res_ext)
            return

        if 'volumes' in f.keys():
            aff_key = kwargs.get('aff_key')
            if aff_key is None:
                aff_key = 'volumes/pred_affs'
                kwargs['aff_key'] = aff_key

            # 3D/ISBI2012:
            rotate_axes = False
            if patchshape is not None:
                patchshape_lin = int(np.prod(patchshape))
                shape = f[aff_key].shape
                if shape[-1] == patchshape_lin and \
                   shape[0] != patchshape_lin:
                    rotate_axes = True

            if len(f[aff_key].shape) == 3:
                if rotate_axes:
                    affinities = np.squeeze(np.array(
                        f[aff_key]
                        [kwargs.get('crop_y_s', 0):
                         kwargs.get('crop_y_e', None),
                         kwargs.get('crop_x_s', 0):
                         kwargs.get('crop_x_e', None),
                         :]
                    ))
                    affinities = np.ascontiguousarray(
                        np.moveaxis(affinities, -1, 0))
                else:
                    affinities = np.squeeze(np.array(
                        f[aff_key]
                        [:,
                         kwargs.get('crop_y_s', 0):
                         kwargs.get('crop_y_e', None),
                         kwargs.get('crop_x_s', 0):
                         kwargs.get('crop_x_e', None)]
                    ))
                affinities = np.expand_dims(affinities, axis=1)
            elif len(f[aff_key].shape) == 4:
                if rotate_axes:
                    affinities = np.squeeze(np.array(
                        f[aff_key]
                        [kwargs.get('crop_z_s', 0):
                         kwargs.get('crop_z_e', None),
                         kwargs.get('crop_y_s', 0):
                         kwargs.get('crop_y_e', None),
                         kwargs.get('crop_x_s', 0):
                         kwargs.get('crop_x_e', None),
                         :]
                    ))
                    affinities = np.ascontiguousarray(
                        np.moveaxis(affinities, -1, 0))
                else:
                    affinities = np.squeeze(np.array(
                        f[aff_key]
                        [:,
                         kwargs.get('crop_z_s', 0):
                         kwargs.get('crop_z_e', None),
                         kwargs.get('crop_y_s', 0):
                         kwargs.get('crop_y_e', None),
                         kwargs.get('crop_x_s', 0):
                         kwargs.get('crop_x_e', None)]
                    ))
            else:
                raise RuntimeError("check dimensions of array %s %s", aff_file, aff_key)
            if kwargs['isbiHack']:
                affinities = affinities[:, :, ::2, ::2]
        else:
            # "images" for 2D
            affinities = np.array(f['images/pred_affs'])
            if affinities.shape[1] != 1:
                affinities = np.expand_dims(affinities, axis=1)

            logger.info("affinities shape %s", affinities.shape)

        numinst = maybeLoadNuminst(f, **kwargs)
        foreground, _ = loadFg(f, **kwargs, patchshape=patchshape)


        if aff_file.endswith(".hdf"):
            f.close()

    # import numpy array
    elif aff_file.endswith("npy"):
        affinities = np.load(aff_file)
        if affinities.shape[1] != 1:
            affinities = np.expand_dims(affinities, axis=1)
        logger.info("%s", affinities.shape)
        mid = np.prod(patchshape) // 2
        foreground = np.array(affinities[mid])
        fg_thresh = getFgThreshold(**kwargs)
        foreground = foreground > fg_thresh
        numinst = 1*foreground

    # not implemented file format
    else:
        logger.info("invalid affinities file, zarr, hdf or npy")
        exit(-1)

    if np.min(affinities) < 0 and np.max(affinities) > 1:
        affinities = scipy.special.expit(affinities)
    return affinities, numinst, foreground


def getFgThreshold(**kwargs):
    if kwargs.get('fg_thresh_vi', -1) > 0:
        return kwargs['fg_thresh_vi']
    else:
        return kwargs['patch_threshold']

def maybeLoadNuminst(f, **kwargs):
    numinst = None
    if kwargs.get('numinst_key') is not None:
        numinst_prob = np.array(f[kwargs['numinst_key']])
        numinst_prob = np.squeeze(numinst_prob)
        if len(numinst_prob.shape) == 3:
            numinst_prob = np.expand_dims(numinst_prob, axis=1)
        numinst = np.argmax(numinst_prob, axis=0).astype(np.uint8)
        if kwargs.get('numinst_threshs'):
            numinst = np.zeros(numinst_prob.shape[1:], dtype=np.uint8)
            for i in range(len(kwargs['numinst_threshs'])):
                numinst[numinst_prob[i+1]>kwargs['numinst_threshs'][i]] = i+1
    return numinst


def loadFg(f, **kwargs):
    aff_key = kwargs['aff_key']
    fg_key = kwargs.get('fg_key', None)
    numinst_key = kwargs.get('numinst_key', None)
    fg_thresh = getFgThreshold(**kwargs)
    foreground = None
    # should be set explicitly, so no .get(.., default)
    # in [vote_instances]

    if fg_key is not None:
        foreground = np.array(f[fg_key])
        key = fg_key
    elif numinst_key is not None:
        numinst_prob = np.array(f[kwargs['numinst_key']])
        numinst = np.argmax(numinst_prob, axis=0).astype(np.uint8)
        if kwargs.get('numinst_threshs'):
            numinst = np.zeros(numinst_prob.shape[1:], dtype=np.uint8)
            for i in range(len(kwargs['numinst_threshs'])):
                numinst[numinst_prob[i+1]>kwargs['numinst_threshs'][i]] = i+1
        foreground = (numinst > 0).astype(np.float32)
        foreground = np.expand_dims(foreground, axis=0)
        key = numinst_key
    else:
        mid = np.prod(kwargs['patchshape']) // 2
        foreground = np.expand_dims(np.array(f[aff_key][mid]), axis=0)
        key = aff_key

    foreground = foreground > fg_thresh
    return foreground, key


def returnFg(affs, numinst, fg, **kwargs):
    fg_key = kwargs.get('fg_key', None)
    numinst_key = kwargs.get('numinst_key', None)
    fg_thresh = getFgThreshold(**kwargs)
    foreground = None
    # should be set explicitly, so no .get(.., default)
    # in [vote_instances]
    if fg_key is not None:
        foreground = np.squeeze(fg)
    elif numinst_key is not None:
        foreground = numinst > 0
    else:
        mid = np.prod(kwargs['patchshape']) // 2
        foreground = affs[mid]

    foreground = foreground > fg_thresh
    return foreground


def getResKey(**kwargs):
    # keep it short
    res_ext = ""
    res_ext = '_' + str(kwargs['patch_threshold']).replace('.', '')
    if not kwargs.get('skipThinCover', False):
        # "thin foreground cover" via greedy set cover algorithm
        res_ext += "_tfgc"
    if kwargs['mws']:
        res_ext += "_mws"
    if kwargs['sample'] < 1.0:
        res_ext += "_smp" + str(kwargs['sample']).replace('.', '')

    return res_ext


def loadKernelFromFile(filename, affshape, patchshape, neighshape,
                       patch_threshold):
    with open(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        filename), 'r') as f:
        code = ""
        for ln in f:
            if "#ifdef MAIN" in ln:
                break
            code += ln

    datacsize = affshape[0]
    datazsize = affshape[1]
    dataysize = affshape[2]
    dataxsize = affshape[3]
    nsz = int(neighshape[0])
    nsy = int(neighshape[1])
    nsx = int(neighshape[2])
    psz = int(patchshape[0])
    psy = int(patchshape[1])
    psx = int(patchshape[2])
    th = patch_threshold
    thi = 1.0 - th

    code = code.replace("{", "{{").replace("}", "}}") \
        .replace("DATAZSIZE", "{datazsize}") \
        .replace("DATAYSIZE", "{dataysize}") \
        .replace("DATAXSIZE", "{dataxsize}") \
        .replace("NSZ", "{nsz}") \
        .replace("NSY", "{nsy}") \
        .replace("NSX", "{nsx}") \
        .replace("THI", "{thi}")
    code = re.sub(r'(?<![_O])TH', '{th}', code)
    code = re.sub(r'PSZ(?!H)', '{psz}', code)
    code = re.sub(r'PSY(?!H)', '{psy}', code)
    code = re.sub(r'PSX(?!H)', '{psx}', code)

    code = code.format(datacsize=datacsize, datazsize=datazsize,
                       dataysize=dataysize, dataxsize=dataxsize,
                       nsz=nsz, nsy=nsy, nsx=nsx, psz=psz, psy=psy, psx=psx,
                       th=th, thi=thi)
    code = code.replace("{{", "{").replace("}}", "}")

    return code


def setKernelBuildOptions(step=None, **kwargs):
    build_options = []
    if kwargs.get("vi_bg_use_inv_th", True):
        logger.info("bg defined as: 1-th")
        if kwargs['patch_threshold'] < 0.5:
            logger.warning("bg defined as: 1-th, invalid: th < 0.5, switching..")
            logger.info("bg defined as: <th")
            build_options = ['-DUSE_LESS_THAN_TH']
        else:
            build_options = ['-DUSE_INV_TH']
    elif kwargs.get("vi_bg_use_half_th", False):
        logger.info("bg defined as: th/2")
        build_options = ['-DUSE_HALF_TH']
    elif kwargs.get("vi_bg_use_less_than_th", False):
        logger.info("bg defined as: <th")
        build_options = ['-DUSE_LESS_THAN_TH']
    else:
        raise RuntimeError("how is bg defined for vote instances?")

    if kwargs.get("overlapping_inst", False):
        logger.info("ignoring overlap in consensus computation")
        build_options.append('-DOVERLAP')

    if step == "consensus":
        if kwargs.get("consensus_norm_prob_product", True):
            logger.info("accumulate normalized prob product [0,1] as aff")
            build_options.append("-DNORM_PROB_PRODUCT")
        elif kwargs.get("consensus_prob_product", True):
            logger.info(
                "accumulate unnorm. prob product (range depends on th) as aff")
            build_options.append("-DPROB_PRODUCT")
        else:
            assert \
                not kwargs.get("consensus_norm_aff", True) and \
                not kwargs.get("consensus_interleaved_cnt", True) and \
                not kwargs.get("consensus_norm_prob_product", True) and \
                not kwargs.get("consensus_prob_product", True), \
                "no normalizing for accumulate consensus counter available"
            logger.info("accumulate counter as aff")

    if step == "rank":
        if kwargs.get("rank_norm_patch_score", True):
            logger.info("rank patch score will be normalized")
            build_options.append('-DNORM_PATCH_RANK')
        else:
            logger.info("rank patch score will NOT be normalized")

        if kwargs.get("rank_int_counter", False):
            logger.info("accumulate rank with int counter (not aff floats)")
            build_options.append('-DCOUNT_POS_NEG')
        else:
            logger.info("accumulate rank with aff floats")

    if step == "patch_graph":
        if kwargs.get("patch_graph_norm_aff", True):
            logger.info("patch affinities will be normalized")
            build_options = ['-DNORM_PATCH_AFFINITY']
        else:
            logger.info("patch affinities will NOT be normalized")

    return build_options


def get_block_shape(datazsize, dataysize, dataxsize):
    return (8, 8, 8)


def get_grid_shape(datazsize, dataysize, dataxsize):
    block_shape = get_block_shape(datazsize, dataysize, dataxsize)

    return \
        (dataxsize + block_shape[2] - 1) // block_shape[2], \
        (dataysize + block_shape[1] - 1) // block_shape[1], \
        (datazsize + block_shape[0] - 1) // block_shape[0]
