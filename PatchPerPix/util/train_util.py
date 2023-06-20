"""Utility functions for training process
"""
import glob
import logging
import math
import re

import numpy as np
try:
    import torch
except:
    print("no torch found! (train_util.py, might be ok if tf is used)")

import gunpowder as gp

logger = logging.getLogger(__name__)


def get_latest_checkpoint(basename):
    """Looks for the checkpoint with the highest step count

    Checks for files name basename + '_checkpoint_*'
    The suffix should be the iteration count
    Selects the one with the highest one and returns the path to it
    and the step count

    Args
    ----
    basename: str
        Path to and prefix of model checkpoints

    Returns
    -------
    2-tuple: str, int
        Path to and iteration of latest checkpoint

    """
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split(r'(\d+)', text)]

    checkpoints = glob.glob(basename + '_checkpoint_*')
    checkpoints.sort(key=natural_keys)

    if len(checkpoints) > 0:
        checkpoint = checkpoints[-1]
        iteration = int(checkpoint.split('_')[-1].split('.')[0])
        return checkpoint, iteration

    return None, 0


def crop(x, shape):
    '''Center-crop x to match spatial dimensions given by shape.'''

    dims = len(x.size()) - len(shape)
    x_target_size = x.size()[:dims] + shape

    offset = tuple(
        (a - b)//2
        for a, b in zip(x.size(), x_target_size))

    slices = tuple(
        slice(o, o + s)
        for o, s in zip(offset, x_target_size))

    return x[slices]


def crop_to_factor(x, factor, kernel_sizes):
    '''Crop feature maps to ensure translation equivariance with stride of
    upsampling factor. This should be done right after upsampling, before
    application of the convolutions with the given kernel sizes.

    The crop could be done after the convolutions, but it is more efficient
    to do that before (feature maps will be smaller).
    '''

    shape = x.size()
    dims = len(x.size()) - 2
    spatial_shape = shape[-dims:]

    # the crop that will already be done due to the convolutions
    convolution_crop = tuple(
        sum(ks[d] - 1 for ks in kernel_sizes)
        for d in range(dims)
    )

    # we need (spatial_shape - convolution_crop) to be a multiple of
    # factor, i.e.:
    #
    # (s - c) = n*k
    #
    # we want to find the largest n for which s' = n*k + c <= s
    #
    # n = floor((s - c)/k)
    #
    # this gives us the target shape s'
    #
    # s' = n*k + c

    ns = (
        int(math.floor(float(s - c)/f))
        for s, c, f in zip(spatial_shape, convolution_crop, factor)
    )
    target_spatial_shape = tuple(
        n*f + c
        for n, c, f in zip(ns, convolution_crop, factor)
    )

    if target_spatial_shape != spatial_shape:

        assert all((
            (t > c) for t, c in zip(
                target_spatial_shape,
                convolution_crop))
                   ), \
                   "Feature map with shape %s is too small to ensure " \
                   "translation equivariance with factor %s and following " \
                   "convolutions %s" % (
                       shape,
                       factor,
                       kernel_sizes)

        return crop(x, target_spatial_shape)

    return x


def normalize(
        pipeline, normalization, raw=None, is_gp=True,
        strict_clip=False, **kwargs):
    """Add data normalization node to pipeline

    Args
    ----
    pipeline: gp.BatchFilter or array-like
        Gunpowder node/pipeline, typically a source node containing data
        that should be normalized or an array
    normalization: NormalizeConfig
        Normalization configuration object used to determine which type of
        normalization should be performed
    raw: gp.ArrayKey
        Key identifying which array to normalize
    is_gp: bool
        Flag if input is gunpowder pipeline or array
    kargs: dict
        Additional stats

    Returns
    -------
    gp.BatchFilter or array-like
        Pipeline extended by a normalization node or transformed array

    Notes
    -----
    Which normalization method should be used?
    None/default:
        [0,1] based on data type
    minmax:
        Normalize such that lower bound is at 0 and upper bound at 1
        clipping is less strict, some data might be outside of range
    percminmax:
        Use (precomputed) percentile values for minmax normalization.
        Set perc_min/max to tag to be used
    mean/median
        Normalize such that (precomputed) mean/median is at 0 and 1 std/mad is
        at -+1. Set perc_min/max tags for clipping beforehand.
    """
    if strict_clip:
        clip_fac = 1
    else:
        clip_fac = 2
    if normalization is None or \
       normalization["type"] == 'default':
        logger.info("default normalization")
        try:
            dtype = kwargs['dtype']
        except:
            if is_gp:
                dtype = pipeline.spec[raw].dtype
            else:
                dtype = pipeline.dtype
        if is_gp:
            pipeline = pipeline + \
                gp.Normalize(raw, factor=1.0/np.iinfo(dtype).max)
        else:
            pipeline /= 1.0/np.iinfo(dtype).max
    elif normalization["type"] == 'minmax':
        mn = normalization["norm_bounds"][0]
        mx = normalization["norm_bounds"][1]
        logger.info("minmax normalization %s %s", mn, mx)
        if is_gp:
            pipeline = pipeline + \
                gp.Clip(raw, mn=mn/clip_fac, mx=mx*clip_fac) + \
                gp.NormalizeLowerUpper(
                    raw, lower=mn, upper=mx, interpolatable=False)
        else:
            pipeline = np.clip(pipeline, mn/clip_fac, mx*clip_fac)
            pipeline = (pipeline - mn) / (mx - mn)
    elif normalization["type"] == 'percminmax':
        mn = normalization["perc_min"]
        mx = normalization["perc_max"]
        logger.info("perc minmax normalization %s %s", mn, mx)
        # gp.Clip(raw, mn=mn/2, mx=mx*2) + \
        if is_gp:
            pipeline = pipeline + \
                gp.Clip(raw, mn=mn/clip_fac, mx=mx*clip_fac) + \
                gp.NormalizeLowerUpper(raw, lower=mn, upper=mx)
        else:
            pipeline = np.clip(pipeline, mn/clip_fac, mx*clip_fac)
            pipeline = (pipeline - mn) / (mx - mn)
    elif normalization["type"] == 'mean':
        mean = kwargs['mean']
        std = kwargs['std']
        mn = normalization["perc_min"]
        mx = normalization["perc_max"]
        logger.info("mean normalization %s %s %s %s", mean, std, mn, mx)
        if is_gp:
            pipeline = pipeline + \
                gp.Clip(raw, mn=mn, mx=mx) + \
                gp.NormalizeAroundZero(raw, mapped_to_zero=mean,
                                       diff_mapped_to_one=std)
        else:
            pipeline = np.clip(pipeline, mn, mx)
            pipeline = (pipeline - mean) / std
    elif normalization["type"] == 'median':
        median = kwargs['median']
        mad = kwargs['mad']
        mn = normalization["perc_min"]
        mx = normalization["perc_max"]
        logger.info("median normalization %s %s %s %s", median, mad, mn, mx)
        if is_gp:
            pipeline = pipeline + \
                gp.Clip(raw, mn=mn, mx=mx) + \
                gp.NormalizeAroundZero(raw, mapped_to_zero=median,
                                       diff_mapped_to_one=mad)
        else:
            pipeline = np.clip(pipeline, mn, mx)
            pipeline = (pipeline - median) / mad
    else:
        raise RuntimeError("invalid normalization method %s",
                           normalization["type"])
    return pipeline


def gather_nd_torch(params, indices, batch_dim=1):
    """ A PyTorch porting of tensorflow.gather_nd
    This implementation can handle leading batch dimensions in params, see below for detailed explanation.

    The majority of this implementation is from Michael Jungo @ https://stackoverflow.com/a/61810047/6670143
    I just ported it compatible to leading batch dimension.

    Args:
      params: a tensor of dimension [b1, ..., bn, g1, ..., gm, c].
      indices: a tensor of dimension [b1, ..., bn, x, m]
      batch_dim: indicate how many batch dimension you have, in the above example, batch_dim = n.

    Returns:
      gathered: a tensor of dimension [b1, ..., bn, x, c].

    Example:
    >>> batch_size = 5
    >>> inputs = torch.randn(batch_size, batch_size, batch_size, 4, 4, 4, 32)
    >>> pos = torch.randint(4, (batch_size, batch_size, batch_size, 12, 3))
    >>> gathered = gather_nd_torch(inputs, pos, batch_dim=3)
    >>> gathered.shape
    torch.Size([5, 5, 5, 12, 32])

    >>> inputs_tf = tf.convert_to_tensor(inputs.numpy())
    >>> pos_tf = tf.convert_to_tensor(pos.numpy())
    >>> gathered_tf = tf.gather_nd(inputs_tf, pos_tf, batch_dims=3)
    >>> gathered_tf.shape
    TensorShape([5, 5, 5, 12, 32])

    >>> gathered_tf = torch.from_numpy(gathered_tf.numpy())
    >>> torch.equal(gathered_tf, gathered)
    True
    """
    batch_dims = params.size()[:batch_dim]  # [b1, ..., bn]
    batch_size = np.cumprod(list(batch_dims))[-1]  # b1 * ... * bn
    c_dim = params.size()[-1]  # c
    grid_dims = params.size()[batch_dim:-1]  # [g1, ..., gm]
    n_indices = indices.size(-2)  # x
    n_pos = indices.size(-1)  # m

    # reshape leadning batch dims to a single batch dim
    params = params.reshape(batch_size, *grid_dims, c_dim)
    indices = indices.reshape(batch_size, n_indices, n_pos)

    # build gather indices
    # gather for each of the data point in this "batch"
    batch_enumeration = torch.arange(batch_size).unsqueeze(1)
    gather_dims = [indices[:, :, i] for i in range(len(grid_dims))]
    gather_dims.insert(0, batch_enumeration)
    gathered = params[gather_dims]

    # reshape back to the shape with leading batch dims
    gathered = gathered.reshape(*batch_dims, n_indices, c_dim)
    return gathered


def gather_nd_torch_no_batch(params, indices):
    """ A PyTorch porting of tensorflow.gather_nd
    This implementation can handle leading batch dimensions in params, see below for detailed explanation.

    The majority of this implementation is from Michael Jungo @ https://stackoverflow.com/a/61810047/6670143
    I just ported it compatible to leading batch dimension.

    Args:
      params: a tensor of dimension [g1, ..., gm, c].
      indices: a tensor of dimension [x, m]

    Returns:
      gathered: a tensor of dimension [x, c].

    Example:
    >>> inputs = torch.randn(4, 4, 4, 32)
    >>> pos = torch.randint(4, (12, 3))
    >>> gathered = gather_nd_torch(inputs, pos)
    >>> gathered.shape
    torch.Size([12, 32])

    >>> inputs_tf = tf.convert_to_tensor(inputs.numpy())
    >>> pos_tf = tf.convert_to_tensor(pos.numpy())
    >>> gathered_tf = tf.gather_nd(inputs_tf, pos_tf)
    >>> gathered_tf.shape
    TensorShape([12, 32])

    >>> gathered_tf = torch.from_numpy(gathered_tf.numpy())
    >>> torch.equal(gathered_tf, gathered)
    True
    """
    c_dim = params.size()[-1]  # c
    grid_dims = params.size()[:-1]  # [g1, ..., gm]
    n_indices = indices.size(-2)  # x
    n_pos = indices.size(-1)  # m

    # build gather indices
    gather_dims = [indices[:, i] for i in range(len(grid_dims))]
    gathered = params[gather_dims]

    return gathered



def seg_to_affgraph_3d_multi_torch_code_batch(seg, nhood, device, ps, psH, samples_loc, b):
    # based on:
    # https://github.com/TuragaLab/malis/blob/572ef0420107eee3c721bdafb58775a8a0fc467a/malis/malis.pyx
    # https://github.com/funkelab/gunpowder/blob/f202a6062dea65a4367a29816e8a2cc87d28b4b6/gunpowder/nodes/add_affinities.py
    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)

    shape = seg.shape[2:]
    nEdge = nhood.shape[0]

    aff = torch.zeros(
        (len(samples_loc), nEdge),
        dtype=torch.float32, device=device)

    patches = torch.zeros(
        (len(samples_loc), seg.shape[1],) + (ps, ps, ps),
        dtype=seg.dtype, device=device)
    for idx, sample_loc in enumerate(samples_loc):
        patches[idx] = seg[
            b, :,
            sample_loc[0]:sample_loc[0]+ps,
            sample_loc[1]:sample_loc[1]+ps,
            sample_loc[2]:sample_loc[2]+ps]

    fg_seg = torch.any(patches, dim=1).to(torch.int32)
    for e in range(nEdge):

        center = patches[:, :, psH, psH, psH]
        offset = patches[:, :, nhood[e, 0], nhood[e, 1], nhood[e, 2]]

        # first == second pixel?
        t2 = fg_seg[:, psH, psH, psH]
        t3 = fg_seg[:, nhood[e, 0], nhood[e, 1], nhood[e, 2]]
        t1 = (center == offset).to(torch.int32)
        t1[center == 0] = 0
        # t1 = t1 * center
        partial_same = torch.any(t1, dim=1)

        aff[:, e] = partial_same * t2 * t3

    return aff


def seg_to_affgraph_2d_multi_torch_code_batch(seg, nhood, device, ps, psH, samples_loc, b):
    # based on:
    # https://github.com/TuragaLab/malis/blob/572ef0420107eee3c721bdafb58775a8a0fc467a/malis/malis.pyx
    # https://github.com/funkelab/gunpowder/blob/f202a6062dea65a4367a29816e8a2cc87d28b4b6/gunpowder/nodes/add_affinities.py
    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, y, x)
    # nhood.shape = (edges, 2)

    shape = seg.shape[2:]
    nEdge = nhood.shape[0]

    aff = torch.zeros(
        (len(samples_loc), nEdge),
        dtype=torch.float32, device=device)

    patches = torch.zeros(
        (len(samples_loc), seg.shape[1],) + (ps, ps),
        dtype=seg.dtype, device=device)
    for idx, sample_loc in enumerate(samples_loc):
        patches[idx] = seg[
            b, :,
            sample_loc[0]:sample_loc[0]+ps,
            sample_loc[1]:sample_loc[1]+ps]

    fg_seg = torch.any(patches, dim=1).to(torch.int32)
    for e in range(nEdge):

        center = patches[:, :, psH, psH]
        offset = patches[:, :, nhood[e, 0], nhood[e, 1]]

        # first == second pixel?
        t2 = fg_seg[:, psH, psH]
        t3 = fg_seg[:, nhood[e, 0], nhood[e, 1]]
        t1 = (center == offset).to(torch.int32)
        t1[center == 0] = 0
        # t1 = t1 * center
        partial_same = torch.any(t1, dim=1)

        aff[:, e] = partial_same * t2 * t3

    return aff


def seg_to_affgraph_3d_multi_torch_code(seg, nhood, device, ps, psH, samples_loc):
    # based on:
    # https://github.com/TuragaLab/malis/blob/572ef0420107eee3c721bdafb58775a8a0fc467a/malis/malis.pyx
    # https://github.com/funkelab/gunpowder/blob/f202a6062dea65a4367a29816e8a2cc87d28b4b6/gunpowder/nodes/add_affinities.py
    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)

    shape = seg.shape[2:]
    nEdge = nhood.shape[0]

    aff = torch.zeros(
        (len(samples_loc), nEdge),
        dtype=torch.float32, device=device)

    patches = torch.zeros(
        (len(samples_loc), seg.shape[1],) + (ps, ps, ps),
        dtype=seg.dtype, device=device)
    for idx, sample_loc in enumerate(samples_loc):
        patches[idx] = seg[
            sample_loc[0], :,
            sample_loc[1]:sample_loc[1]+ps,
            sample_loc[2]:sample_loc[2]+ps,
            sample_loc[3]:sample_loc[3]+ps]

    fg_seg = torch.any(patches, dim=1).to(torch.int32)
    for e in range(nEdge):

        center = patches[:, :, psH, psH, psH]
        offset = patches[:, :, nhood[e, 0], nhood[e, 1], nhood[e, 2]]
        # first == second pixel?
        t2 = fg_seg[:, psH, psH, psH]
        t3 = fg_seg[:, nhood[e, 0], nhood[e, 1], nhood[e, 2]]
        t1 = (center == offset).to(torch.int32)
        t1[center == 0] = 0

        partial_same = torch.any(t1, dim=1)
        aff[:, e] = partial_same * t2 * t3

    return aff


def seg_to_affgraph_2d_multi_torch_code(seg, nhood, device, ps, psH, samples_loc):
    # based on:
    # https://github.com/TuragaLab/malis/blob/572ef0420107eee3c721bdafb58775a8a0fc467a/malis/malis.pyx
    # https://github.com/funkelab/gunpowder/blob/f202a6062dea65a4367a29816e8a2cc87d28b4b6/gunpowder/nodes/add_affinities.py
    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, y, x)
    # nhood.shape = (edges, 2)

    shape = seg.shape[2:]
    nEdge = nhood.shape[0]

    aff = torch.zeros(
        (len(samples_loc), nEdge),
        dtype=torch.float32, device=device)

    patches = torch.zeros(
        (len(samples_loc), seg.shape[1],) + (ps, ps),
        dtype=seg.dtype, device=device)
    for idx, sample_loc in enumerate(samples_loc):
        patches[idx] = seg[
            sample_loc[0], :,
            sample_loc[1]:sample_loc[1]+ps,
            sample_loc[2]:sample_loc[2]+ps]

    fg_seg = torch.any(patches, dim=1).to(torch.int32)
    for e in range(nEdge):

        center = patches[:, :, psH, psH]
        offset = patches[:, :, nhood[e, 0], nhood[e, 1]]
        # first == second pixel?
        t2 = fg_seg[:, psH, psH]
        t3 = fg_seg[:, nhood[e, 0], nhood[e, 1]]
        t1 = (center == offset).to(torch.int32)
        t1[center == 0] = 0

        partial_same = torch.any(t1, dim=1)
        aff[:, e] = partial_same * t2 * t3

    return aff


def seg_to_affgraph_3d_multi_torch(seg, nhood, device):
    # based on:
    # https://github.com/TuragaLab/malis/blob/572ef0420107eee3c721bdafb58775a8a0fc467a/malis/malis.pyx
    # https://github.com/funkelab/gunpowder/blob/f202a6062dea65a4367a29816e8a2cc87d28b4b6/gunpowder/nodes/add_affinities.py
    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)

    shape = seg.shape[2:]
    nEdge = nhood.shape[0]

    aff = torch.zeros(
        (seg.shape[0], nEdge,) + shape, dtype=torch.float32, device=device)

    fg_seg = torch.any(seg, dim=1).to(torch.int32)
    for e in range(nEdge):

        slice_center_z = slice(max(0, -nhood[e, 0]),
                               min(shape[0], shape[0] - nhood[e, 0]))
        slice_center_y = slice(max(0, -nhood[e, 1]),
                               min(shape[1], shape[1] - nhood[e, 1]))
        slice_center_x = slice(max(0, -nhood[e, 2]),
                               min(shape[2], shape[2] - nhood[e, 2]))

        slice_offset_z = slice(max(0, nhood[e, 0]),
                               min(shape[0], shape[0] + nhood[e, 0]))
        slice_offset_y = slice(max(0, nhood[e, 1]),
                               min(shape[1], shape[1] + nhood[e, 1]))
        slice_offset_x = slice(max(0, nhood[e, 2]),
                               min(shape[2], shape[2] + nhood[e, 2]))

        # first == second pixel?
        center = seg[:, :, slice_center_z, slice_center_y, slice_center_x]
        offset = seg[:, :, slice_offset_z, slice_offset_y, slice_offset_x]
        t2 = fg_seg[:, slice_center_z, slice_center_y, slice_center_x]
        t3 = fg_seg[:, slice_offset_z, slice_offset_y, slice_offset_x]
        t1 = (center == offset).to(torch.int32)
        t1[center == 0] = 0
        partial_same = torch.any(t1, dim=1)

        aff[:, e, slice_center_z, slice_center_y, slice_center_x] = \
            partial_same * t2 * t3

    return aff


def seg_to_affgraph_2d_multi_torch(seg, nhood, device):
    # based on:
    # https://github.com/TuragaLab/malis/blob/572ef0420107eee3c721bdafb58775a8a0fc467a/malis/malis.pyx
    # https://github.com/funkelab/gunpowder/blob/f202a6062dea65a4367a29816e8a2cc87d28b4b6/gunpowder/nodes/add_affinities.py
    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, y, x)
    # nhood.shape = (edges, 2)

    shape = seg.shape[2:]
    nEdge = nhood.shape[0]

    aff = torch.zeros(
        (seg.shape[0], nEdge,) + shape, dtype=torch.float32, device=device)

    fg_seg = torch.any(seg, dim=1).to(torch.int32)
    for e in range(nEdge):

        slice_center_y = slice(max(0, -nhood[e, 0]),
                               min(shape[0], shape[0] - nhood[e, 0]))
        slice_center_x = slice(max(0, -nhood[e, 1]),
                               min(shape[1], shape[1] - nhood[e, 1]))

        slice_offset_y = slice(max(0, nhood[e, 0]),
                               min(shape[0], shape[0] + nhood[e, 0]))
        slice_offset_x = slice(max(0, nhood[e, 1]),
                               min(shape[1], shape[1] + nhood[e, 1]))

        # first == second pixel?
        center = seg[:, :, slice_center_y, slice_center_x]
        offset = seg[:, :, slice_offset_y, slice_offset_x]
        t2 = fg_seg[:, slice_center_y, slice_center_x]
        t3 = fg_seg[:, slice_offset_y, slice_offset_x]
        t1 = (center == offset).to(torch.int32)
        t1[center == 0] = 0
        partial_same = torch.any(t1, dim=1)

        aff[:, e, slice_center_y, slice_center_x] = \
            partial_same * t2 * t3

    return aff


def seg_to_affgraph_2d_torch(seg, nhood, device):
    # based on:
    # https://github.com/TuragaLab/malis/blob/572ef0420107eee3c721bdafb58775a8a0fc467a/malis/malis.pyx
    # https://github.com/funkelab/gunpowder/blob/f202a6062dea65a4367a29816e8a2cc87d28b4b6/gunpowder/nodes/add_affinities.py
    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, y, x)
    # nhood.shape = (edges, 2)

    shape = seg.shape[1:]
    nEdge = nhood.shape[0]

    aff = torch.zeros(
        (seg.shape[0], nEdge,) + shape, dtype=torch.float32, device=device)

    for e in range(nEdge):

        slice_center_y = slice(max(0, -nhood[e, 0]),
                               min(shape[0], shape[0] - nhood[e, 0]))
        slice_center_x = slice(max(0, -nhood[e, 1]),
                               min(shape[1], shape[1] - nhood[e, 1]))

        slice_offset_y = slice(max(0, nhood[e, 0]),
                               min(shape[0], shape[0] + nhood[e, 0]))
        slice_offset_x = slice(max(0, nhood[e, 1]),
                               min(shape[1], shape[1] + nhood[e, 1]))

        # first == second pixel?
        center = seg[:, slice_center_y, slice_center_x]
        offset = seg[:, slice_offset_y, slice_offset_x]
        t1 = (center == offset).to(torch.int32)

        t2 = seg[:, slice_center_y, slice_center_x]
        t3 = seg[:, slice_offset_y, slice_offset_x]

        aff[:, e, slice_center_y, slice_center_x] = \
            t1 * t2 * t3

    return aff


def seg_to_affgraph_3d_torch(seg, nhood, device):
    # based on:
    # https://github.com/TuragaLab/malis/blob/572ef0420107eee3c721bdafb58775a8a0fc467a/malis/malis.pyx
    # https://github.com/funkelab/gunpowder/blob/f202a6062dea65a4367a29816e8a2cc87d28b4b6/gunpowder/nodes/add_affinities.py
    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)

    shape = seg.shape[1:]
    nEdge = nhood.shape[0]

    aff = torch.zeros(
        (seg.shape[0], nEdge,) + shape, dtype=torch.float32, device=device)

    for e in range(nEdge):

        slice_center_z = slice(max(0, -nhood[e, 0]),
                               min(shape[0], shape[0] - nhood[e, 0]))
        slice_center_y = slice(max(0, -nhood[e, 1]),
                               min(shape[1], shape[1] - nhood[e, 1]))
        slice_center_x = slice(max(0, -nhood[e, 2]),
                               min(shape[2], shape[2] - nhood[e, 2]))

        slice_offset_z = slice(max(0, nhood[e, 0]),
                               min(shape[0], shape[0] + nhood[e, 0]))
        slice_offset_y = slice(max(0, nhood[e, 1]),
                               min(shape[1], shape[1] + nhood[e, 1]))
        slice_offset_x = slice(max(0, nhood[e, 2]),
                               min(shape[2], shape[2] + nhood[e, 2]))

        # first == second pixel?
        center = seg[:, slice_center_z, slice_center_y, slice_center_x]
        offset = seg[:, slice_offset_z, slice_offset_y, slice_offset_x]
        t1 = (center == offset).to(torch.int32)
        t2 = seg[:, slice_center_z, slice_center_y, slice_center_x]
        t3 = seg[:, slice_offset_z, slice_offset_y, slice_offset_x]

        aff[:, e, slice_center_z, slice_center_y, slice_center_x] = \
            t1 * t2 * t3

    return aff


def seg_to_affgraph_3d_torch_code(seg, nhood, device, ps, psH, samples_loc):
    # based on:
    # https://github.com/TuragaLab/malis/blob/572ef0420107eee3c721bdafb58775a8a0fc467a/malis/malis.pyx
    # https://github.com/funkelab/gunpowder/blob/f202a6062dea65a4367a29816e8a2cc87d28b4b6/gunpowder/nodes/add_affinities.py
    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)

    shape = seg.shape[1:]
    nEdge = nhood.shape[0]

    aff = torch.zeros(
        (len(samples_loc), nEdge),
        dtype=torch.float32, device=device)

    patches = torch.zeros(
        (len(samples_loc),) + (ps, ps, ps),
        dtype=seg.dtype, device=device)
    for idx, sample_loc in enumerate(samples_loc):
        patches[idx] = seg[
            sample_loc[0],
            sample_loc[1]:sample_loc[1]+ps,
            sample_loc[2]:sample_loc[2]+ps,
            sample_loc[3]:sample_loc[3]+ps]

    for e in range(nEdge):

        center = patches[:, psH, psH, psH]
        offset = patches[:, nhood[e, 0], nhood[e, 1], nhood[e, 2]]
        # first == second pixel?
        t1 = (center == offset).to(torch.int32)
        t2 = seg[:, psH, psH, psH]
        t3 = seg[:, nhood[e, 0], nhood[e, 1], nhood[e, 2]]

        aff[:, e] = t1 * t2 * t3

    return aff


def seg_to_affgraph_2d_torch_code(seg, nhood, device, ps, psH, samples_loc):
    # based on:
    # https://github.com/TuragaLab/malis/blob/572ef0420107eee3c721bdafb58775a8a0fc467a/malis/malis.pyx
    # https://github.com/funkelab/gunpowder/blob/f202a6062dea65a4367a29816e8a2cc87d28b4b6/gunpowder/nodes/add_affinities.py
    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, y, x)
    # nhood.shape = (edges, 2)

    shape = seg.shape[1:]
    nEdge = nhood.shape[0]

    aff = torch.zeros(
        (len(samples_loc), nEdge),
        dtype=torch.float32, device=device)

    patches = torch.zeros(
        (len(samples_loc),) + (ps, ps),
        dtype=seg.dtype, device=device)
    for idx, sample_loc in enumerate(samples_loc):
        patches[idx] = seg[
            sample_loc[0],
            sample_loc[1]:sample_loc[1]+ps,
            sample_loc[2]:sample_loc[2]+ps]

    for e in range(nEdge):

        center = patches[:, psH, psH]
        offset = patches[:, nhood[e, 0], nhood[e, 1]]
        # first == second pixel?
        t1 = (center == offset).to(torch.int32)
        t2 = fg_seg[:, psH, psH]
        t3 = fg_seg[:, nhood[e, 0], nhood[e, 1]]

        aff[:, e] = t1 * t2 * t3

    return aff
