import tensorflow as tf
import numpy as np


def conv_pass(
    fmaps_in, *,
    kernel_size,
    num_fmaps,
    num_repetitions,
    activation,
    padding,
    name,
    shortcut=False):
    '''Create a convolution pass::

        f_in --> f_1 --> ... --> f_n

    where each ``-->`` is a convolution followed by a (non-linear) activation
    function and ``n`` ``num_repetitions``. Each convolution will decrease the
    size of the feature maps by ``kernel_size-1``.

    Args:

        f_in:

            The input tensor of shape ``(batch_size, channels, depth, height,
            width)`` or ``(batch_size, channels, height, width)``.

        kernel_size:

            Size of the kernel. Forwarded to the tensorflow convolution layer.

        num_fmaps:

            The number of feature maps to produce with each convolution.

        num_repetitions:

            How many convolutions to apply.

        activation:

            Which activation to use after a convolution. Accepts the name of any
            tensorflow activation function (e.g., ``relu`` for ``tf.nn.relu``).

    '''

    fmaps = fmaps_in
    if activation is not None:
        activation = getattr(tf.nn, activation)

    conv_layer = getattr(
        tf.layers,
        {2: 'conv2d', 3: 'conv3d'}[fmaps_in.get_shape().ndims - 2])

    for i in range(num_repetitions):
        fmaps = conv_layer(
            inputs=fmaps,
            filters=num_fmaps,
            kernel_size=kernel_size,
            padding=padding,
            data_format='channels_first',
            activation=activation,
            name=name + '_%i' % i)

    if shortcut:
        fmaps_in = crop_spatial(fmaps_in, fmaps.get_shape().as_list())
        fmaps = tf.concat([fmaps, fmaps_in], 1)
    return fmaps


def downsample(fmaps_in, factors, name='down', padding='valid'):
    pooling_layer = getattr(
        tf.layers,
        {2: 'max_pooling2d', 3: 'max_pooling3d'}[
            fmaps_in.get_shape().ndims - 2])

    fmaps = pooling_layer(
        fmaps_in,
        pool_size=factors,
        strides=factors,
        padding=padding,
        data_format='channels_first',
        name=name)

    return fmaps


def upsample(fmaps_in, factors, num_fmaps,
             activation='relu', name='up',
             padding='valid',
             upsampling="trans_conv"):
    if activation is not None:
        activation = getattr(tf.nn, activation)

    if upsampling == "resize_conv":
        conv_layer = getattr(
            tf.layers,
            {2: 'conv2d', 3: 'conv3d'}[fmaps_in.get_shape().ndims - 2])

        if fmaps_in.get_shape().ndims - 2 == 2:
            fmaps = tf.keras.backend.resize_images(
                fmaps_in,
                factors[0],
                factors[1],
                "channels_first")
        else:
            fmaps = tf.keras.backend.resize_volumes(
                fmaps_in,
                factors[0],
                factors[1],
                factors[2],
                "channels_first")

        # fmaps = conv_layer(
        #     inputs=fmaps,
        #     filters=num_fmaps,
        #     kernel_size=kernel_size,
        #     padding=padding,
        #     data_format='channels_first',
        #     activation=activation,
        #     name=name)
    elif upsampling == "trans_conv":
        conv_trans_layer = getattr(
            tf.layers,
            {2: 'conv2d_transpose', 3: 'conv3d_transpose'}[
                fmaps_in.get_shape().ndims - 2])

        fmaps = conv_trans_layer(
            fmaps_in,
            filters=num_fmaps,
            kernel_size=factors,
            strides=factors,
            padding=padding,
            data_format='channels_first',
            activation=activation,
            name=name)
    else:
        raise ValueError("invalid value for upsampling method", upsampling)

    return fmaps


def crop_to_factor(fmaps_in, factor, kernel_sizes):
    '''Crop feature maps to ensure translation equivariance with stride of
    upsampling factor. This should be done right after upsampling, before
    application of the convolutions with the given kernel sizes.
    The crop could be done after the convolutions, but it is more efficient to
    do that before (feature maps will be smaller).
    '''

    shape = fmaps_in.get_shape().as_list()
    spatial_dims = 3 if len(shape) == 5 else 4
    spatial_shape = shape[-spatial_dims:]

    # the crop that will already be done due to the convolutions
    convolution_crop = list(
        sum(
            (ks if isinstance(ks, int) else ks[d]) - 1
            for ks in kernel_sizes
        )
        for d in range(spatial_dims)
    )
    print("crop_to_factor: factor =", factor)
    print("crop_to_factor: kernel_sizes =", kernel_sizes)
    print("crop_to_factor: convolution_crop =", convolution_crop)

    # we need (spatial_shape - convolution_crop) to be a multiple of factor,
    # i.e.:
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
        int(np.floor(float(s - c)/f))
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
            "Feature map with shape %s is too small to ensure translation " \
            "equivariance with factor %s and following convolutions %s" % (
                shape,
                factor,
                kernel_sizes)

        target_shape = list(shape)
        target_shape[-spatial_dims:] = target_spatial_shape

        print("crop_to_factor: shape =", shape)
        print("crop_to_factor: spatial_shape =", spatial_shape)
        print("crop_to_factor: target_spatial_shape =", target_spatial_shape)
        print("crop_to_factor: target_shape =", target_shape)
        fmaps = crop(
            fmaps_in,
            target_shape)
    else:
        fmaps = fmaps_in

    return fmaps


def crop_spatial(fmaps_in, shape):
    '''Crop only the spacial dimensions to match shape.

    Args:

        fmaps_in:

            The input tensor.

        shape:

            A list (not a tensor) with the requested shape [_, _, z, y, x] or
            [_, _, y, x].
    '''

    in_shape = fmaps_in.get_shape().as_list()

    offset = [0, 0] + [(in_shape[i] - shape[i]) // 2 for i in
                       range(2, len(shape))]
    size = in_shape[0:2] + shape[2:]

    fmaps = tf.slice(fmaps_in, offset, size)
    return fmaps


def crop(a, shape):
    '''Crop a to a new shape, centered in a.

    Args:

        a:

            The input tensor.

        shape:

            A list (not a tensor) with the requested shape.
    '''

    in_shape = a.get_shape().as_list()

    offset = list([
        (i - s) // 2
        for i, s in zip(in_shape, shape)
    ])

    b = tf.slice(a, offset, shape)

    return b


def unet(
    fmaps_in, *,
    num_fmaps,
    fmap_inc_factors,
    fmap_dec_factors,
    downsample_factors,
    activation,
    padding,
    kernel_size,
    num_repetitions,
    upsampling="trans_conv",
    shortcut=False,
    kernel_size_down=None,
    kernel_size_up=None,
    crop_factor=False,
    layer=0):
    '''Create a 2D or 3D U-Net::

        f_in --> f_left --------------------------->> f_right--> f_out
                    |                                   ^
                    v                                   |
                 g_in --> g_left ------->> g_right --> g_out
                             |               ^
                             v               |
                                   ...

    where each ``-->`` is a convolution pass (see ``conv_pass``), each `-->>` a
    crop, and down and up arrows are max-pooling and transposed convolutions,
    respectively.

    The U-Net expects tensors to have shape ``(batch=1, channels, depth, height,
    width)`` for 3D or ``(batch=1, channels, height, width)`` for 2D.

    This U-Net performs only "valid" convolutions, i.e., sizes of the feature
    maps decrease after each convolution.

    Args:

        fmaps_in:

            The input tensor.

        num_fmaps:

            The number of feature maps in the first layer. This is also the
            number of output feature maps.

        fmap_inc_factor:

            By how much to multiply the number of feature maps between layers.
            If layer 0 has ``k`` feature maps, layer ``l`` will have
            ``k*fmap_inc_factor**l``.

        downsample_factors:

            List of lists ``[z, y, x]`` or ``[y, x]`` to use to down- and
            up-sample the feature maps between layers.

        activation:

            Which activation to use after a convolution. Accepts the name of any
            tensorflow activation function (e.g., ``relu`` for ``tf.nn.relu``).

        layer:

            Used internally to build the U-Net recursively.
    '''

    prefix = "    " * layer
    print(prefix + "Creating U-Net layer %i" % layer)
    print(prefix + "f_in: " + str(fmaps_in.shape))

    if kernel_size_down is None:
        kernel_size_down = [[3, 3]]*(len(downsample_factors) + 1)
    if kernel_size_up is None:
        kernel_size_up = [[3, 3]]*len(downsample_factors)

    # convolve
    f_left = conv_pass(
        fmaps_in,
        num_fmaps=num_fmaps,
        kernel_size=kernel_size,
        num_repetitions=num_repetitions,
        activation=activation,
        padding=padding,
        name='unet_layer_%i_left' % layer,
        shortcut=shortcut)
    print(prefix + "f_left: " + str(f_left.shape))

    # last layer does not recurse
    bottom_layer = (layer == len(downsample_factors))
    if bottom_layer:
        print(prefix + "bottom layer")
        print(prefix + "f_out: " + str(f_left.shape))
        return f_left

    # downsample
    g_in = downsample(
        f_left,
        downsample_factors[layer],
        name='unet_down_%i_to_%i' % (layer, layer + 1),
        padding=padding)

    # recursive U-net
    g_out = unet(
        g_in,
        num_fmaps=int(num_fmaps * fmap_inc_factors[layer]),
        fmap_inc_factors=fmap_inc_factors,
        fmap_dec_factors=fmap_dec_factors,
        downsample_factors=downsample_factors,
        activation=activation,
        padding=padding,
        kernel_size=kernel_size,
        num_repetitions=num_repetitions,
        upsampling=upsampling,
        shortcut=shortcut,
        kernel_size_down=kernel_size_down,
        kernel_size_up=kernel_size_up,
        crop_factor=crop_factor,
        layer=layer + 1,
    )

    print(prefix + "g_out: " + str(g_out.shape))

    num_fmaps_up = int(num_fmaps * np.prod(fmap_inc_factors[layer:]) / np.prod(
        fmap_dec_factors[layer:]))

    # upsample
    g_out_upsampled = upsample(
        g_out,
        downsample_factors[layer],
        num_fmaps=int(num_fmaps_up),
        activation=activation,
        name='unet_up_%i_to_%i' % (layer + 1, layer),
        upsampling=upsampling,
        padding=padding)

    print(prefix + "g_out_upsampled: " + str(g_out_upsampled.shape))

    if crop_factor:
        factor_product = None
        for factor in downsample_factors[layer:]:
            if factor_product is None:
                factor_product = list(factor)
            else:
                factor_product = list(
                    f*ff
                    for f, ff in zip(factor, factor_product))
            print("factor product", factor_product, factor)
        g_out_upsampled = crop_to_factor(
                    g_out_upsampled,
                    factor=factor_product,
                    kernel_sizes=kernel_size_up[layer])
        print(prefix + "g_out_upsampled factor crop: " + str(g_out_upsampled.shape))
        # copy-crop
    f_left_cropped = crop_spatial(f_left, g_out_upsampled.get_shape().as_list())

    print(prefix + "f_left_cropped: " + str(f_left_cropped.shape))

    # concatenate along channel dimension
    f_right = tf.concat([f_left_cropped, g_out_upsampled], 1)

    print(prefix + "f_right: " + str(f_right.shape))

    # convolve
    f_out = conv_pass(
        f_right,
        kernel_size=kernel_size,
        num_fmaps=num_fmaps_up,
        num_repetitions=num_repetitions,
        activation=activation,
        padding=padding,
        name='unet_layer_%i_right' % layer,
        shortcut=shortcut)

    print(prefix + "f_out: " + str(f_out.shape))

    return f_out
