import logging

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def conv_pass(
    fmaps_in, *,
    kernel_size,
    num_fmaps,
    num_repetitions,
    activation,
    padding,
    name):
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


def global_average_pool(net, activation, name='global_average_pool'):
    net = tf.reduce_mean(
        net,
        axis=list(range(2, net.get_shape().ndims)),
        keep_dims=True,
        name=name
    )
    if activation is not None:
        activation = getattr(tf.nn, activation)
        net = activation(net)
    logger.info("global_average_pool activation %s", activation)
    return net


def dense(net, n, activation, regularizer=None, name=None):
    net = tf.layers.dense(net, n, activation=activation,
                          kernel_regularizer=regularizer, name=name)
    sums = []
    sums.append(tf.summary.histogram(net.op.name + '/activation', net))
    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)
    sums.append(tf.summary.histogram(net.op.name + '/weights', vars[0]))
    sums.append(tf.summary.histogram(net.op.name + '/bias', vars[0]))
    logger.info(net)
    return net, sums


def autoencoder(fmaps_in, **kwargs):
    if kwargs.get('only_decode'):
        net = kwargs['dummy_in']
    else:
        net = fmaps_in
    logger.info(net)

    regularizer = None
    if kwargs.get('regularizer') == 'l1':
        regularizer = tf.contrib.layers.l1_regularizer(
            scale=kwargs['regularizer_weight'])
    elif kwargs.get('regularizer') == 'l2':
        regularizer = tf.contrib.layers.l2_regularizer(
            scale=kwargs['regularizer_weight'])

    ps = np.prod(kwargs['patchshape'])
    is_training = kwargs['is_training']
    sums = []
    code = None

    if kwargs['network_type'] == 'dense':
        net = tf.layers.flatten(net, name="flatten_in")
        logger.info(net)

        for idx, units in enumerate(kwargs['encoder_units']):
            net, s = dense(net, units, kwargs['activation'],
                           regularizer=regularizer,
                           name="enc" + str(idx))
        sums += s
        net, s = dense(net, kwargs['code_units'], kwargs['code_activation'],
                       regularizer=regularizer,
                       name="enc" + str(idx + 1))
        code = net
        sums += s
        for idx, units in enumerate(kwargs['decoder_units']):
            net, s = dense(net, units, kwargs['activation'],
                           regularizer=regularizer,
                           name="dec" + str(idx))
            sums += s
        net, s = dense(net, ps, None, regularizer=regularizer,
                       name="dec" + str(idx + 1))
        sums += s
    elif kwargs['network_type'] == 'conv':
        num_channels = kwargs.get('num_channels', 1)
        net = tf.reshape(
            net,
            shape=(-1, num_channels) + kwargs['input_shape_squeezed'],
            name="to_patch")
        for idx, nf in enumerate(kwargs['num_fmaps']):
            net = conv_pass(
                net,
                num_fmaps=nf,
                kernel_size=kwargs['kernel_size'],
                num_repetitions=kwargs['num_repetitions'],
                activation=kwargs['activation'],
                padding=kwargs['padding'],
                name='encoder_layer_%i' % idx)
            logger.info(net)
            net = downsample(net, kwargs['downsample_factors'][idx],
                             name='downsample_%i' % idx,
                             padding=kwargs['padding'])
            logger.info(net)

        shape_in = net.get_shape().as_list()[1:]
        if kwargs['code_method'] == 'dense':
            net = tf.layers.flatten(net, name="flatten_in")
            logger.info(net)
            net, s = dense(net, kwargs['code_units'],
                           kwargs.get('code_activation'),
                           regularizer=regularizer,
                           name="fc_to_code")
            sums += s
            code = net
            if kwargs.get('only_decode'):
                net = fmaps_in
            net, s = dense(net, np.prod(shape_in),
                           kwargs['activation']
                           if kwargs.get('code_activation') else None,
                           regularizer=regularizer,
                           name="fc_from_code")
            sums += s
            net = tf.reshape(net, [-1] + shape_in, name="deflatten_out")
            logger.info(net)
        elif kwargs['code_method'] == 'global_average_pool':
            in_shape = net.get_shape().as_list()[2:]
            net = global_average_pool(
                net, kwargs.get('code_activation', 'sigmoid'),
                name='global_average_pool')

            code = net
            logger.info(net)

            if kwargs.get('only_decode'):
                net = tf.reshape(fmaps_in, [-1, kwargs['code_units'], 1, 1])
                logger.info(net)

            net = upsample(net, in_shape, kwargs['num_fmaps'][-1],
                           name='upsample_code',
                           upsampling=kwargs['upsampling'],
                           padding=kwargs['padding']
                           )
            logger.info(net)
        elif kwargs['code_method'] == 'conv1x1':
            net = conv_pass(
                net,
                num_fmaps=7,
                kernel_size=1,
                num_repetitions=1,
                activation=kwargs.get('code_activation'),
                padding=kwargs['padding'],
                name='to_code_layer')
            logger.info(net)

            code = tf.layers.flatten(net, name='code')
            logger.info(code)

            if kwargs.get('only_decode'):
                shape = net.get_shape().as_list()[1:]
                net = fmaps_in
                net = tf.reshape(net, [-1] + shape, name="deflatten_out")
                logger.info(net)
        elif kwargs['code_method'] == 'conv1x1_b':
            code_fmaps = kwargs.get('code_fmaps', 7)
            net = conv_pass(
                net,
                num_fmaps=code_fmaps,
                kernel_size=1,
                num_repetitions=1,
                activation=kwargs.get('code_activation'),
                padding=kwargs['padding'],
                name='to_code_layer')
            logger.info(net)

            code = tf.layers.flatten(net, name='code')
            logger.info(code)

            if kwargs.get('only_decode'):
                shape = net.get_shape().as_list()[1:]
                net = fmaps_in
                net = tf.reshape(net, [-1] + shape, name="deflatten_out")
                logger.info(net)

            net = conv_pass(
                net,
                num_fmaps=kwargs['num_fmaps'][-1],
                kernel_size=1,
                num_repetitions=1,
                activation=kwargs['activation'],
                padding=kwargs['padding'],
                name='from_code_layer')
            logger.info(net)

        elif kwargs['code_method'] == 'conv':
            fm = kwargs['num_fmaps'][-1]
            net = conv_pass(
                net,
                num_fmaps=fm,
                kernel_size=kwargs['kernel_size'],
                num_repetitions=1,
                activation=kwargs.get('code_activation'),
                padding=kwargs['padding'],
                name='to_code_layer')
            logger.info(net)

            code = tf.layers.flatten(net, name='code')
            logger.info(code)

            if kwargs.get('only_decode'):
                shape = net.get_shape().as_list()[1:]
                net = fmaps_in
                net = tf.reshape(net, [-1] + shape, name="deflatten_out")
                logger.info(net)

        elif kwargs['code_method'] == 'conv_sym':
            fm = kwargs['num_fmaps'][-1]
            net = conv_pass(
                net,
                num_fmaps=fm,
                kernel_size=kwargs['kernel_size'],
                num_repetitions=1,
                activation=kwargs.get('code_activation'),
                padding=kwargs['padding'],
                name='to_code_layer')
            logger.info(net)

            code = tf.layers.flatten(net, name='code')
            logger.info(code)

            if kwargs.get('only_decode'):
                shape = net.get_shape().as_list()[1:]
                net = fmaps_in
                net = tf.reshape(net, [-1] + shape, name="deflatten_out")
                logger.info(net)

            net = conv_pass(
                net,
                num_fmaps=fm,
                kernel_size=kwargs['kernel_size'],
                num_repetitions=1,
                activation=kwargs['activation'],
                padding=kwargs['padding'],
                name='from_code_layer')
            logger.info(net)

        else:
            raise NotImplementedError

        for idx, nf in enumerate(list(reversed(kwargs['num_fmaps']))[1:] + [1]):
            net = upsample(net, kwargs['downsample_factors'][-idx],
                           nf,
                           kwargs['activation'],
                           name='upsample_%i' % idx,
                           upsampling=kwargs['upsampling'],
                           padding=kwargs['padding'])

            net = conv_pass(
                net,
                num_fmaps=nf,
                kernel_size=kwargs['kernel_size'],
                num_repetitions=kwargs['num_repetitions'],
                activation=None if nf == 1 else kwargs['activation'],
                padding=kwargs['padding'],
                name='decoder_layer_%i' % idx)
            logger.info(net)
        shape_out = net.get_shape().as_list()
        offset = [0] * len(shape_out)
        if kwargs.get('output_shape'):
            output_shape = kwargs['output_shape']
            offset = list([(i - s) // 2 for i, s in zip(
                kwargs['input_shape_squeezed'], output_shape)
            ])
            offset = [0, 0] + offset
        else:
            output_shape = kwargs['input_shape_squeezed']
        net = tf.slice(net, offset, (-1, 1) + tuple(output_shape),
                       name='crop')
        logger.info(net)
    else:
        raise RuntimeError("invalid network type")

    return net, sums, code


def decoder(fmaps_in, **kwargs):
    regularizer = None
    if kwargs.get('regularizer') == 'l1':
        regularizer = tf.contrib.layers.l1_regularizer(
            scale=kwargs['regularizer_weight'])
    elif kwargs.get('regularizer') == 'l2':
        regularizer = tf.contrib.layers.l2_regularizer(
            scale=kwargs['regularizer_weight'])

    ps = np.prod(kwargs['patchshape'])
    is_training = kwargs['is_training']
    sums = []
    code = None

    if kwargs['network_type'] == 'conv':
        if kwargs['code_method'] == 'conv1x1_b':
            shape = kwargs['code_reshape']
            net = fmaps_in
            net = tf.reshape(net, [-1] + shape, name="deflatten_out")
            logger.info(net)

            net = conv_pass(
                net,
                num_fmaps=kwargs['num_fmaps'][-1],
                kernel_size=1,
                num_repetitions=1,
                activation=kwargs['activation'],
                padding=kwargs['padding'],
                name='from_code_layer'
            )
            logger.info(net)
        else:
            raise NotImplementedError

        for idx, nf in enumerate(list(reversed(kwargs['num_fmaps']))[1:] + [1]):
            idx += 1

            net = upsample(net, kwargs['downsample_factors'][-idx],
                           nf,
                           kwargs['activation'],
                           name='upsample_%i' % idx,
                           upsampling=kwargs['upsampling'],
                           padding=kwargs['padding'])
            net = conv_pass(
                net,
                num_fmaps=nf,
                kernel_size=kwargs['kernel_size'],
                num_repetitions=kwargs['num_repetitions'][-idx],
                activation=None if nf == 1 else kwargs['activation'],
                padding=kwargs['padding'],
                name='decoder_layer_%i' % idx)
            logger.info(net)
        shape_out = net.get_shape().as_list()
        offset = [0] * len(shape_out)
        net = tf.slice(net, offset, (-1, 1) + kwargs['input_shape_squeezed'],
                       name='crop')
        logger.info(net)
    else:
        raise RuntimeError("invalid network type")

    return net, sums, code
