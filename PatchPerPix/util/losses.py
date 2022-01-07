import sys

import tensorflow as tf


def get_loss_fn(loss):
    if loss == "mse":
        loss_fn = tf.losses.mean_squared_error
    elif loss == "ce":
        loss_fn = tf.losses.sigmoid_cross_entropy
    elif loss == "ssce":
        loss_fn = tf.losses.sparse_softmax_cross_entropy
    else:
        raise ValueError("invalid loss function", loss)
    return loss_fn


def get_loss(gt, pred, loss_type, name, do_sigmoid, do_tanh=False):
    loss_fn = get_loss_fn(loss_type)

    if do_sigmoid and loss_type == "mse":
        pred = tf.sigmoid(pred)
    if do_tanh:
        assert loss_type == "mse", "tanh only with mse loss"
        assert not do_sigmoid, "either sigmoid or tanh for net output"
        pred = tf.tanh(pred)
    loss = loss_fn(
        gt,
        pred)
    if do_sigmoid and loss_type == "ce":
        pred = tf.sigmoid(pred)

    return loss, pred, get_loss_print(loss, name)


def get_loss_weighted(gt, pred, loss_weights, loss_type, name, do_sigmoid):
    loss_fn = get_loss_fn(loss_type)

    if do_sigmoid and loss_type == "mse":
        pred = tf.sigmoid(pred)
    loss_weighted = loss_fn(
        gt,
        pred,
        loss_weights)
    if do_sigmoid and loss_type == "ce":
        pred = tf.sigmoid(pred)

    return loss_weighted, pred, \
        get_loss_print(loss_weighted, name, loss_weights)


def get_loss_print(loss, name, loss_weights=None):
    if loss_weights is None:
        print_loss_op = tf.print(
            name+" (nonweighted):",
            loss,
            output_stream=sys.stdout)
        return [print_loss_op]
    else:
        print_loss_op = tf.print(
            name+" (weighted):",
            loss,
            output_stream=sys.stdout)
        print_loss_weights_op = tf.print(
            name+"-loss weights:",
            tf.reduce_sum(loss_weights),
            output_stream=sys.stdout)
        return [print_loss_op, print_loss_weights_op]
