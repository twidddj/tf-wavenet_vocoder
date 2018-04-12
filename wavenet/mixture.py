# coding:utf-8
"""
the code is adapted from:
https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
https://github.com/r9y9/wavenet_vocoder/blob/master/wavenet_vocoder/mixture.py
https://github.com/azraelkuan/tensorflow_wavenet_vocoder/tree/dev
"""
import tensorflow as tf
import numpy as np

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keepdims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x - m2), axis))


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis, keepdims=True)
    return x - m - tf.log(tf.reduce_sum(tf.exp(x - m), axis, keepdims=True))


def discretized_mix_logistic_loss(y_hat, y, num_class=256, log_scale_min=float(np.log(1e-14)), reduce=True):
    """
    Discretized mixture of logistic distributions loss
    :param y_hat: B x T x C
    :param y:  B x T x 1
    :param num_class:
    :param log_scale_min:
    :param reduce:
    :return: loss
    """
    y_hat_shape = y_hat.get_shape().as_list()

    assert len(y_hat_shape) == 3
    assert y_hat_shape[2] % 3 == 0

    nr_mix = y_hat_shape[2] // 3

    # unpack parameters
    logit_probs = y_hat[:, :, :nr_mix]
    means = y_hat[:, :, nr_mix:2 * nr_mix]
    log_scales = tf.maximum(y_hat[:, :, nr_mix * 2:nr_mix * 3], log_scale_min)

    # B x T x 1 => B x T x nr_mix
    y = tf.tile(y, [1, 1, nr_mix])

    centered_y = y - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_y + 1. / (num_class - 1))
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_y - 1. / (num_class - 1))
    cdf_min = tf.nn.sigmoid(min_in)

    log_cdf_plus = plus_in - tf.nn.softplus(plus_in)  # log probability for edge case of 0 (before scaling)

    log_one_minus_cdf_min = -tf.nn.softplus(min_in)  # log probability for edge case of 255 (before scaling)

    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_y

    log_pdf_mid = mid_in - log_scales - 2. * tf.nn.softplus(mid_in)

    log_probs = tf.where(y < -0.999, log_cdf_plus,
                         tf.where(y > 0.999, log_one_minus_cdf_min,
                                  tf.where(cdf_delta > 1e-5,
                                           tf.log(tf.maximum(cdf_delta, 1e-12)),
                                           log_pdf_mid - np.log((num_class - 1) / 2))))

    log_probs = log_probs + tf.nn.log_softmax(logit_probs, -1)
    # log_probs = log_probs + log_prob_from_logits(logit_probs)

    if reduce:
        return -tf.reduce_sum(log_sum_exp(log_probs))
    else:
        return -log_sum_exp(log_probs)


def sample_from_discretized_mix_logistic(y, log_scale_min=float(np.log(1e-14))):
    """

    :param y: B x T x C
    :param log_scale_min:
    :return: [-1, 1]
    """
    y_shape = y.get_shape().as_list()

    assert len(y_shape) == 3
    assert y_shape[2] % 3 == 0
    nr_mix = y_shape[2] // 3

    logit_probs = y[:, :, :nr_mix]

    sel = tf.one_hot(
        tf.argmax(
            logit_probs - tf.log(-tf.log(
                tf.random_uniform(
                    tf.shape(logit_probs), minval=1e-5, maxval=1. - 1e-5))), 2),
        depth=nr_mix, dtype=tf.float32)

    means = tf.reduce_sum(y[:, :, nr_mix:nr_mix * 2] * sel, axis=2)

    log_scales = tf.maximum(tf.reduce_sum(y[:, :, nr_mix * 2:nr_mix * 3] * sel, axis=2), log_scale_min)

    u = tf.random_uniform(tf.shape(means), minval=1e-5, maxval=1. - 1e-5)
    x = means + tf.exp(log_scales) * (tf.log(u) - tf.log(1. - u))

    x = tf.minimum(tf.maximum(x, -1.), 1.)
    return x
