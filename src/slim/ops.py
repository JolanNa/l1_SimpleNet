from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from slim.activation_functions import *


@tf.contrib.framework.add_arg_scope
def conv2d_multi_act(inputs,
                     filter_shape,
                     b_shape=[-1],
                     strides=[1, 1, 1, 1],
                     padding='SAME',
                     bias_init=0.1,
                     AF=None,
                     AF_set=None,
                     AF_weights_init='default',
                     W_blend_trainable=True,
                     AF_blend_mode='unrestricted',
                     swish_beta_trainable=True,
                     preblend_batchnorm=False,
                     reuse=False,
                     varscope=None):
    """

    """

    with tf.variable_scope(varscope, reuse=reuse):
        if b_shape == [-1]:
            b_shape = [filter_shape[-1]]
        filter_initializer = tf.truncated_normal_initializer(stddev=tf.sqrt(2. / (filter_shape[0] * filter_shape[1] * filter_shape[2])))  # stddev=0.1
        filter = tf.get_variable('filter', filter_shape, initializer=filter_initializer)
        b = tf.get_variable('biases', b_shape, initializer=tf.constant_initializer(bias_init))
        state = tf.nn.conv2d(inputs, filter, strides, padding)
        if b_shape != [0]:
            state += b
        if not AF is None:
            return AF(state)
        if AF_set is None:
            return state
        return multi_activation(state, AF_set, AF_weights_init)

@tf.contrib.framework.add_arg_scope
def dense_multi_act(inputs,
                      W_shape,
                      b_shape=[-1],
                      bias_init=0.1, AF=None, AF_set=None,
                    AF_weights_init='default', W_blend_trainable=True, AF_blend_mode='unrestricted',
                      swish_beta_trainable=True, preblend_batchnorm=False, reuse=False, varscope=None):
    """

    """
    with tf.variable_scope(varscope, reuse=reuse):
        flat_input = tf.layers.flatten(inputs)
        input_dims = flat_input.get_shape().as_list()[1]
        W_shape = [input_dims, W_shape[0]]
        if b_shape == [-1]:
            b_shape = [W_shape[-1]]
        W = tf.get_variable('weights', W_shape, initializer=tf.truncated_normal_initializer(
            stddev=tf.sqrt(2. / (W_shape[0] * W_shape[1]))))  # stddev=0.1
        b = tf.get_variable('biases', b_shape, initializer=tf.constant_initializer(bias_init))
        state = tf.matmul(flat_input, W)
        if b_shape != [0]:
            state += b
        if not AF is None:
            return AF(state)
        if AF_set is None:
            return state
        return multi_activation(state, AF_set, AF_weights_init)


@tf.contrib.framework.add_arg_scope
def multi_activation(drive, af_set, af_weights_init):
    """
    """
    activations = []
    shape = drive.get_shape().as_list()
    # print(tf.get_variable_scope().name)
    for af, af_name in af_set:
        weight = tf.get_variable('act_weight/' + af_name, initializer=af_weights_init[af_name](shape))
        if af_name == "swish":
            # print(af_inits[af_name](shape))
            beta = tf.get_variable('act_bias/' + af_name, initializer=af_weights_init[af_name](shape))
            tf.summary.scalar('act_bias/swish_beta', beta)
            activations.append(weight * af(drive, beta))
            # tf.summary.scalar('act_weight/'+name+"/test"+af_name, activations[-1][0])
        else:
            activations.append(weight * af(drive))

        tf.summary.scalar('act_weight/' + af_name, weight)

    return tf.add_n(activations)