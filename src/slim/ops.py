from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from slim.activation_functions import *



def conv_layer_mulit_act(layer_input, filter_shape, b_shape=[-1], strides=[1, 1, 1, 1], padding='SAME',
                         bias_init=0.1, AF=None, AF_set=None, af_weights_init='default', W_blend_trainable=True,
                         AF_blend_mode='unrestricted', swish_beta_trainable=True, preblend_batchnorm=False, reuse=False,
                         varscope=None):
    """

    """
    with tf.variable_scope(varscope, reuse=reuse):
        if b_shape == [-1]:
            b_shape = [filter_shape[-1]]
        filter_initializer = tf.truncated_normal_initializer(stddev=tf.sqrt(2. / (filter_shape[0] * filter_shape[1] * filter_shape[2])))  # stddev=0.1
        filter = tf.get_variable('filter', filter_shape, initializer=filter_initializer)
        b = tf.get_variable('biases', b_shape, initializer=tf.constant_initializer(bias_init))
        state = tf.nn.conv2d(layer_input, filter, strides, padding)
        if b_shape != [0]:
            state += b
        if not AF is None:
            return AF(state)
        if AF_set is None:
            return state
        return activate(state, AF_set, af_weights_init, varscope)

def conv2d_multi_act(inputs,
                     num_filters_out,
                     kernel_size,
                     stride=1,
                     padding='SAME',
                     activation=None,
                     multi_activation=None,
                     AF_set=None,
                     AF_weights_init=None,
                     stddev=0.01,
                     bias=0.0,
                     weight_decay=0,
                     batch_norm_params=None,
                     is_training=True,
                     trainable=True,
                     restore=True,
                     scope=None,
                     reuse=None):
    """Adds a 2D convolution followed by an optional batch_norm layer.

    conv2d creates a variable called 'weights', representing the convolutional
    kernel, that is convolved with the input. If `batch_norm_params` is None, a
    second variable called 'biases' is added to the result of the convolution
    operation.

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_filters_out: the number of output filters.
      kernel_size: a list of length 2: [kernel_height, kernel_width] of
        of the filters. Can be an int if both values are the same.
      stride: a list of length 2: [stride_height, stride_width].
        Can be an int if both strides are the same.  Note that presently
        both strides must have the same value.
      padding: one of 'VALID' or 'SAME'.
      activation: activation function.
      stddev: standard deviation of the truncated guassian weight distribution.
      bias: the initial value of the biases.
      weight_decay: the weight decay.
      batch_norm_params: parameters for the batch_norm. If is None don't use it.
      is_training: whether or not the model is in training mode.
      trainable: whether or not the variables should be trainable or not.
      restore: whether or not the variables should be marked for restore.
      scope: Optional scope for variable_scope.
      reuse: whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
    Returns:
      a tensor representing the output of the operation.

    """
    with tf.variable_scope(scope, 'Conv', [inputs], reuse=reuse):
        kernel_h, kernel_w = _two_element_tuple(kernel_size)
        stride_h, stride_w = _two_element_tuple(stride)
        num_filters_in = inputs.get_shape()[-1]
        weights_shape = [kernel_h, kernel_w,
                         num_filters_in, num_filters_out]
        weights_initializer = tf.truncated_normal_initializer(stddev=stddev)
        l2_regularizer = None
        if weight_decay and weight_decay > 0:
            l2_regularizer = losses.l2_regularizer(weight_decay)
        weights = variables.variable('weights',
                                     shape=weights_shape,
                                     initializer=weights_initializer,
                                     regularizer=l2_regularizer,
                                     trainable=trainable,
                                     restore=restore)
        conv = tf.nn.conv2d(inputs, weights, [1, stride_h, stride_w, 1],
                            padding=padding)
        if batch_norm_params is not None:
            with scopes.arg_scope([batch_norm], is_training=is_training,
                                  trainable=trainable, restore=restore):
                outputs = batch_norm(conv, **batch_norm_params)
        else:
            bias_shape = [num_filters_out, ]
            bias_initializer = tf.constant_initializer(bias)
            biases = variables.variable('biases',
                                        shape=bias_shape,
                                        initializer=bias_initializer,
                                        trainable=trainable,
                                        restore=restore)
            outputs = tf.nn.bias_add(conv, biases)
        if not activation is None:
            outputs = activation(outputs)
        elif not AF_set is None:
            return multi_activation(outputs, AF_set, AF_weights_init)
        return outputs

def dense_multi_act_layer(layer_input, W_shape, b_shape=[-1], bias_init=0.1, AF=None, AF_set=None,
                          af_weights_init='default', W_blend_trainable=True, AF_blend_mode='unrestricted',
                          swish_beta_trainable=True, preblend_batchnorm=False, reuse=False, varscope=None):
    """

    """
    with tf.variable_scope(varscope, reuse=reuse):
        flat_input = tf.layers.flatten(layer_input)
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
        return activate(state, AF_set, af_weights_init, varscope)