from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from slim.activation_functions import *

FLAGS = tf.app.flags.FLAGS


def _two_element_tuple(int_or_tuple):
  """Converts `int_or_tuple` to height, width.

  Several of the functions that follow accept arguments as either
  a tuple of 2 integers or a single integer.  A single integer
  indicates that the 2 values of the tuple are the same.

  This functions normalizes the input value by always returning a tuple.

  Args:
    int_or_tuple: A list of 2 ints, a single int or a tf.TensorShape.

  Returns:
    A tuple with 2 values.

  Raises:
    ValueError: If `int_or_tuple` it not well formed.
  """
  if isinstance(int_or_tuple, (list, tuple)):
    if len(int_or_tuple) != 2:
      raise ValueError('Must be a list with 2 elements: %s' % int_or_tuple)
    return int(int_or_tuple[0]), int(int_or_tuple[1])
  if isinstance(int_or_tuple, int):
    return int(int_or_tuple), int(int_or_tuple)
  if isinstance(int_or_tuple, tf.TensorShape):
    if len(int_or_tuple) == 2:
      return int_or_tuple[0], int_or_tuple[1]
  raise ValueError('Must be an int, a list with 2 elements or a TensorShape of '
                   'length 2')


@tf.contrib.framework.add_arg_scope
def multi_activation(drive, af_set, af_weights_init):
    """
    """
    #
    activations = []
    shape = drive.get_shape().as_list()
    weights = []
    # print(tf.get_variable_scope().name)
    for af, af_name in af_set:
        if FLAGS.random_sample_activation_weights:
            weight = tf.get_variable('act_weight/' + af_name, shape=[], initializer=tf.truncated_normal_initializer(mean=0.2, stddev=FLAGS.activation_weight_stdv))
        else:
            weight = tf.get_variable('act_weight/' + af_name, initializer=0.2)
        weights.append(weight)
        if af_name == "swish":
            beta = tf.get_variable('act_bias/' + af_name, initializer=af_weights_init[af_name](shape))
            tf.summary.scalar('act_bias/swish_beta', beta)
            activations.append(weight * af(drive, beta))
        else:
            activations.append(weight * af(drive))

        tf.summary.scalar('act_weight/' + af_name, weight)


    tf.add_to_collection("L1_WEIGHTS_COLLECTION", weights)

    print("Multi Activated: " + str(tf.get_variable_scope().name))
    return tf.add_n(activations)

@tf.contrib.framework.add_arg_scope
def conv2d_multi_act_old(inputs,
           num_filters_out,
           kernel_size,
           stride=1,
           padding='SAME',
           AF=None,
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
           reuse=None,
           weights_shape=None):
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

  # with tf.variable_scope(scope, reuse=reuse):
  #     kernel_h, kernel_w = _two_element_tuple(kernel_size)
  #     stride_h, stride_w = _two_element_tuple(stride)
  #     num_filters_in = inputs.get_shape()[-1]
  #     weights_shape = [kernel_h, kernel_w, num_filters_in, num_filters_out]
  #     weights_shape = [5,5,3,16]
  #
  #     weights_initializer = tf.truncated_normal_initializer(stddev=stddev)
  #     l2_regularizer = None
  #     #   # if weight_decay and weight_decay > 0:
  #     #   #   l2_regularizer = losses.l2_regularizer(weight_decay)
  #     weights = tf.get_variable('weights',
  #                                    shape=weights_shape,
  #                                    # regularizer=l2_regularizer,
  #
  #                                    initializer=weights_initializer)
  #
  #
  #     filter_shape = weights_shape
  #     strides = [1, stride_h, stride_w, 1]
  #     b_shape = [num_filters_out,]
  #     bias_init = bias
  #     filter_initializer = weights_initializer
  #
  #
  #     # filter_initializer = tf.truncated_normal_initializer(
  #     #     stddev=tf.sqrt(2. / (filter_shape[0] * filter_shape[1] * filter_shape[2])))  # stddev=0.1
  #     filter = tf.get_variable('filter', filter_shape, initializer=filter_initializer)
  #     filter = weights
  #
  #     b = tf.get_variable('biases', b_shape, initializer=tf.constant_initializer(bias_init))
  #     state = tf.nn.conv2d(inputs, filter, strides, padding)
  #     if b_shape != [0]:
  #         state += b
  #     if not AF is None:
  #         return AF(state)
  #     if AF_set is None:
  #         return state
  #     return multi_activation(state, AF_set, AF_weights_init)

  with tf.variable_scope(scope, reuse=reuse):
    kernel_h, kernel_w = _two_element_tuple(kernel_size)
    stride_h, stride_w = _two_element_tuple(stride)
    # num_filters_in = inputs.get_shape()[-1]
    if weights_shape is None:
        num_filters_in = inputs.get_shape().as_list()[-1]

        weights_shape = [kernel_h, kernel_w,
                         num_filters_in, num_filters_out]
    weights_initializer = tf.truncated_normal_initializer(stddev=stddev)
    l2_regularizer = None
    # if weight_decay and weight_decay > 0:
    #   l2_regularizer = losses.l2_regularizer(weight_decay)
    weights = tf.get_variable('weights',
                                 shape=weights_shape,
                                 initializer=weights_initializer,
                                 # regularizer=l2_regularizer,
                                 trainable=trainable)

    # print("\nweight_shape -> " + str(weights_shape) + "\n" + "strides ->" + str([1, stride_h, stride_w, 1]) + "\n")

    conv = tf.nn.conv2d(inputs, weights, [1, stride_h, stride_w, 1],
                        padding=padding)
    # if batch_norm_params is not None:
    #   with tf.contrib.framework.arg_scope([batch_norm], is_training=is_training,
    #                         trainable=trainable, restore=restore):
    #     outputs = batch_norm(conv, **batch_norm_params)
    # else:
    bias_shape = [num_filters_out,]
    bias_initializer = tf.constant_initializer(bias)
    biases = tf.get_variable('biases',
                                  shape=bias_shape,
                                  initializer=bias_initializer,
                                  trainable=trainable)
    outputs = tf.nn.bias_add(conv, biases)

    # return tf.nn.relu(outputs)

    if AF_set is None:
        if AF is None:
            return outputs
        else:
            if AF == swish:
                print("is swish")
            return AF(outputs)

    return multi_activation(outputs, AF_set, AF_weights_init)

    # if not AF is None:
    #     return AF(outputs)
    # if AF_set is None:
    #     return outputs
    # return multi_activation(outputs, AF_set, AF_weights_init)


@tf.contrib.framework.add_arg_scope
def fc_multi_act(inputs,
       num_units_out,
       AF=tf.nn.relu,
       AF_set=None,
       AF_weights_init=None,
       stddev=0.1,
       bias=0.0,
       weight_decay=0,
       batch_norm_params=None,
       is_training=True,
       trainable=True,
       restore=True,
       scope=None,
       reuse=None):
  """Adds a fully connected layer followed by an optional batch_norm layer.

  FC creates a variable called 'weights', representing the fully connected
  weight matrix, that is multiplied by the input. If `batch_norm` is None, a
  second variable called 'biases' is added to the result of the initial
  vector-matrix multiplication.

  Args:
    inputs: a [B x N] tensor where B is the batch size and N is the number of
            input units in the layer.
    num_units_out: the number of output units in the layer.
    activation: activation function.
    stddev: the standard deviation for the weights.
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
     the tensor variable representing the result of the series of operations.
  """
  with tf.variable_scope(scope, 'FC', [inputs], reuse=reuse):
    # num_units_in = inputs.get_shape()[1]
    num_units_in = inputs.get_shape().as_list()[1]

    weights_shape = [num_units_in, num_units_out]
    weights_initializer = tf.truncated_normal_initializer(stddev=stddev)
    l2_regularizer = None
    if weight_decay and weight_decay > 0:
      l2_regularizer = losses.l2_regularizer(weight_decay)
    weights = tf.get_variable('weights',
                                 shape=weights_shape,
                                 initializer=weights_initializer,
                                 regularizer=l2_regularizer,
                                 trainable=trainable)
    if batch_norm_params is not None:
      outputs = tf.matmul(inputs, weights)
      with tf.contrib.framework.arg_scope([batch_norm], is_training=is_training,
                            trainable=trainable):
        outputs = batch_norm(outputs, **batch_norm_params)
    else:
      bias_shape = [num_units_out,]
      bias_initializer = tf.constant_initializer(bias)
      biases = tf.get_variable('biases',
                                  shape=bias_shape,
                                  initializer=bias_initializer,
                                  trainable=trainable)
      outputs = tf.nn.xw_plus_b(inputs, weights, biases)
    # if AF:
    #   outputs = AF(outputs)
    # return outputs
    if AF_set is None:
        if AF is None:
            return outputs
        else:
            return AF(outputs)

    return multi_activation(outputs, AF_set, AF_weights_init)
    # if not AF is None:
    #   return AF(outputs)
    # if AF_set is None:
    #   return outputs
    # return multi_activation(outputs, AF_set, AF_weights_init)


def flatten(inputs, scope=None):
  """Flattens the input while maintaining the batch_size.

    Assumes that the first dimension represents the batch.

  Args:
    inputs: a tensor of size [batch_size, ...].
    scope: Optional scope for name_scope.

  Returns:
    a flattened tensor with shape [batch_size, k].
  Raises:
    ValueError: if inputs.shape is wrong.
  """
  if len(inputs.get_shape()) < 2:
    raise ValueError('Inputs must be have a least 2 dimensions')
  dims = inputs.get_shape()[1:]
  k = dims.num_elements()
  with tf.name_scope(scope, 'Flatten', [inputs]):
    return tf.reshape(inputs, [-1, k])

