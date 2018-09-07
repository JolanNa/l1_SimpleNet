from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


import slim.ops as ops
from slim.activation_functions import *



def build_ianntf_network(X):
    """

    """
    AF_set = [
        [swish, 'swish'],
        [tf.nn.relu, 'relu'],
        [tf.nn.elu, 'elu'],
        [tf.nn.tanh, 'tanh'],
        [identity_activation, 'identity']
    ]

    AF_weights_init = {'swish': lambda fan_in: 0.1,  # swish
                       'relu': lambda fan_in: 2.0 / fan_in[2] if len(fan_in) == 4 else 2.0 / fan_in[1],  # relu
                       'elu': lambda fan_in: 0.1,  # elu
                       'tanh': lambda fan_in: fan_in[2] ** (-1 / 2) if len(fan_in) == 4 else fan_in[1] ** (-1 / 2),
                       # tanh
                       'identity': lambda fan_in: 0.1  # identity
                       }

    l1_groups = ['conv_01', 'conv_02', 'dense_01']

    with tf.name_scope('ianntf_model'):

        with tf.contrib.framework.arg_scope([ops.conv2d_multi_act, ops.dense_multi_act],
                              AF_set=AF_set, AF_weights_init=AF_weights_init):

            state = ops.conv2d_multi_act(inputs=X, AF=None, filter_shape=[5 , 5 , 3, 16], strides=[1 , 1 , 1 , 1], padding='SAME', bias_init=0.0, varscope='conv_01')
            # state = ops.conv2d_multi_act(inputs=X, kernel_size=5, num_filters_out=16, AF=tf.nn.relu, stride=1, padding='SAME', bias=0.0, scope='conv_01')

            state = tf.nn.max_pool(state, ksize=[1 ,2 ,2 ,1], strides=[1 ,2 ,2 ,1], padding='SAME')

            state = ops.conv2d_multi_act(inputs=state, AF=None, filter_shape=[3 , 3 , 16 , 32], strides=[1 , 1 , 1 , 1], padding='SAME', varscope='conv_02')
            # state = ops.conv2d_multi_act(inputs=state, kernel_size=3, num_filters_out=32, AF=tf.nn.relu, stride=1, padding='SAME', bias=0.0, scope='conv_02')

            state = tf.nn.max_pool(state, ksize=[1 ,3 ,3 ,1], strides=[1 ,2 ,2 ,1], padding='SAME')

            state = ops.dense_multi_act(inputs=state, AF=None, W_shape=[512], varscope='dense_01')

            logits = ops.dense_multi_act(inputs=state, AF=None, AF_set=None, W_shape=[10], varscope='dense_02')

            return logits, l1_groups

def build_simple_network(X):
    l1_groups = ['conv_01',
                 'conv_02',
                 'conv_03',
                 'conv_04',
                 'dense_01',
                 'dense_02']
    # conv1
    state = ops.conv2d_multi_act(inputs=X, filter_shape=[5, 5, 3, 64], varscope='conv_01')
    state = tf.nn.dropout(state, keep_prob=ops.settings.dropout_rate, name='dropout_01')
    state = ops.conv2d_multi_act(inputs=state, filter_shape=[3, 3, 64, 64], varscope='conv_02')
    # pooling 1
    state = tf.nn.max_pool(state, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_01')
    # conv2
    state = ops.conv2d_multi_act(inputs=state, filter_shape=[1, 1, 64, 64], varscope='conv_03')
    state = tf.nn.dropout(state, keep_prob=ops.settings.dropout_rate, name='dropout_02')
    state = ops.conv2d_multi_act(inputs=state, filter_shape=[5, 5, 64, 64], varscope='conv_04')
    # pooling 2
    state = tf.nn.max_pool(state, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_02')
    # dense layer 1
    state = ops.dense_multi_act(state, W_shape=[384], bias_shape=[-1], bias_init=0.1, varscope='dense_01')
    state = tf.nn.dropout(state, keep_prob=ops.settings.dropout_rate, name='dropout03')
    state = ops.dense_multi_act(state, W_shape=[192], bias_shape=[-1], bias_init=0.1, varscope='dense_02')
    # output layer
    logits = ops.dense_multi_act(state, W_shape=[10], bias_shape=[-1], bias_init=0.1, varscope='dense_03')

    return logits, l1_groups


