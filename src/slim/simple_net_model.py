from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


import slim.ops as ops


def build_ianntf_network(X):
    """

    """
    with tf.name_scope('ianntf_model'):

        # with scopes.arg_scope([ops.conv2d_multi_act], multi_activation=ops.multi_activation,
        #                       AF_set=AF_set, AF_weights_init=AF_weights_init):

        state = ops.conv_layer_mulit_act(layer_input=X, AF=tf.nn.relu, filter_shape=[5 , 5 , 3, 16], strides=[1 , 1 , 1 , 1], padding='SAME', bias_init=0.0, varscope='conv_01')

        state = tf.nn.max_pool(state, ksize=[1 ,2 ,2 ,1], strides=[1 ,2 ,2 ,1], padding='SAME')

        state = ops.conv_layer_mulit_act(layer_input=state, AF=tf.nn.relu, filter_shape=[3 , 3 , 16 , 32], strides=[1 , 1 , 1 , 1], padding='SAME', varscope='conv_02')

        state = tf.nn.max_pool(state, ksize=[1 ,3 ,3 ,1], strides=[1 ,2 ,2 ,1], padding='SAME')

        state = ops.dense_multi_act_layer(layer_input=state, AF=tf.nn.relu, W_shape=[512], varscope='dense_01')

        logits = ops.dense_multi_act_layer(layer_input=state, AF=tf.nn.tanh, W_shape=[10], varscope='dense_02')

        return logits