#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""This example builds deep residual network for mnist data.
Reference Paper: http://arxiv.org/pdf/1512.03385.pdf
Note that this is still a work-in-progress. Feel free to submit a PR
to make this better.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from math import sqrt

import functools

from slim import slim
import slim.simple_net_model as simple_net

from dataset import cifar10

import numpy as np
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', "/tmp/data_dir",
                           """The Directory where the Dataset is stored"""
                           """All subsets.""")

tf.app.flags.DEFINE_string('model_dir', "/tmp/model_dir",
                           """The Directory where the created files"""
                           """Includes checkpoints and events""")

tf.app.flags.DEFINE_integer('run_id', -1,
                           """The Id is used for the different runs"""
                           """Especially saving location.""")

N_DIGITS = 10  # Number of digits.
X_FEATURE = 0  # Name of the input feature.



def simple_net_model(features, labels, mode):
    """Builds a simple_net network."""


    x = features[X_FEATURE]
    labels = labels[0]
    input_shape = x.get_shape().as_list()

    # Reshape the input into the right shape if it's 2D tensor
    if len(input_shape) == 2:
        ndim = int(sqrt(input_shape[1]))
        x = tf.reshape(x, [-1, ndim, ndim, 1])

    training = (mode == tf.estimator.ModeKeys.TRAIN)


    # Compute logits (1 per class) and compute loss.
    logits = simple_net.build_ianntf_network(x)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class': predicted_classes,
            'prob': tf.nn.softmax(logits)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Objective
    penalty = tf.constant(0)
    ##TODO l1 and l1 should be managable in the tf.estiamtor.TrainSpec sub class (best would be even over tf.flags)
    l2_regularize = False
    l1_regularize = False
    l2_lambda = 0.01
    l1_regularizer_scale = 0.8

    with tf.name_scope('objective'):
        xentropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        ## TODO: Check if we want to reduce mean!!
        # xentropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        if l1_regularize and l2_regularize:
            raise Exception("[FATAL] L1 Regularization AND L2 regularization is not possible")
        elif l1_regularize:
            weight_sets = []
            for layer in layer_names:
                weight_sets.append(
                    [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if layer + '/act_weight' in v.name])

            l1_regularizer = tf.contrib.layers.l1_regularizer(scale=l1_regularizer_scale,
                                                              scope="l1_regularization")
            penalties = [tf.contrib.layers.apply_regularization(l1_regularizer, weights) for weights in weight_sets]
            penalty = tf.add_n(penalties)
            loss = xentropy + penalty

            tf.summary.scalar('penalty', penalty)
        elif l2_regularize:
            l2 = [tf.nn.l2_loss(v) for f in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if
                       'weight' and not 'act_weight' in v.name]
            weights_norm = tf.reduce_sum(input_tensor=l2_lambda * tf.stack(l2),
                                              name='weights_norm')
            loss = xentropy + weights_norm

        else:
            loss = xentropy

        # tf.summary.scalar('loss', loss)
        # tf.summary.scalar('xentropy', xentropy)


    # Create training op.
    if training:
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    # Compute evaluation metrics.
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predicted_classes)
    }
    return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=eval_metric_ops)

##TAKEN FROM: https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10_estimator/cifar10_main.py
def input_fn(data_dir,
             subset,
             num_shards,
             batch_size,
             use_distortion_for_training=False):
  """Create input graph for model.
  Args:
    data_dir: Directory where TFRecords representing the dataset are located.
    subset: one of 'train', 'validate' and 'eval'.
    num_shards: num of towers participating in data-parallel training.
    batch_size: total batch size for training to be divided by the number of
    shards.
    use_distortion_for_training: True to use distortions.
  Returns:
    two lists of tensors for features and labels, each of num_shards length.
  """
  with tf.device('/cpu:0'):
    use_distortion = subset == 'train' and use_distortion_for_training
    dataset = cifar10.Cifar10DataSet(data_dir, subset, use_distortion)
    image_batch, label_batch = dataset.make_batch(batch_size)
    if num_shards <= 1:
      # No GPU available or only 1 GPU.
      return [image_batch], [label_batch]

    # Note that passing num=batch_size is safe here, even though
    # dataset.batch(batch_size) can, in some cases, return fewer than batch_size
    # examples. This is because it does so only when repeating for a limited
    # number of epochs, but our dataset repeats forever.
    image_batch = tf.unstack(image_batch, num=batch_size, axis=0)
    label_batch = tf.unstack(label_batch, num=batch_size, axis=0)
    feature_shards = [[] for i in range(num_shards)]
    label_shards = [[] for i in range(num_shards)]
    for i in xrange(batch_size):
      idx = i % num_shards
      feature_shards[idx].append(image_batch[i])
      label_shards[idx].append(label_batch[i])
    feature_shards = [tf.parallel_stack(x) for x in feature_shards]
    label_shards = [tf.parallel_stack(x) for x in label_shards]
    return feature_shards, label_shards

def main(unused_args):

    if FLAGS.run_id == -1:
        print("[USAGE ERROR] Please provide a run_id! e.g.: train.py --run_id=12")
        sys.exit()

    # Create a new resnet classifier.
    estimator = tf.estimator.Estimator(model_fn=simple_net_model, model_dir=FLAGS.model_dir + str(FLAGS.run_id) + "/")

    tf.logging.set_verbosity(tf.logging.INFO)  # Show training logs.

    EVAL_INTERVAL = 300

    # train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={X_FEATURE: mnist.train.images},
    #     y=mnist.train.labels.astype(np.int32),
    #     batch_size=32,
    #     num_epochs=None,
    #     shuffle=True)
    # train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
    #                                     max_steps=1000)
    #
    # validation_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={X_FEATURE: mnist.test.images},
    #     y=mnist.test.labels.astype(np.int32),
    #     num_epochs=1,
    #     shuffle=False)
    # eval_spec = tf.estimator.EvalSpec(input_fn= validation_input_fn,
    #                                   start_delay_secs = 60,
    #                                   throttle_secs = EVAL_INTERVAL)
    #
    # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


    num_gpus = 0
    use_distortion_for_training = False
    train_batch_size = 64

    train_input_fn = functools.partial(
        input_fn,
        FLAGS.data_dir,
        subset='train',
        num_shards=num_gpus,
        batch_size=train_batch_size,
        use_distortion_for_training=use_distortion_for_training)


    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=1000)

    eval_batch_size = 5000
    validation_input_fn = functools.partial(
        input_fn,
        FLAGS.data_dir,
        subset='validation',
        batch_size=eval_batch_size,
        num_shards=num_gpus)

    eval_spec = tf.estimator.EvalSpec(input_fn= validation_input_fn,
                                      start_delay_secs = 60,
                                      throttle_secs = EVAL_INTERVAL)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    tf.app.run()