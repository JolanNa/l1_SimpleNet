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
import slim.san_ops as san_ops

from dataset import cifar10
from dataset.DataHandler import DataHandler

import numpy as np
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('total_iterations', 1,
                           """TODO"""
                           """TODO""")

tf.app.flags.DEFINE_string('data_dir', "/tmp/data_dir",
                           """The Directory where the Dataset is stored"""
                           """All subsets.""")

tf.app.flags.DEFINE_string('model_dir', "/tmp/model_dir",
                           """The Directory where the created files"""
                           """Includes checkpoints and events""")

tf.app.flags.DEFINE_integer('run_id', -1,
                           """The Id is used for the different runs"""
                           """Especially saving location.""")

tf.app.flags.DEFINE_integer('num_gpus', 1,
                           """number of gpus"""
                           """which are used""")

tf.app.flags.DEFINE_boolean('evaluate_network', False,
                           """Should the network be evaluated on the test set after training"""
                           """""")

tf.app.flags.DEFINE_boolean('self_activating', False,
                           """Should the network self activate"""
                           """The networks finds the best acitvations by L1_Regularization of the AF_set""")

tf.app.flags.DEFINE_boolean('self_activate_set_to_one', False,
                           """Set the hightest activation weight to one after self activating"""
                           """If not the current value of the weight is fixed""")

tf.app.flags.DEFINE_string('san_mode', "default",
                           """How should the network self activate"""
                           """Possbile Modes: AFTER_N, MEAN_THRESHOLD, AFTER_N_PRUNE, MEAN_THRESHOLD_PRUNE, AFTER_N_DELAYED, AFTER_N_DELAYED_PRUNE""")

tf.app.flags.DEFINE_float('san_gradient_mean_threshold', 0.01,
                           """The gradient_mean_threshold for unregularizing the Activation blocks"""
                           """   """)
tf.app.flags.DEFINE_float('san_after_n', -1,
                           """After how many steps should the network be unregularized"""
                           """   """)
tf.app.flags.DEFINE_float('san_after_n_delay', 1000,
                           """In the delayed mode """
                           """   """)

tf.app.flags.DEFINE_integer('max_steps', 1000,
                            """The Id is used for the different runs"""
                            """Especially saving location.""")

tf.app.flags.DEFINE_boolean('l1_regularization', False,
                            """Boolean whether L1 Regularization should be used"""
                            """...""")

tf.app.flags.DEFINE_boolean('p_norm_regularization', False,
                            """Boolean whether SAN Regularization should be used"""
                            """...""")
tf.app.flags.DEFINE_float('p_norm_degree', 2,
                            """TODO"""
                            """TODO""")

tf.app.flags.DEFINE_boolean('delayed_regularization', False,
                            """Boolean whether SAN Regularization should be used"""
                            """...""")
tf.app.flags.DEFINE_integer('delayed_regularization_after_n', 0,
                            """Boolean whether SAN Regularization should be used"""
                            """...""")

tf.app.flags.DEFINE_float('regularization_scale', 0.1,
                          """The scale term of the l1 Regularization"""
                          """...""")

tf.app.flags.DEFINE_float('activation_weight_stdv', 0.1,
                          """TODO"""
                          """...""")

tf.app.flags.DEFINE_boolean('decayed_regularization_scale', False,
                          """Should the penalty be normalized with respect to the xentropy (loss)"""
                          """...""")

tf.app.flags.DEFINE_boolean('normalized_regularization_penalty', False,
                          """Should the penalty be normalized with respect to the xentropy (loss)"""
                          """...""")
tf.app.flags.DEFINE_boolean('normalized_regularization_penalty_decay', False,
                          """Should the penalty be normalized with respect to the xentropy (loss)"""
                          """...""")


tf.app.flags.DEFINE_float('normalized_regularization_penalty_norm_scale', 0.5,
                          """To wich scale should the penalty be regularized"""
                          """...""")


tf.app.flags.DEFINE_boolean('l2_regularization', False,
                            """Boolean whether L2 Regularization should be used"""
                            """...""")

tf.app.flags.DEFINE_boolean('random_sample_activation_weights', True,
                            """Boolean whether L2 Regularization should be used"""
                            """...""")

tf.app.flags.DEFINE_string('optimizer', "Adam",
                           """TODO"""
                           """TODO""")

tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_boolean('explicit_learning_rate_decay', False,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_string('learning_rate_decay_mode', "linear",
                           """inc which mode should the learning rate be decayed?"""
                           """linear/exponential""")
tf.app.flags.DEFINE_float('learning_rate_decay_final_value', 0.0004,
                           """What is the final value the learning rate should reach"""
                           """""")

tf.app.flags.DEFINE_boolean('learning_rate_boost', False,
                          """Should the learning rate boost be applied""")
tf.app.flags.DEFINE_integer('learning_rate_boost_every_n_steps', 10000,
                          """every N steps the learnin rate boost is applied""")
tf.app.flags.DEFINE_integer('learning_rate_boost_until_global_step', 10000,
                          """Until which global step is the learning rate boost applied""")
tf.app.flags.DEFINE_float('learning_rate_boost_value', 0.0035,
                          """How much should be added to the learning rate""")

tf.app.flags.DEFINE_boolean('gradient_clipping', True,
                          """Sould the gradients be clipped a 1""")

tf.app.flags.DEFINE_float('momentum', 0.9,
                          """momentum value for the optimizer""")
tf.app.flags.DEFINE_float('num_steps_per_decay', 1000.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94,
                          """Learning rate decay factor.""")

tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Number of images to process in a batch.""")

tf.app.flags.DEFINE_integer('eval_batch_size', 64,
                            """Number of images to process in a batch.""")

tf.app.flags.DEFINE_integer('eval_interval', 1,
                            """Duration of Intervall in which the evaluation should be executed"""
                            """To be given in Secs""")
tf.app.flags.DEFINE_integer('save_checkpoints_steps', 1000,
                            """After how many steps should the model be saved"""
                            """Provide as Integer""")
tf.app.flags.DEFINE_integer('save_checkpoints_secs', None,
                            """After how many secs should the model be saved"""
                            """Provide as Integer""")


N_DIGITS = 10  # Number of digits.
X_FEATURE = 0  # Name of the input feature.

## The p_norm
# directly applies the regularization scale just as the l1 regularizer
def p_norm_regularization(weights, regularization_scale, degree=0.5):

    weights = tf.abs(weights)
    weights = tf.pow(weights, degree)

    sum = tf.reduce_sum(weights)
    sum = tf.pow(sum, 1/degree)
    return sum * regularization_scale




"""
Model of the classifier:
    
    SMCN Network
    Regularized Multi Activation Units
    Self Activating Options

"""
def simple_net_model(features, labels, mode):
    """Builds a simple_net network."""
    print("MODE -> " + str(mode))

    # An array that collects all the train ops to be grouped together in the end
    train_ops = []


    x = features[X_FEATURE]
    x = x / 128 - 1 #TODO BACK INSIDE??
    # x = x / 255
    labels = labels[0]

    training = (mode == tf.estimator.ModeKeys.TRAIN)


    # Compute logits (1 per class) and compute loss.
    logits, l1_groups = simple_net.build_simple_network(x, training)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class': predicted_classes,
            'prob': tf.nn.softmax(logits)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    print("[4]")

    # Objective
    penalty = tf.constant(0)
    l1_scale = tf.constant(0)

    ##TODO l1 and l1 should be managable in the tf.estiamtor.TrainSpec sub class (best would be even over tf.flags)
    l2_lambda = 0.01



    xentropy = tf.constant(0.0)

    # Empty arrays to collect values
    weight_sets = []
    penalties = []
    weight_set_in_use = tf.constant(-1.0)
    weights_in_use = []
    normalized_regularization_norm_scale = tf.constant(0.0)

    with tf.name_scope('objective'):
        # xentropy = tf.reduce_mean(
        #     tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        ## TODO: Check if we want to reduce mean!!
        xentropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)


        """
        #####################################################   
        #                                                   #
        #           REGULARIZATION OF THE RMAU              #
        #                                                   #
        #####################################################        
        """

        if (FLAGS.l1_regularization or FLAGS.p_norm_regularization) and FLAGS.l2_regularization:
            raise Exception("[FATAL] (L1 Regularization OR SAN-Regularization) AND L2 regularization is not possible")
        elif FLAGS.l1_regularization and FLAGS.p_norm_regularization:
            raise Exception("[FATAL] L1 Regularization AND SAN-Regularization is not possible")
        elif FLAGS.l1_regularization or FLAGS.p_norm_regularization:
            print("[INFO] Multi-Acitvations are L1-Regularized")

            def get_regularization_penalty(l1_groups, weight_sets, penalties, weights_in_use):
                for layer in l1_groups:
                    weight_sets.append(
                        [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if layer in v.name and '/act_weight' in v.name])

                if FLAGS.normalized_regularization_penalty:
                    regularization_scale = 1.0
                else:
                    if FLAGS.decayed_regularization_scale:
                        regularization_scale = tf.train.polynomial_decay(FLAGS.regularization_scale,
                                                                        tf.train.get_global_step() - FLAGS.delayed_regularization_after_n,
                                                                        (FLAGS.max_steps * 1.0) - FLAGS.delayed_regularization_after_n,
                                                                        0.0,
                                                                        power=1.0)
                    else:
                        regularization_scale = FLAGS.regularization_scale

                regularization_scale_fixed = FLAGS.regularization_scale
                l1_regularizer = tf.contrib.layers.l1_regularizer(scale=regularization_scale_fixed,
                                                                  scope="l1_regularization")


                for weights_idx, weights in enumerate(weight_sets):
                    namescope = weights[0].name[:-19]


                    _tmp_init = 1.0
                    if FLAGS.delayed_regularization:
                        _tmp_init = 0.0
                    elif FLAGS.self_activating and FLAGS.san_mode == "QUEUED_GROUPS":
                        print("self activating and queued gorups")
                        if weights_idx != 0:
                            _tmp_init = 0.0

                    weight_set_in_use = tf.get_variable(namescope + '/act_weight_in_use', initializer=_tmp_init,
                                                        trainable=False)

                    weights_in_use.append(weight_set_in_use)

                    if FLAGS.l1_regularization:
                        print("im if")
                        _tmp_penalty = p_norm_regularization(weights, regularization_scale=regularization_scale, degree=FLAGS.p_norm_degree)
                    elif FLAGS.p_norm_regularization:
                        print("im else if")
                        _tmp_penalty = p_norm_regularization(weights, regularization_scale=regularization_scale, degree=FLAGS.p_norm_degree)
                    else:
                        print("im else")
                        _tmp_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)

                    _p = tf.cond(tf.equal(weight_set_in_use, tf.constant(1.0)),
                                              lambda: _tmp_penalty,
                                              # lambda: tf.contrib.layers.apply_regularization(l1_regularizer, weights),
                                              lambda: tf.constant(0.0))

                    penalties.append(_p)

                penalty = tf.add_n(penalties)

                # If the regularization scale should be normalized with respect to the xentropy
                if FLAGS.normalized_regularization_penalty:
                    print("[INFO] Regularization penalty is normed to ..TODO.. of the loss")
                    if FLAGS.normalized_regularization_penalty_decay:
                        normalized_regularization_norm_scale = tf.train.polynomial_decay(FLAGS.normalized_regularization_penalty_norm_scale,
                                                                                    tf.train.get_global_step() - FLAGS.delayed_regularization_after_n,
                                                                                    (FLAGS.max_steps * 0.75) - FLAGS.delayed_regularization_after_n,
                                                                                    0.0,
                                                                                    power=1.0)
                    else:
                        normalized_regularization_norm_scale = tf.constant(FLAGS.normalized_regularization_penalty_norm_scale)

                    penalty_scale = (xentropy / penalty) * normalized_regularization_norm_scale

                else:
                    penalty_scale = tf.constant(1.0)
                    normalized_regularization_norm_scale = tf.constant(0.0)

                penalty = penalty * penalty_scale

                print(regularization_scale)
                return penalty, regularization_scale, regularization_scale_fixed, penalty_scale, normalized_regularization_norm_scale

            if FLAGS.delayed_regularization:

                global_step = tf.train.get_global_step()
                global_step = tf.cast(global_step, tf.int32)
                _penalty = get_regularization_penalty(l1_groups, weight_sets, penalties, weights_in_use)
                penalty, regularization_scale, regularization_scale_fixed, penalty_scale, normalized_regularization_norm_scale = tf.cond(tf.less(FLAGS.delayed_regularization_after_n, global_step),
                                                    # lambda: get_regularization_penalty(l1_groups, weight_sets, penalties, weights_in_use),
                                                      lambda: _penalty,
                                                      lambda: (tf.constant(0.0),tf.constant(0.0),tf.constant(0.0),tf.constant(0.0),tf.constant(0.0)),
                                                      )
            else:
                penalty, regularization_scale, regularization_scale_fixed, penalty_scale, normalized_regularization_norm_scale = get_regularization_penalty(l1_groups, weight_sets, penalties, weights_in_use)

            loss = xentropy + penalty

            tf.summary.scalar('regularization_scale', regularization_scale)
            tf.summary.scalar('regularization_scale_fixed', regularization_scale_fixed)
            tf.summary.scalar('penalty_scale', penalty_scale)
            tf.summary.scalar('normalized_regularization_norm_scale', normalized_regularization_norm_scale)

            tf.summary.scalar('penalty', penalty)

        elif FLAGS.l2_regularization:
            print("[INFO] Weights are L2-Regularized")

            l2 = [tf.nn.l2_loss(v) for f in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if
                       'weight' and not 'act_weight' in v.name]
            weights_norm = tf.reduce_sum(input_tensor=l2_lambda * tf.stack(l2),
                                              name='weights_norm')
            loss = xentropy + weights_norm

        else:
            print("[INFO] No regularization is applied")
            loss = xentropy


        tf.summary.scalar('loss', loss)
        tf.summary.scalar('xentropy', xentropy)

    print("[6]")

    if FLAGS.delayed_regularization:

        def switch_weights_into_use(weights_in_use):
            weights_in_use_new = []
            for i in range(len(weights_in_use)):
                new_value = tf.constant(1.0)
                weights_in_use_new.append(tf.assign(weights_in_use[i], new_value))

            return tf.equal(weights_in_use_new, tf.zeros(tf.shape(weights_in_use)))

        def keep_weights_out_of_use(weights_in_use):
            return tf.equal(tf.zeros(tf.shape(weights_in_use)), tf.zeros(tf.shape(weights_in_use)))
            ops = []
            for i in range(len(weights_in_use)):
                _op = tf.assign(weights_in_use[i], tf.constant(0.0))
                ops.append(_op)

            main_op = tf.equal(tf.constant(0.0), tf.constant(0.0))

            for op in ops:
                main_op = tf.group(main_op, op)
            return main_op


        global_step = tf.train.get_global_step()
        global_step = tf.cast(global_step, tf.int32)


        delayed_regularization_op = tf.cond(tf.equal(FLAGS.delayed_regularization_after_n, global_step),
                                            lambda: switch_weights_into_use(weights_in_use),
                                            lambda: keep_weights_out_of_use(weights_in_use))

        train_ops.append(delayed_regularization_op)


    accuracy_train = tf.metrics.accuracy(
        labels=labels, predictions=predicted_classes)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predicted_classes)
    }

    # Create training op.
    if training:



        if FLAGS.explicit_learning_rate_decay:
            if FLAGS.learning_rate_decay_mode == "linear":
                learning_rate = tf.train.polynomial_decay(FLAGS.initial_learning_rate,
                                                          tf.train.get_global_step(),
                                                          FLAGS.max_steps,
                                                          FLAGS.learning_rate_decay_final_value,
                                                          power=1.0)
            else:
                learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                           tf.train.get_global_step(),
                                                           FLAGS.num_steps_per_decay,
                                                           FLAGS.learning_rate_decay_factor,
                                                           staircase=True)
        else:
            learning_rate = FLAGS.initial_learning_rate

        if FLAGS.learning_rate_boost:
            learning_rate_boost = tf.get_variable('learning_rate_boost', initializer=0.0, trainable=False)
            increased_learning_rate_boost = learning_rate_boost + FLAGS.learning_rate_boost_value

            global_step = tf.train.get_global_step()
            global_step = tf.cast(global_step, tf.int32)
            learning_rate_boost_new = tf.cond(tf.less(global_step, FLAGS.learning_rate_boost_until_global_step),
                                              lambda: tf.cond(tf.greater(global_step, tf.constant(0)),
                                                              lambda: tf.cond(tf.equal(tf.mod(tf.train.get_global_step(), FLAGS.learning_rate_boost_every_n_steps), 0),
                                                                            lambda: increased_learning_rate_boost,
                                                                            lambda: learning_rate_boost),
                                                              lambda: learning_rate_boost),
                                              lambda: learning_rate_boost)


            _tmp_op = tf.assign(learning_rate_boost, learning_rate_boost_new)

            train_ops.append(_tmp_op)

            learning_rate = learning_rate + learning_rate_boost

        if FLAGS.optimizer == "Adam":
            print("[INFO] Using Adam Optimizer (lr: " + str(FLAGS.initial_learning_rate) +")")
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            learning_rate = optimizer._lr
        elif FLAGS.optimizer == "Adagrad":
            print("[INFO] Using Adagrad Optimizer (lr: " + str(FLAGS.initial_learning_rate) +")")
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
            learning_rate = optimizer._learning_rate

        elif FLAGS.optimizer == "Momentum":
            print("[INFO] Using Momentum Optimizer (lr: " + str(FLAGS.initial_learning_rate) +", momentum: "+str(FLAGS.momentum)+")")
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=FLAGS.momentum, use_nesterov=False)
        elif FLAGS.optimizer == "Nesterov":
            print("[INFO] Using Momentum Optimizer with Nesterov (lr: " + str(FLAGS.initial_learning_rate) +", momentum: "+str(FLAGS.momentum)+")")
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=FLAGS.momentum, use_nesterov=True)
            learning_rate = optimizer._learning_rate
        else:
            print("[INFO] Using Standard Gradient Descent Optimizer (lr: " + str(FLAGS.initial_learning_rate) +")")
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.initial_learning_rate)
            learning_rate = optimizer._learning_rate


        tf.summary.scalar('learning_rate', learning_rate)

        trainable_variables = [var for var in tf.trainable_variables() if "/act_weight" not in var.name]

        gradients = tf.gradients(loss, trainable_variables)
        gradients_with_vars_raw = np.stack((gradients, trainable_variables), axis=-1)



        argmax = None
        weight_set = None
        unregularize_until_layer = tf.constant(0.0)

        active_weight_set_in_queue = tf.get_variable("active_weight_set_in_queue", initializer=0,
                                              trainable=False)

        """
        #####################################################   
        #                                                   #
        #          SELF ACTIVATING OF THE NETWORK           #
        #                                                   #
        #####################################################        
        
        3 Different Modes:
        
            1) AFTER_N
            
            2) AFTER_N_DELAYED
            
            3) QUEUED_GROUPS
        
        """


        if FLAGS.self_activating:

            #sort gradients into the different activation groups/regularization groups
            collected_gradients = []
            collected_vars = []
            for idx_weight_set, weight_set in enumerate(weight_sets):
                collected_gradients_with_vars = []

                # Is the weights set of the layer in use?
                weight_set_in_use = weights_in_use[idx_weight_set]


                if FLAGS.san_mode == "AFTER_N":
                    print("[INFO] SAN-MODE => AFTER_N")
                    #Get the names
                    weight_set_names = [w.name for w in weight_set]
                    #Get the variables from the tf.trainable_variables() set to apply gradients later
                    weight_set = [var for var in tf.trainable_variables() if var.name in weight_set_names]

                    # Intialize the argmax for the activation weights. This weight is picked as the favorite one
                    unregularize_argmax = tf.get_variable("unregularize_argmax_" + str(idx_weight_set), initializer=-1.0,
                                                          trainable=False)

                    #compute the argmax of the absolute acivation weights.
                    absolute_weights = tf.abs(weight_set)
                    argmax = tf.argmax(absolute_weights)  # TODO, DAS HIER MUSS VON DEN WEIGHTS SEIN
                    argmax = tf.cast(argmax, tf.float32)

                    # cast values for tf.cond
                    after_n = FLAGS.san_after_n
                    after_n = tf.cast(after_n, tf.int32)
                    global_step = tf.train.get_global_step()
                    global_step = tf.cast(global_step, tf.int32)

                    # If global_step > after N : set the variable unregularize_argmaz to argmax. If no set to -1
                    # If unregularize_argmaz is -1 the weights are not unregularized
                    new_value = tf.cond(tf.less(after_n, global_step), lambda: argmax, lambda: tf.constant(-1.0))
                    train_op_1 = tf.assign(unregularize_argmax, new_value)

                    train_ops.append(train_op_1)

                    # function that unregularized the weights:
                    # -> a weight is unregularized if the weight set in use vairable is set to 0
                    def unregularize_weights():

                        for i in range(len(weight_set)):
                            if FLAGS.self_activate_set_to_one:
                                new_value = tf.cond(tf.equal(unregularize_argmax, i), lambda: tf.constant(1.0),
                                                    lambda: tf.constant(0.0))
                            else:
                                new_value = tf.cond(tf.equal(unregularize_argmax, i), lambda: weight_set[i],
                                                    lambda: tf.constant(0.0))
                            _op = tf.assign(weight_set[i], new_value)
                            train_ops.append(_op)

                        _assign_op = tf.assign(weight_set_in_use, tf.constant(0.0))
                        train_ops.append(_assign_op)
                        return [tf.constant(0.0) for w in weight_set]

                    # the gradients are computed
                    # and all set to 0 if the weights are unregularized
                    # unfortunatly gradients have to be computed. If they are only computed conditionally,
                    # they are different from when they are always computed
                    _tmp_gradients = tf.gradients(loss, weight_set)
                    _gradients = tf.cond(tf.equal(unregularize_argmax, tf.constant(-1.0)),
                                         lambda: _tmp_gradients,
                                         lambda: unregularize_weights())

                    ## Collect all the variables and gradients to apply them later
                    if len(collected_vars) == 0:
                        collected_vars = weight_set
                        collected_gradients = _gradients
                    else:
                        collected_gradients = np.concatenate([collected_gradients, _gradients])
                        collected_vars = np.concatenate([collected_vars, weight_set])

                elif FLAGS.san_mode == "AFTER_N_PRUNE":
                    unregularize_argmin = tf.get_variable("unregularize_argmin_" + str(idx_weight_set), initializer=-1.0,
                                                          trainable=False)
                    _train_op = tf.cond(tf.equal(unregularize_argmin, tf.constant(-1.0)),
                                        lambda: san_ops.san_regularize_after_n_prune(weight_set,
                                                                        loss,
                                                                        optimizer,
                                                                        unregularize_argmax,
                                                                        idx_weight_set),
                                        lambda: san_ops.san_unregularize_after_n_prune(weight_set,
                                                                              unregularize_argmax,
                                                                                idx_weight_set))
                elif FLAGS.san_mode == "AFTER_N_DELAYED":
                    print("[INFO] SAN-MODE => AFTER_N_DELAYED")
                    weight_set_names = [w.name for w in weight_set]
                    weight_set = [var for var in tf.trainable_variables() if var.name in weight_set_names]
                    unregularize_argmax = tf.get_variable("unregularize_argmax_" + str(idx_weight_set), initializer=-1.0,
                                                          trainable=False)
                    # Unregularize_until_layer,
                    # is initialized with 2.0 as we want to use tf.less
                    unregularize_until_layer = tf.get_variable("unregularize_until_layer" + str(idx_weight_set), initializer=1,
                                                          trainable=False)
                    # check if set should be unregularized
                    absolute_weights = tf.abs(weight_set)
                    argmax = tf.argmax(absolute_weights)  # TODO, DAS HIER MUSS VON DEN WEIGHTS SEIN
                    argmax = tf.cast(argmax, tf.float32)

                    after_n = FLAGS.san_after_n
                    after_n = tf.cast(after_n, tf.int32)
                    after_n_delay = FLAGS.san_after_n_delay
                    after_n_delay = tf.cast(after_n_delay, tf.int32)
                    global_step = tf.train.get_global_step()
                    global_step = tf.cast(global_step, tf.int32)

                    new_value = tf.cond(tf.less(after_n, global_step), lambda: argmax, lambda: tf.constant(-1.0))
                    train_op_1 = tf.assign(unregularize_argmax, new_value)
                    train_ops.append(train_op_1)

                    new_value_delay = tf.cond(tf.less((after_n + unregularize_until_layer * after_n_delay), global_step), lambda: unregularize_until_layer + 1, lambda: unregularize_until_layer)
                    train_op_2 = tf.assign(unregularize_until_layer, new_value_delay)
                    train_ops.append(train_op_2)


                    def unregularize_weights():

                        for i in range(len(weight_set)):
                            if FLAGS.self_activate_set_to_one:
                                new_value = tf.cond(tf.equal(unregularize_argmax, i), lambda: tf.constant(1.0),
                                                    lambda: tf.constant(0.0))
                            else:
                                new_value = tf.cond(tf.equal(unregularize_argmax, i), lambda: weight_set[i],
                                                    lambda: tf.constant(0.0))
                            _op = tf.assign(weight_set[i], new_value)
                            train_ops.append(_op)

                        _assign_op = tf.assign(weight_set_in_use, tf.constant(0.0))
                        train_ops.append(_assign_op)
                        return [tf.constant(0.0) for w in weight_set]


                    _tmp_gradients = tf.gradients(loss, weight_set)
                    _gradients = tf.cond(tf.equal(unregularize_argmax, tf.constant(-1.0)),
                                         lambda: _tmp_gradients,
                                         lambda: tf.cond(tf.less(idx_weight_set, unregularize_until_layer),
                                                         lambda: unregularize_weights(),
                                                         lambda: _tmp_gradients))


                    if len(collected_vars) == 0:
                        collected_vars = weight_set
                        collected_gradients = _gradients
                    else:
                        collected_gradients = np.concatenate([collected_gradients, _gradients])
                        collected_vars = np.concatenate([collected_vars, weight_set])


                elif FLAGS.san_mode == "QUEUED_GROUPS":
                    print("[INFO] SAN-MODE => QUEUED_GROUPS")
                    weight_set_names = [w.name for w in weight_set]
                    weight_set = [var for var in tf.trainable_variables() if var.name in weight_set_names]
                    unregularize_argmax = tf.get_variable("unregularize_argmax_" + str(idx_weight_set), initializer=-1.0,
                                                          trainable=False)
                    # Unregularize_until_layer,
                    unregularize_until_layer = tf.get_variable("unregularize_until_layer" + str(idx_weight_set), initializer=0,
                                                               trainable=False)
                    # check if set should be unregularized
                    absolute_weights = tf.abs(weight_set)
                    argmax = tf.argmax(absolute_weights)  # TODO, DAS HIER MUSS VON DEN WEIGHTS SEIN
                    argmax = tf.cast(argmax, tf.float32)

                    after_n = FLAGS.san_after_n
                    after_n = tf.cast(after_n, tf.int32)
                    after_n_delay = FLAGS.san_after_n_delay
                    after_n_delay = tf.cast(after_n_delay, tf.int32)
                    global_step = tf.train.get_global_step()
                    global_step = tf.cast(global_step, tf.int32)

                    new_value = tf.cond(tf.less(after_n, global_step), lambda: argmax, lambda: tf.constant(-1.0))
                    train_op_1 = tf.assign(unregularize_argmax, new_value)
                    train_ops.append(train_op_1)


                    if FLAGS.delayed_regularization:
                        _minimum_global_step = tf.constant(FLAGS.delayed_regularization_after_n)
                    else:
                        _minimum_global_step = tf.constant(0)

                    _tmp = unregularize_until_layer + 1
                    new_value_delay = tf.cond(tf.less((_minimum_global_step + after_n + unregularize_until_layer * after_n), global_step),
                                              lambda: _tmp,
                                              lambda: unregularize_until_layer)
                    train_op_2 = tf.assign(unregularize_until_layer, new_value_delay)
                    train_ops.append(train_op_2)

                    def unregularize_weights():

                        for i in range(len(weight_set)):
                            if FLAGS.self_activate_set_to_one:
                                new_value = tf.cond(tf.equal(unregularize_argmax, i), lambda: tf.constant(1.0),
                                                    lambda: tf.constant(0.0))
                            else:
                                new_value = tf.cond(tf.equal(unregularize_argmax, i), lambda: weight_set[i],
                                                    lambda: tf.constant(0.0))
                            _op = tf.assign(weight_set[i], new_value)
                            train_ops.append(_op)



                        return [tf.constant(0.0) for w in weight_set]



                    new_weight_set_in_use = tf.cond(tf.greater(global_step, _minimum_global_step),
                                                    lambda: tf.cond(tf.equal(idx_weight_set, active_weight_set_in_queue),
                                                            lambda: tf.constant(1.0),
                                                            lambda: tf.constant(0.0)),
                                                    lambda: tf.constant(0.0))


                    _tmp_op = tf.assign(weight_set_in_use, new_weight_set_in_use)
                    train_ops.append(_tmp_op)


                    new_active_weight_set_in_queue = active_weight_set_in_queue + tf.constant(1)
                    new_active_weight_set_in_queue = tf.cond(tf.greater(global_step, _minimum_global_step),
                                                             lambda: tf.cond(tf.equal(tf.mod(global_step + _minimum_global_step, after_n, name="equal_new_active_weight_set_in_queue"), tf.constant(0)),
                                                                             lambda: new_active_weight_set_in_queue,
                                                                             lambda: active_weight_set_in_queue),
                                                             lambda: active_weight_set_in_queue)
                    _tmp_op = tf.assign(active_weight_set_in_queue, new_active_weight_set_in_queue)
                    train_ops.append(_tmp_op)


                    _tmp_gradients = tf.gradients(loss, weight_set)

                    tmp_idx_weight_set = tf.cast(idx_weight_set, tf.int32)
                    _gradients = tf.cond(tf.less(idx_weight_set, new_active_weight_set_in_queue),
                                         lambda: unregularize_weights(),
                                         lambda: tf.cond(tf.equal(tmp_idx_weight_set, new_active_weight_set_in_queue),
                                                         lambda: tf.cond(tf.equal(weight_set_in_use, tf.constant(1.0)),
                                                                         lambda: _tmp_gradients,
                                                                         lambda: [tf.constant(0.0) for w in weight_set]),
                                                         lambda: [tf.constant(0.0) for w in weight_set]))


                    if len(collected_vars) == 0:
                        collected_vars = weight_set
                        collected_gradients = _gradients
                    else:
                        collected_gradients = np.concatenate([collected_gradients, _gradients])
                        collected_vars = np.concatenate([collected_vars, weight_set])

                ###### DEPRECATED ######
                elif FLAGS.san_mode == "MEAN_THRESHOLD":
                    unregularize_argmax = tf.get_variable("unregularize_argmax_" + str(idx_weight_set), initializer=-1.0,
                                                          trainable=False)

                    _train_op = tf.cond(tf.equal(unregularize_argmax, tf.constant(-1.0)),
                                    lambda: san_ops.keep_activation_group_regularized(weight_set,
                                                                              loss,
                                                                              optimizer,
                                                                              unregularize_argmax,
                                                                              idx_weight_set),
                                    lambda: san_ops.unregularize_activation_group(weight_set,
                                                                          unregularize_argmax,
                                                                          idx_weight_set))
                else:
                    print('SAN is activated! However, no valid SAN_MODE is provided!')
                    weight_set_names = [w.name for w in weight_set]
                    weight_set = [var for var in tf.trainable_variables() if var.name in weight_set_names]
                    gradients = tf.gradients(loss, weight_set)
                    gradients_with_vars = np.stack((gradients, weight_set), axis=-1)
                    if FLAGS.gradient_clipping:
                        gradients_with_vars = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients_with_vars]


            if FLAGS.gradient_clipping:
                collected_gradients_with_vars = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in collected_gradients_with_vars]

            _train_op = optimizer.apply_gradients(collected_gradients_with_vars, global_step=tf.train.get_global_step())

            train_ops.append(_train_op)
        elif FLAGS.l1_regularization:
            collected_gradients = []
            collected_vars = []
            for idx_weight_set, weight_set in enumerate(weight_sets):
                weight_set_in_use = weights_in_use[idx_weight_set]

                _tmp_gradients = tf.gradients(loss, weight_set)

                tmp_idx_weight_set = tf.cast(idx_weight_set, tf.float32)
                _gradients = tf.cond(tf.equal(tf.constant(1.0), weight_set_in_use),
                                     lambda: _tmp_gradients,
                                     lambda: [tf.constant(0.0) for w in weight_set])

                if len(collected_vars) == 0:
                    collected_vars = weight_set
                    collected_gradients = _gradients
                else:
                    collected_gradients = np.concatenate([collected_gradients, _gradients])
                    collected_vars = np.concatenate([collected_vars, weight_set])



            collected_gradients_with_vars = np.stack((collected_gradients, collected_vars), axis=-1)

            gradients_with_vars_activation = collected_gradients_with_vars
            if FLAGS.gradient_clipping:
                gradients_with_vars_activation = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in collected_gradients_with_vars]

            train_op_activation = optimizer.apply_gradients(gradients_with_vars_activation, global_step=tf.train.get_global_step())
            train_ops.append(train_op_activation)

        # Group the train_ops together:
        # train_op_raw -> train_op for all trainable variables expect the ones of multi activated acitvations
        # train_ops_activation -> the train_ops of the multi acitvated activations

        if FLAGS.gradient_clipping:
            gradients_with_vars_raw = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in
                                              gradients_with_vars_raw]

        train_op_raw = optimizer.apply_gradients(gradients_with_vars_raw, global_step=tf.train.get_global_step())

        tf.summary.scalar('accuracy_train', accuracy_train[1])

        #Group the train ops together
        train_op = tf.group(train_op_raw)
        for op in train_ops:
            train_op = tf.group(train_op, op)

        logging_hook = tf.train.LoggingTensorHook({
                                                    "loss": loss,
                                                   "step": tf.train.get_global_step(),
                                                   "acc": accuracy_train[1],
                                                   "l1_pen": penalty,
                                                    "active_ws_in_queue": active_weight_set_in_queue,
                                                   },
                                                  every_n_iter=100)

        return tf.estimator.EstimatorSpec(mode,
                                          loss=loss,
                                          train_op=train_op,
                                          training_hooks=[logging_hook],
                                          )

    # Compute evaluation metrics.
    return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=eval_metric_ops)

#TAKEN FROM: https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10_estimator/cifar10_main.py
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

    print("[INFO] +--------------------------------------+")
    print("[INFO] |              SETTTINGS               |")
    print("[INFO] +--------------------------------------+")
    for key in FLAGS.__flags.keys():
        print('  {}: {}'.format(key, getattr(FLAGS, key)))
    print("[INFO] +--------------------------------------+")

    for iteration in range(FLAGS.total_iterations):

        config = tf.estimator.RunConfig(save_summary_steps=100,
                                        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
                                        save_checkpoints_secs=FLAGS.save_checkpoints_secs,
                                        )
        # Create a new classifier.
        estimator = tf.estimator.Estimator(model_fn=simple_net_model, model_dir=FLAGS.model_dir + str(FLAGS.run_id) + "/" + str(iteration) + "/", config=config)

        tf.logging.set_verbosity(tf.logging.INFO)  # Show training logs.

        EVAL_INTERVAL = 600

        num_gpus = 1
        use_distortion_for_training = True

        train_input_fn = functools.partial(
            input_fn,
            FLAGS.data_dir,
            subset='train',
            num_shards=num_gpus,
            batch_size=FLAGS.batch_size,
            use_distortion_for_training=use_distortion_for_training)


        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                            max_steps=FLAGS.max_steps)

        validation_input_fn = functools.partial(
            input_fn,
            FLAGS.data_dir,
            subset='validation',
            batch_size=FLAGS.eval_batch_size,
            num_shards=num_gpus)

        eval_spec = tf.estimator.EvalSpec(input_fn= validation_input_fn,
                                          start_delay_secs = 60,
                                          throttle_secs = FLAGS.eval_interval)

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

        if FLAGS.evaluate_network:
            print("--START EVALUATION OF NETWORK--")

            train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                                max_steps=FLAGS.max_steps + 10)
            validation_input_fn = functools.partial(
                input_fn,
                FLAGS.data_dir,
                subset='eval',
                batch_size=FLAGS.eval_batch_size,
                num_shards=num_gpus)
            eval_spec = tf.estimator.EvalSpec(input_fn=validation_input_fn,
                                              start_delay_secs=1,
                                              throttle_secs=FLAGS.eval_interval)
            tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)



if __name__ == '__main__':
    tf.app.run()