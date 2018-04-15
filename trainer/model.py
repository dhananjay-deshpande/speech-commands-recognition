# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Audio Commands classification model.
"""

import argparse
import logging
import os
import math

import tensorflow as tf
from tensorflow.contrib import layers

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils as saved_model_utils

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

import util
from util import override_if_not_in_args

LOGITS_TENSOR_NAME = 'logits_tensor'

AUDIO_URI_COLUMN = 'audio_uri'
LABEL_COLUMN = 'label'
FINGERPRINT_COLUMN = 'fingerprint'
DEFAULT_CHECKPOINT = ''

class GraphMod():
    TRAIN = 1
    EVALUATE = 2
    PREDICT = 3

def build_signature(inputs, outputs):
    """Build the signature.

    Not using predic_signature_def in saved_model because it is replacing the
    tensor name, b/35900497.

    Args:
      inputs: a dictionary of tensor name to tensor
      outputs: a dictionary of tensor name to tensor
    Returns:
      The signature, a SignatureDef proto.
    """
    signature_inputs = {key: saved_model_utils.build_tensor_info(tensor)
                        for key, tensor in inputs.items()}
    signature_outputs = {key: saved_model_utils.build_tensor_info(tensor)
                         for key, tensor in outputs.items()}

    signature_def = signature_def_utils.build_signature_def(
        signature_inputs, signature_outputs,
        signature_constants.PREDICT_METHOD_NAME)

    return signature_def

def create_model():
    """Factory method that creates model to be used by generic task.py."""
    parser = argparse.ArgumentParser()
    # Label count needs to correspond to nubmer of labels in dictionary used
    # during preprocessing.
    parser.add_argument(
        '--label_count',
        type=int,
        default=5)
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.5)
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='cnn',
        help='What model architecture to use')
    parser.add_argument(
        '--train_dir',
        type=str,
        default='/tmp/speech_commands_train',
        help='Directory to write event logs and checkpoint.')
    parser.add_argument(
        '--save_step_interval',
        type=int,
        default=100,
        help='Save model checkpoint every save_steps.')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs')
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the wavs')
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=30.0,
        help='How long each spectrogram timeslice is.')
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=10.0,
        help='How far to move in time between spectogram timeslices.')
    parser.add_argument(
        '--dct_coefficient_count',
        type=int,
        default=40,
        help='How many bins to use for the MFCC fingerprint',)

    args, task_args = parser.parse_known_args()

    override_if_not_in_args('--max_steps', '1000', task_args)
    override_if_not_in_args('--batch_size', '100', task_args)
    override_if_not_in_args('--eval_set_size', '370', task_args)
    override_if_not_in_args('--eval_interval_secs', '2', task_args)
    override_if_not_in_args('--log_interval_secs', '2', task_args)
    override_if_not_in_args('--min_train_eval_rate', '2', task_args)

    if args.model_architecture == 'cnn':
        return Model(args), task_args

    # By default return cnn model
    return Model(args), task_args

class GraphReferences(object):
    """Holder of base tensors used for training model using common task."""

    def __init__(self):
        self.examples = None
        self.train = None
        self.global_step = None
        self.metric_updates = []
        self.metric_values = []
        self.keys = None
        self.predictions = []
        self.input_wav = None

class Model(object):
    """CNN TensorFlow model for the audio commands problem."""

    def __init__(self, args):
        self.args = args

        self.dropout = args.dropout
        self.desired_samples = int(args.sample_rate * args.clip_duration_ms / 1000)
        self.window_size_samples = int(args.sample_rate * args.window_size_ms / 1000)
        self.window_stride_samples = int(args.sample_rate * args.window_stride_ms / 1000)
        self.checkpoint_file = os.path.join(args.train_dir, args.model_architecture + '.ckpt')
        self.length_minus_window = (self.desired_samples - self.window_size_samples)
        if self.length_minus_window < 0:
            self.spectrogram_length = 0
        else:
            self.spectrogram_length = 1 + int(self.length_minus_window / self.window_stride_samples)
        self.fingerprint_size = args.dct_coefficient_count * self.spectrogram_length


    def add_final_training_ops(self,
                               fingerprints,
                               all_labels_count,
                               is_training,
                               dropout_keep_prob=None):
        """Adds a new softmax and fully-connected layer for training.

         The set up for the softmax and fully-connected layers is based on:
         https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

         This function can be customized to add arbitrary layers for
         application-specific requirements.
        Args:
          fingerprints: The fingerprints tensor.
          all_labels_count: The number of all labels including the default label.
          hidden_layer_size: The size of the hidden_layer. Roughtly, 1/4 of the
                             bottleneck tensor size.
          dropout_keep_prob: the percentage of activation values that are retained.
        Returns:
          softmax: The softmax or tensor. It stores the final scores.
          logits: The logits tensor.
        """
        """Builds a convolutional model with low compute requirements.
        
        This is roughly the network labeled as 'cnn-one-fstride4' in the
        'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
        http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

        Here's the layout of the graph:

        (fingerprint_input)
                v
        [Conv2D]<-(weights)
                v
        [BiasAdd]<-(bias)
                v
            [Relu]
                v
        [MatMul]<-(weights)
                v
        [BiasAdd]<-(bias)
                v
        [MatMul]<-(weights)
                v
        [BiasAdd]<-(bias)
                v
        [MatMul]<-(weights)
                v
        [BiasAdd]<-(bias)
                v

        This produces slightly lower quality results than the 'conv' model, but needs
        fewer weight parameters and computations.

        During training, dropout nodes are introduced after the relu, controlled by a
        placeholder.

        Args:
            fingerprint_input: TensorFlow node that will output audio feature vectors.
            model_settings: Dictionary of information about the model.
            is_training: Whether the model is going to be used for training.

        Returns:
            TensorFlow node outputting logits results, and optionally a dropout
            placeholder.
        """

        input_frequency_size = self.dct_coefficient_count
        input_time_size = self.spectrogram_length
        fingerprint_4d = tf.reshape(fingerprints,
                                    [-1, input_time_size, input_frequency_size, 1])
        first_filter_width = 8
        first_filter_height = input_time_size
        first_filter_count = 186
        first_filter_stride_x = 1
        first_filter_stride_y = 1
        first_weights = tf.Variable(
            tf.truncated_normal(
                [first_filter_height, first_filter_width, 1, first_filter_count],
                stddev=0.01))
        first_bias = tf.Variable(tf.zeros([first_filter_count]))
        first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [
            1, first_filter_stride_y, first_filter_stride_x, 1
        ], 'VALID') + first_bias
        first_relu = tf.nn.relu(first_conv)
        if is_training:
            first_dropout = tf.nn.dropout(first_relu, dropout_keep_prob)
        else:
            first_dropout = first_relu
        first_conv_output_width = math.floor(
            (input_frequency_size - first_filter_width + first_filter_stride_x) /
            first_filter_stride_x)
        first_conv_output_height = math.floor(
            (input_time_size - first_filter_height + first_filter_stride_y) /
            first_filter_stride_y)
        first_conv_element_count = int(
            first_conv_output_width * first_conv_output_height * first_filter_count)
        flattened_first_conv = tf.reshape(first_dropout,
                                          [-1, first_conv_element_count])
        first_fc_output_channels = 128
        first_fc_weights = tf.Variable(
            tf.truncated_normal(
                [first_conv_element_count, first_fc_output_channels], stddev=0.01))
        first_fc_bias = tf.Variable(tf.zeros([first_fc_output_channels]))
        first_fc = tf.matmul(flattened_first_conv, first_fc_weights) + first_fc_bias
        if is_training:
            second_fc_input = tf.nn.dropout(first_fc, dropout_keep_prob)
        else:
            second_fc_input = first_fc
        second_fc_output_channels = 128
        second_fc_weights = tf.Variable(
            tf.truncated_normal(
                [first_fc_output_channels, second_fc_output_channels], stddev=0.01))
        second_fc_bias = tf.Variable(tf.zeros([second_fc_output_channels]))
        second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias
        if is_training:
            final_fc_input = tf.nn.dropout(second_fc, dropout_keep_prob)
        else:
            final_fc_input = second_fc
        label_count = all_labels_count
        final_fc_weights = tf.Variable(
            tf.truncated_normal(
                [second_fc_output_channels, label_count], stddev=0.01))
        final_fc_bias = tf.Variable(tf.zeros([label_count]))
        final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias

        softmax = tf.nn.softmax(final_fc, name='softmax')
        return softmax, final_fc

    def build_fingerprints_graph(self):
        """Builds a fingerprint graph and adds the necessary input & output tensors.

          To use other fingerprint models modify this file. Also preprocessing must be
          modified accordingly.

        Returns:
          input_wav: A placeholder for audio string batch that allows feeding the
                      fingerprints layer with audio bytes for prediction.
          fingerprints: The fingerprints tensor.
        """
        audio_str_tensor = tf.placeholder(tf.string, shape=[None])

        # The CloudML Prediction API always "feeds" the Tensorflow graph with
        # dynamic batch sizes e.g. (?,).  decode_jpeg only processes scalar
        # strings because it cannot guarantee a batch of images would have
        # the same output size.  We use tf.map_fn to give decode_jpeg a scalar
        # string from dynamic batches.
        def decode(audio_str_tensor, desired_samples, window_size_samples, window_stride_samples, dct_coefficient_count):

            wav_decoder = contrib_audio.decode_wav(
			audio_str_tensor, desired_channels=1, desired_samples=desired_samples)

            spectrogram = contrib_audio.audio_spectrogram(
                wav_decoder,
                window_size=window_size_samples,
                stride=window_stride_samples,
                magnitude_squared=True)

            mfcc_fingerprint = contrib_audio.mfcc(
                spectrogram,
                wav_decoder.sample_rate,
                dct_coefficient_count=dct_coefficient_count)

            return mfcc_fingerprint

        fingerprints = tf.map_fn(
                decode, audio_str_tensor, back_prop=False)

        return audio_str_tensor, fingerprints

    def build_graph(self, data_paths, batch_size, graph_mod):
        """Builds generic graph for training or eval."""
        tensors = GraphReferences()
        is_training = graph_mod == GraphMod.TRAIN
        if data_paths:
            tensors.keys, tensors.examples = util.read_examples(
                data_paths,
                batch_size,
                shuffle=is_training,
                num_epochs=None if is_training else 2)
        else:
            tensors.examples = tf.placeholder(tf.string, name='input', shape=(None,))

        if graph_mod == GraphMod.PREDICT:
            audio_input, audio_fingerprints = self.build_fingerprints_graph()
            # Build the fingerprints graph. We later add final trained layers
            # to this graph. This is currently used only for prediction.
            # For training, we use pre-processed data, so it is not needed.
            fingerprints = audio_fingerprints
            tensors.input_audio = audio_input
        else:
            # For training and evaluation we assume data is preprocessed, so the
            # inputs are tf-examples.
            # Generate placeholders for examples.
            with tf.name_scope('inputs'):
                feature_map = {
                    'audio_uri':
                        tf.FixedLenFeature(
                            shape=[], dtype=tf.string, default_value=['']),
                    # Some images may have no labels. For those, we assume a default
                    # label. So the number of labels is label_count+1 for the default
                    # label.
                    'label':
                        tf.FixedLenFeature(
                            shape=[1], dtype=tf.int64,
                            default_value=[self.label_count]),
                    'fingerprint':
                        tf.FixedLenFeature(
                            shape=[self.fingerprint_size], dtype=tf.float32)
                }
                parsed = tf.parse_example(tensors.examples, features=feature_map)
                labels = tf.squeeze(parsed['label'])
                uris = tf.squeeze(parsed['audio_uri'])
                fingerprints = parsed['fingerprint']

        # We assume a default label, so the total number of labels is equal to
        # label_count+1.
        all_labels_count = self.label_count + 1
        with tf.name_scope('final_ops'):
            softmax, logits = self.add_final_training_ops(
                fingerprints,
                all_labels_count,
                dropout_keep_prob=self.dropout if is_training else None)

        # Prediction is the index of the label with the highest score. We are
        # interested only in the top score.
        prediction = tf.argmax(softmax, 1)
        tensors.predictions = [prediction, softmax, fingerprints]

        if graph_mod == GraphMod.PREDICT:
            return tensors

        with tf.name_scope('evaluate'):
            loss_value = loss(logits, labels)

        # Add to the Graph the Ops that calculate and apply gradients.
        if is_training:
            tensors.train, tensors.global_step = training(loss_value)
        else:
            tensors.global_step = tf.Variable(0, name='global_step', trainable=False)

        # Add means across all batches.
        loss_updates, loss_op = util.loss(loss_value)
        accuracy_updates, accuracy_op = util.accuracy(logits, labels)

        if not is_training:
            tf.summary.scalar('accuracy', accuracy_op)
            tf.summary.scalar('loss', loss_op)

        tensors.metric_updates = loss_updates + accuracy_updates
        tensors.metric_values = [loss_op, accuracy_op]
        return tensors

    def build_train_graph(self, data_paths, batch_size):
        return self.build_graph(data_paths, batch_size, GraphMod.TRAIN)

    def build_eval_graph(self, data_paths, batch_size):
        return self.build_graph(data_paths, batch_size, GraphMod.EVALUATE)

    def restore_from_checkpoint(self, session, checkpoint_file,
                                trained_checkpoint_file):
        """To restore model variables from the checkpoint file.

           The graph is assumed to consist of an inception model and other
           layers including a softmax and a fully connected layer. The former is
           pre-trained and the latter is trained using the pre-processed data. So
           we restore this from two checkpoint files.
        Args:
          session: The session to be used for restoring from checkpoint.
          checkpoint_file: Path to the checkpoint file
        """
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(session, checkpoint_file)

    def build_prediction_graph(self):
        """Builds prediction graph and registers appropriate endpoints."""

        tensors = self.build_graph(None, 1, GraphMod.PREDICT)

        keys_placeholder = tf.placeholder(tf.string, shape=[None])
        inputs = {
            'key': keys_placeholder,
            'audio_bytes': tensors.input_wav
        }

        # To extract the id, we need to add the identity function.
        keys = tf.identity(keys_placeholder)
        outputs = {
            'key': keys,
            'prediction': tensors.predictions[0],
            'scores': tensors.predictions[1]
        }

        return inputs, outputs

    def export(self, last_checkpoint, output_dir):
        """Builds a prediction graph and xports the model.

        Args:
          last_checkpoint: Path to the latest checkpoint file from training.
          output_dir: Path to the folder to be used to output the model.
        """
        logging.info('Exporting prediction graph to %s', output_dir)
        with tf.Session(graph=tf.Graph()) as sess:
            # Build and save prediction meta graph and trained variable values.
            inputs, outputs = self.build_prediction_graph()
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            self.restore_from_checkpoint(sess, self.checkpoint_file)
            signature_def = build_signature(inputs=inputs, outputs=outputs)
            signature_def_map = {
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
            }
            builder = saved_model_builder.SavedModelBuilder(output_dir)
            builder.add_meta_graph_and_variables(
                sess,
                tags=[tag_constants.SERVING],
                signature_def_map=signature_def_map)
            builder.save()

    def format_metric_values(self, metric_values):
        """Formats metric values - used for logging purpose."""

        # Early in training, metric_values may actually be None.
        loss_str = 'N/A'
        accuracy_str = 'N/A'
        try:
            loss_str = '%.3f' % metric_values[0]
            accuracy_str = '%.3f' % metric_values[1]
        except (TypeError, IndexError):
            pass

        return '%s, %s' % (loss_str, accuracy_str)


def loss(logits, labels):
    """Calculates the loss from the logits and the labels.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].
    Returns:
      loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def training(loss_op):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(epsilon=0.001)
        train_op = optimizer.minimize(loss_op, global_step)
        return train_op, global_step
