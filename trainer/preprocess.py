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

"""Example dataflow pipeline for preparing audio training data.

The tool requires two main input files:

'input' - URI to csv file, using format:
gs://audio_uri1,labela
gs://audio_uri2,labelb
...

'input_dict' - URI to a text file listing all labels (one label per line):
labela
labelb

The output data is in format accepted by Cloud ML framework.

This tool produces outputs as follows.
It creates one training example per each line of the created csv file.
When processing CSV file:
- all labels that are not present in input_dict are treated as unknown

To execute this pipeline locally using default options, run this script
with no arguments. To execute on cloud pass single argument --cloud.

To execute this pipeline on the cloud using the Dataflow service and non-default
options:
python -E preprocess.py \
--input_path=PATH_TO_INPUT_CSV_FILE \
--input_dict=PATH_TO_INPUT_DIC_TXT_FILE \
--output_path=YOUR_OUTPUT_PATH \
--cloud

To run this pipeline locally run the above command without --cloud.

"""

import argparse
import csv
import datetime
import errno
import io
import logging
import os
import subprocess
import sys
import numpy as np

BACKGROUND_NOISE_DIR_NAME = '_background_noise_'

#logging.basicConfig(filename='test.log', filemode='w', level=logging.DEBUG)
log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import apache_beam as beam
from apache_beam.metrics import Metrics
try:
    try:
        from apache_beam.options.pipeline_options import PipelineOptions
    except ImportError:
        from apache_beam.utils.pipeline_options import PipelineOptions
except ImportError:
    from apache_beam.utils.options import PipelineOptions


import tensorflow as tf

from tensorflow.python.ops import io_ops
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import gfile
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

slim = tf.contrib.slim

error_count = Metrics.counter('main', 'errorCount')
missing_label_count = Metrics.counter('main', 'missingLabelCount')
csv_rows_count = Metrics.counter('main', 'csvRowsCount')
labels_count = Metrics.counter('main', 'labelsCount')
labels_without_ids = Metrics.counter('main', 'labelsWithoutIds')
existing_file = Metrics.counter('main', 'existingFile')
non_existing_file = Metrics.counter('main', 'nonExistingFile')
skipped_empty_line = Metrics.counter('main', 'skippedEmptyLine')
fingerprint_good = Metrics.counter('main', 'fingerprint_good')
fingerprint_bad = Metrics.counter('main', 'fingerprint_bad')
incompatible_audio = Metrics.counter('main', 'incompatible_audio')
invalid_uri = Metrics.counter('main', 'invalid_file_name')
unlabeled_audio = Metrics.counter('main', 'unlabeled_audio')
unknown_label = Metrics.counter('main', 'unknown_label')

class Default(object):
    """Default values of variables."""
    FORMAT = 'wav'

class ExtractLabelIdsDoFn(beam.DoFn):
    """Extracts (uri, label_id) tuples from CSV rows.
	"""

    def start_bundle(self, context=None):
        self.label_to_id_map = {}

    # The try except is for compatiblity across multiple versions of the sdk
    def process(self, row, all_labels):

        log.info('Inside ExtractLabelIdsDoFn')

        try:
            row = row.element
        except AttributeError:
            pass

        if not self.label_to_id_map:
            for i, label in enumerate(all_labels):
                label = label.strip()
                if label:
                    self.label_to_id_map[label] = i

        # Row format is: audio_uri,label_id
        if not row:
            skipped_empty_line.inc()
            return

        csv_rows_count.inc()
        uri = row[0]
        #if not uri or not uri.startswith('gs://'):
        #    invalid_uri.inc()
        #    return

        # Default label for labels not in labels file is 'unknown'
        # with id 11

        label_ids = []
        for label in row[1:]:
            try:
                label_ids.append(self.label_to_id_map[label.strip()])
            except KeyError:
                unknown_label.inc()

        labels_count.inc(len(label_ids))

        # If label wasnt found add '_unknown_' -> 1
        if not label_ids:
            unlabeled_audio.inc()
            label_ids.append(1)

        log.info('Emitting filename:' + str(row[0])+ ' label ids: ' + str(label_ids))
        yield row[0], label_ids

class MFCCGraph(object):
    """Builds a graph and uses it to extract mfcc fingerprint from audio
	"""

    def __init__(self, tf_session, opt):
        self.tf_session = tf_session
        self.opt = opt

        self.desired_samples = int(self.opt.sample_rate * self.opt.clip_duration_ms / 1000)
        self.window_size_samples = int(self.opt.sample_rate * self.opt.window_size_ms / 1000)
        self.window_stride_samples = int(self.opt.sample_rate * self.opt.window_stride_ms / 1000)
        self.time_shift_samples = int((self.opt.time_shift_ms * self.opt.sample_rate) / 1000)
        self.background_frequency = self.opt.background_frequency
        self.background_volume_range = self.opt.background_volume
        # input_wav_filename is the tensor that contains the audio URI
        # It is used to decode wav files and obtain mfcc fingerprints.

        self.prepare_background_data()
        self.build_graph()

        init_op = tf.global_variables_initializer()
        self.tf_session.run(init_op)

    def prepare_background_data(self):
        """Searches a folder for background noise audio, and loads it into memory.

        It's expected that the background audio samples will be in a subdirectory
        specified in 'background_noise_path', as .wavs that match
        the sample rate of the training data, but can be much longer in duration.

        If the background noise folder doesn't exist at all, this isn't an
        error, it's just taken to mean that no background noise augmentation should
        be used. If the folder does exist, but it's empty, that's treated as an
        error.

        Returns:
            List of raw PCM-encoded audio samples of background noise.

        """

        self.background_data = []

        if not self.opt.background_noise_path:
            return

        background_dir = self.opt.background_noise_path
        if not os.path.exists(background_dir):
            return

        with tf.Session(graph=tf.Graph()) as sess:
            wav_filename_placeholder = tf.placeholder(tf.string, [])
            wav_loader = io_ops.read_file(wav_filename_placeholder)
            wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
            search_path = os.path.join(background_dir, '*.wav')
            for wav_path in gfile.Glob(search_path):
                wav_data = sess.run(wav_decoder,
                    feed_dict={wav_filename_placeholder: wav_path}).audio.flatten()
                self.background_data.append(wav_data)

    def build_graph(self):
        """ Graph to extract mfcc fingerprint given wav file

		  Here we add the necessary input & output tensors, to decode wav,
		  serialize mfcc fingerprint, restore from checkpoint etc.

		Returns:
		  input_wav_filename: A tensor containing wav filename as the input layer.
		  mfcc_fingerprint: The MFCC fingerprint tensor, that will be materialized later.
		"""

        self.wav_filename_placeholder_ = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(self.wav_filename_placeholder_)
        wav_decoder = contrib_audio.decode_wav(
			wav_loader, desired_channels=1, desired_samples=self.desired_samples)

        # Allow the audio sample's volume to be adjusted.
        self.foreground_volume_placeholder_ = tf.placeholder(tf.float32, [])
        scaled_foreground = tf.multiply(wav_decoder.audio,
                                    self.foreground_volume_placeholder_)

        # Shift the sample's start position, and pad any gaps with zeros.
        self.time_shift_padding_placeholder_ = tf.placeholder(tf.int32, [2, 2])
        self.time_shift_offset_placeholder_ = tf.placeholder(tf.int32, [2])
        padded_foreground = tf.pad(
            scaled_foreground,
            self.time_shift_padding_placeholder_,
            mode='CONSTANT')
        sliced_foreground = tf.slice(padded_foreground,
                                self.time_shift_offset_placeholder_,
                                [self.desired_samples, -1])
        # Mix in background noise.
        self.background_data_placeholder_ = tf.placeholder(tf.float32,
                                                       [self.desired_samples, 1])
        self.background_volume_placeholder_ = tf.placeholder(tf.float32, [])
        background_mul = tf.multiply(self.background_data_placeholder_,
                                 self.background_volume_placeholder_)
        background_add = tf.add(background_mul, sliced_foreground)
        background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)

        spectrogram = contrib_audio.audio_spectrogram(
            background_clamp,
            window_size=self.window_size_samples,
            stride=self.window_stride_samples,
            magnitude_squared=True)

        self.mfcc_fingerprint_ = contrib_audio.mfcc(
            spectrogram,
            wav_decoder.sample_rate,
            dct_coefficient_count=self.opt.dct_coefficient_count)

    def calculate_fingerprint(self, filename, label_ids):
        """Get the mfcc fingerprint for a given WAV audio.

		Args:
			filename : URI of WAV file

		Returns:
			mfcc fingerprint for WAV file
		"""

        #print('Input filename %s' % filename)

        # If we're time shifting, set up the offset for this sample.
        if self.time_shift_samples > 0:
            time_shift_amount = np.random.randint(-self.time_shift_samples, self.time_shift_samples)
        else:
            time_shift_amount = 0

        if time_shift_amount > 0:
            time_shift_padding = [[time_shift_amount, 0], [0, 0]]
            time_shift_offset = [0, 0]
        else:
            time_shift_padding = [[0, -time_shift_amount], [0, 0]]
            time_shift_offset = [-time_shift_amount, 0]

        input_dict = {
            self.wav_filename_placeholder_: filename,
            self.time_shift_padding_placeholder_: time_shift_padding,
            self.time_shift_offset_placeholder_: time_shift_offset,
        }

        use_background = False
        if self.opt.background_noise_path:
            use_background = True

        # Choose a section of background noise to mix in.
        if use_background:
            background_index = np.random.randint(len(self.background_data))
            background_samples = self.background_data[background_index]
            background_offset = np.random.randint(
                0, len(background_samples) - self.desired_samples)
            background_clipped = background_samples[background_offset:(
            background_offset + self.desired_samples)]
            background_reshaped = background_clipped.reshape([self.desired_samples, 1])
            if np.random.uniform(0, 1) < self.background_frequency:
                background_volume = np.random.uniform(0, self.background_volume_range)
            else:
                background_volume = 0
        else:
            background_reshaped = np.zeros([self.desired_samples, 1])
            background_volume = 0

        input_dict[self.background_data_placeholder_] = background_reshaped
        input_dict[self.background_volume_placeholder_] = background_volume


        # handle silence
        if 0 in label_ids:
            input_dict[self.foreground_volume_placeholder_] = 0
        else:
            input_dict[self.foreground_volume_placeholder_] = 1

        return self.tf_session.run(
            self.mfcc_fingerprint_, feed_dict=input_dict)

class TFExampleFromAudioDoFn(beam.DoFn):
    """Embeds audio mfcc fingerprint and labels, stores them in tensorflow.Example.

	(uri, label_id, mfcc fingerprint) -> (tensorflow.Example).

	Output proto contains 'label', 'audio_uri' and 'mfcc fingerprint'.

	Attributes:
	  audio_graph_uri: an uri to gcs bucket where serialized audio graph is
					   stored.
	"""

    def __init__(self, opt):
        self.tf_session = None
        self.graph = None
        self.preprocess_graph = None
        assert isinstance(opt, object)
        self.opt = opt

    def start_bundle(self, context=None):
        # There is one tensorflow session per instance of TFExampleFromAudioDoFn.
        # The same instance of session is re-used between bundles.
        # Session is closed by the destructor of Session object, which is called
        # when instance of TFExampleFromAudioDoFn() is destructed.
        if not self.graph:
            self.graph = tf.Graph()
            self.tf_session = tf.InteractiveSession(graph=self.graph)
            with self.graph.as_default():
                self.preprocess_graph = MFCCGraph(self.tf_session, self.opt)

    def process(self, element):

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

        def _float_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))

        try:
            element = element.element
        except AttributeError:
            pass

        uri, label_ids = element

        log.info('Got URI %s: %s', uri, str(label_ids))

        try:
            fingerprint = self.preprocess_graph.calculate_fingerprint(uri, label_ids)
        except errors.InvalidArgumentError as e:
            incompatible_audio.inc()
            log.warning('Could not generate fingerprint from %s: %s', uri, str(e))
            return

        if fingerprint.any():
            fingerprint_good.inc()
        else:
            fingerprint_bad.inc()

        example = tf.train.Example(features=tf.train.Features(feature={
            'audio_uri': _bytes_feature([uri]),
            'fingerprint': _float_feature(fingerprint.ravel().tolist()),
        }))

        if label_ids:
            label_ids.sort()
            example.features.feature['label'].int64_list.value.extend(label_ids)

        yield example

def configure_pipeline(p, opt):
    """Specify PCollection and transformations in pipeline."""
    log.info('reading input source')
    read_input_source = beam.io.ReadFromText(
        opt.input_path, strip_trailing_newlines=True)
    log.info('reading label source: ' + str(opt.input_dict))
    read_label_source = beam.io.ReadFromText(
        opt.input_dict, strip_trailing_newlines=True)
    log.info('reading labels')
    labels = (p | 'Read dictionary' >> read_label_source)
    log.info(labels)
    _ = (p
         | 'Read input' >> read_input_source
         | 'Parse input' >> beam.Map(lambda line: csv.reader([line]).next())
         | 'Extract label ids' >> beam.ParDo(ExtractLabelIdsDoFn(),
                                             beam.pvalue.AsIter(labels))
         | 'Extract Fingerprint and make TFExample' >> beam.ParDo(TFExampleFromAudioDoFn(opt))
         | 'SerializeToString' >> beam.Map(lambda x: x.SerializeToString())
         | 'Save to disk'
         >> beam.io.WriteToTFRecord(opt.output_path,
                                    file_name_suffix='.tfrecord.gz'))

def run(in_args=None):
    """Runs the pre-processing pipeline."""

    log.info('Running pipeline')
    pipeline_options = PipelineOptions.from_dictionary(vars(in_args))
    with beam.Pipeline(options=pipeline_options) as p:
        try:
            configure_pipeline(p, in_args)
        except:
            log.error('Error while executing pipeline')


def default_args(argv):
    """Provides default values for Workflow flags."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_path',
        required=True,
        help='Input specified as uri to CSV file. Each line of csv file '
             'contains comma-separated GCS uri to an audio and label')
    parser.add_argument(
        '--input_dict',
        dest='input_dict',
        required=True,
        help='Input dictionary. Specified as text file uri. '
             'Each line of the file stores one label.')
    parser.add_argument(
        '--output_path',
        required=True,
        help='Output directory to write results to.')
    parser.add_argument(
        '--project',
        type=str,
        help='The cloud project name to be used for running this pipeline')
    parser.add_argument(
        '--job_name',
        type=str,
        default='speech-commands-recognition-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
        help='A unique job identifier.')
    parser.add_argument(
        '--num_workers', default=20, type=int, help='The number of workers.')
    parser.add_argument('--cloud', default=False, action='store_true')
    parser.add_argument(
        '--runner',
        help='See Dataflow runners, may be blocking'
             ' or not, on cloud or not, etc.')
    parser.add_argument(
        '--background_noise_path',
        dest='background_noise_path',
        help='GCS uri to an audio background noise directory')
    parser.add_argument(
        '--background_volume',
        type=float,
        default=0.1,
        help="""\
        How loud the background noise should be, between 0 and 1.
        """)
    parser.add_argument(
        '--background_frequency',
        type=float,
        default=0.8,
        help="""\
        How many of the training samples have background noise mixed in.
        """)
    parser.add_argument(
        '--time_shift_ms',
        type=float,
        default=100.0,
        help="""\
        Range to randomly shift the training audio by in time.
        """)
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs',)
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the wavs',)
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=30.0,
        help='How long each spectrogram timeslice is.',)
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=10.0,
        help='How far to move in time between spectogram timeslices.',)
    parser.add_argument(
        '--dct_coefficient_count',
        type=int,
        default=40,
        help='How many bins to use for the MFCC fingerprint',)

    parsed_args, _ = parser.parse_known_args(argv)

    if parsed_args.cloud:
        # Flags which need to be set for cloud runs.
        default_values = {
            'project':
                get_cloud_project(),
            'temp_location':
                os.path.join(os.path.dirname(parsed_args.output_path), 'temp'),
            'runner':
                'DataflowRunner',
            'save_main_session':
                True,
        }
    else:
        # Flags which need to be set for local runs.
        default_values = {
            'runner': 'DirectRunner',
        }

    for kk, vv in default_values.iteritems():
        if kk not in parsed_args or not vars(parsed_args)[kk]:
            vars(parsed_args)[kk] = vv

    return parsed_args

def get_cloud_project():
    cmd = [
        'gcloud', '-q', 'config', 'list', 'project',
        '--format=value(core.project)'
    ]
    with open(os.devnull, 'w') as dev_null:
        try:
            res = subprocess.check_output(cmd, stderr=dev_null).strip()
            if not res:
                raise Exception('--cloud specified but no Google Cloud Platform '
                                'project found.\n'
                                'Please specify your project name with the --project '
                                'flag or set a default project: '
                                'gcloud config set project YOUR_PROJECT_NAME')
            return res
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise Exception('gcloud is not installed. The Google Cloud SDK is '
                                'necessary to communicate with the Cloud ML service. '
                                'Please install and set up gcloud.')
            raise

def main(argv):
    arg_dict = default_args(argv)
    run(arg_dict)

if __name__ == '__main__':
	main(sys.argv[1:])
	print(labels_count)
