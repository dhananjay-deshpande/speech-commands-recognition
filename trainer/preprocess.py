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

#logging.basicConfig(filename='test.log', filemode='w', level=logging.DEBUG)

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
incompatible_image = Metrics.counter('main', 'incompatible_image')
invalid_uri = Metrics.counter('main', 'invalid_file_name')
unlabeled_image = Metrics.counter('main', 'unlabeled_image')
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
        if not uri or not uri.startswith('gs://'):
            invalid_uri.inc()
            return

        # Default label for labels not in labels file is 'unknown'
        # with id 11

        label_ids = []
        for label in row[1:]:
            try:
                label_ids.append(self.label_to_id_map[label.strip()])
            except KeyError:
                unknown_label.inc()

        labels_count.inc(len(label_ids))

        if not label_ids:
            unlabeled_image.inc()
            label_ids.append(1)

        yield row[0], label_ids

class MFCCGraph(object):
    """Builds a graph and uses it to extract mfcc fingerprint from audio
	"""

    def __init__(self, tf_session):
        self.tf_session = tf_session

        # input_wav_filename is the tensor that contains the audio URI
        # It is used to decode wav files and obtain mfcc fingerprints.

        self.input_wav_filename, self.mfcc_fingerprint = self.build_graph()

        init_op = tf.global_variables_initializer()
        self.tf_session.run(init_op)

    def build_graph(self):
        """ Graph to extract mfcc fingerprint given wav file

		  Here we add the necessary input & output tensors, to decode wav,
		  serialize mfcc fingerprint, restore from checkpoint etc.

		Returns:
		  input_wav_filename: A tensor containing wav filename as the input layer.
		  mfcc_fingerprint: The MFCC fingerprint tensor, that will be materialized later.
		"""

        input_wav_filename = tf.placeholder(tf.string, [])


        wav_loader = io_ops.read_file(input_wav_filename)

        wav_decoder = contrib_audio.decode_wav(
			wav_loader, desired_channels=1, desired_samples=16000)

        spectrogram = contrib_audio.audio_spectrogram(
            wav_decoder.audio,
            window_size=480,
            stride=160,
            magnitude_squared=True)

        mfcc_fingerprint = contrib_audio.mfcc(
            spectrogram,
            wav_decoder.sample_rate,
            dct_coefficient_count=40)

        return input_wav_filename, mfcc_fingerprint

    def calculate_fingerprint(self, filename):
        """Get the mfcc fingerprint for a given WAV image.

		Args:
			filename : URI of WAV file

		Returns:
			mfcc fingerprint for WAV file
		"""

        print('Input filename %s' % filename)

        return self.tf_session.run(
            self.mfcc_fingerprint, feed_dict={self.input_wav_filename: filename})

class TFExampleFromAudioDoFn(beam.DoFn):
    """Embeds audio mfcc fingerprint and labels, stores them in tensorflow.Example.

	(uri, label_id, mfcc fingerprint) -> (tensorflow.Example).

	Output proto contains 'label', 'image_uri' and 'mfcc fingerprint'.

	Attributes:
	  image_graph_uri: an uri to gcs bucket where serialized image graph is
					   stored.
	"""

    def __init__(self):
        self.tf_session = None
        self.graph = None
        self.preprocess_graph = None

    def start_bundle(self, context=None):
        # There is one tensorflow session per instance of TFExampleFromAudioDoFn.
        # The same instance of session is re-used between bundles.
        # Session is closed by the destructor of Session object, which is called
        # when instance of TFExampleFromAudioDoFn() is destructed.
        if not self.graph:
            self.graph = tf.Graph()
            self.tf_session = tf.InteractiveSession(graph=self.graph)
            with self.graph.as_default():
                self.preprocess_graph = MFCCGraph(self.tf_session)

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

        logging.info('Got URI %s: %s', uri, str(label_ids))

        try:
            fingerprint = self.preprocess_graph.calculate_fingerprint(uri)
        except errors.InvalidArgumentError as e:
            incompatible_image.inc()
            logging.warning('Could not generate fingerprint from %s: %s', uri, str(e))
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
    read_input_source = beam.io.ReadFromText(
        opt.input_path, strip_trailing_newlines=True)
    read_label_source = beam.io.ReadFromText(
        opt.input_dict, strip_trailing_newlines=True)
    labels = (p | 'Read dictionary' >> read_label_source)
    print(labels)
    _ = (p
         | 'Read input' >> read_input_source
         | 'Parse input' >> beam.Map(lambda line: csv.reader([line]).next())
         | 'Extract label ids' >> beam.ParDo(ExtractLabelIdsDoFn(),
                                             beam.pvalue.AsIter(labels))
         | 'Extract Fingerprint and make TFExample' >> beam.ParDo(TFExampleFromAudioDoFn())
         | 'SerializeToString' >> beam.Map(lambda x: x.SerializeToString())
         | 'Save to disk'
         >> beam.io.WriteToTFRecord(opt.output_path,
                                    file_name_suffix='.tfrecord.gz'))

def run(in_args=None):
    """Runs the pre-processing pipeline."""

    logging.info('Running pipeline')
    pipeline_options = PipelineOptions.from_dictionary(vars(in_args))
    with beam.Pipeline(options=pipeline_options) as p:
        try:
            configure_pipeline(p, in_args)
        except:
            logging.error('Error while executing pipeline')


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
        default='speechcommands-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
        help='A unique job identifier.')
    parser.add_argument(
        '--num_workers', default=20, type=int, help='The number of workers.')
    parser.add_argument('--cloud', default=False, action='store_true')
    parser.add_argument(
        '--runner',
        help='See Dataflow runners, may be blocking'
             ' or not, on cloud or not, etc.')

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
