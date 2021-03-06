import os
import argparse
import hashlib
import re
import csv

SILENCE_LABEL = '_silence_'

labels = ['bed',
          'bird',
          'cat',
          'dog',
          'down',
          'eight',
          'five',
          'four',
          'go',
          'happy',
          'house',
          'left',
          'marvin',
          'nine',
          'no',
          'off',
          'on',
          'one',
          'right',
          'seven',
          'sheila',
          'six',
          'stop',
          'three',
          'tree',
          'two',
          'up',
          'wow',
          'yes',
          'zero']

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

def which_set(filename, validation_percentage, testing_percentage):
    """Determines which data partition the file should belong to.

	We want to keep files in the same training, validation, or testing sets even
	if new ones are added over time. This makes it less likely that testing
	samples will accidentally be reused in training when long runs are restarted
	for example. To keep this stability, a hash of the filename is taken and used
	to determine which set it should belong to. This determination only depends on
	the name and the set proportions, so it won't change as other files are added.

	It's also useful to associate particular files as related (for example words
	spoken by the same person), so anything after '_nohash_' in a filename is
	ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
	'bobby_nohash_1.wav' are always in the same set, for example.

	Args:
	  filename: File path of the data sample.
	  validation_percentage: How much of the data set to use for validation.
	  testing_percentage: How much of the data set to use for testing.

	Returns:
	  String, one of 'training', 'validation', or 'testing'.
	"""
    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(hash_name).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_WAVS_PER_CLASS + 1)) *
                       (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='Percentage of data for validation')

    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='Percentage of data for validation')

    parser.add_argument(
        '--cloud',
        type=bool,
        default=False,
        help='Use cloud prefix')

    parser.add_argument(
        '--cloud_prefix',
        type=str,
        default='gs://speech-commands-recognition-ml/data/',
        help='Location of speech commands folder on cloud')

    parser.add_argument(
        '--local_prefix',
        type=str,
        default='/Users/dhananjaydeshpande/PycharmProjects/speech-commands-recognition/data/',
        help='Location of speech commands folder locally')

    parser.add_argument(
        '--speech_commands_data_folder',
        type=str,
        default='speech_commands_v0.01',
        help='Unarchived data from Google Speech Commands Dataset')

    parser.add_argument(
        '--silence_percentage',
        type=float,
        default=10.0,
        help="""\
        How much of the training data should be silence.
        """)

    args = parser.parse_args()

    print('validation percentage : %d' % args.validation_percentage)
    print('testing_percentage : %d' % args.testing_percentage)

    if args.cloud:
        prefix = args.cloud_prefix
    else:
        prefix = args.local_prefix

    root_dir = args.local_prefix + args.speech_commands_data_folder

    print(root_dir)
    with open('eval.csv', 'w') as fvalidate, open('test.csv', 'w') as ftest, open('train.csv', 'w') as ftrain:

        wvalidate = csv.writer(fvalidate)
        wtest = csv.writer(ftest)
        wtrain = csv.writer(ftrain)

        eval_index = 0
        train_index = 0
        test_index = 0

        silence_path = None

        for dir_name, sub_dir, file_list in os.walk(root_dir, topdown=False):
            label = dir_name.split('/')[-1]

            if label in labels:
                for fname in file_list:
                    print('\t%s' % fname)

                    absolute_path = prefix +  args.speech_commands_data_folder + '/' + label + '/' + fname

                    if silence_path == None:
                        silence_path = absolute_path
                        wvalidate.writerow([silence_path, SILENCE_LABEL])
                        wtest.writerow([silence_path, SILENCE_LABEL])
                        wtrain.writerow([silence_path, SILENCE_LABEL])
                        continue

                    mode = which_set(fname, args.validation_percentage, args.testing_percentage)

                    if mode == 'validation':

                        eval_index+=1
                        if (eval_index % (100 / args.silence_percentage) == 0):
                            wvalidate.writerow([silence_path, SILENCE_LABEL])
                            eval_index+=1

                        wvalidate.writerow([absolute_path, label])

                    elif mode == 'testing':

                        test_index+=1
                        if (test_index % (100 / args.silence_percentage) == 0):
                            wtest.writerow([silence_path, SILENCE_LABEL])
                            test_index+=1

                        wtest.writerow([absolute_path, label])

                    else:

                        train_index+=1
                        if (train_index % (100 / args.silence_percentage) == 0):
                            wtrain.writerow([silence_path, SILENCE_LABEL])
                            train_index+=1

                        wtrain.writerow([absolute_path, label])

        print('Total validation samples : %d' % eval_index)
        print('Total test samples : %d' % test_index)
        print('Total training samples : %d' % train_index)
