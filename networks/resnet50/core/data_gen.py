"""
    The batch creation file to create generators that yield the batches.

    Work of Gerber and Glinski.
"""
from random import randint

import numpy as np
import time

def generate_test_data_h5(test_type, dataset, segment_size, spectrogram_height, max_files_per_speaker=0, max_segments_per_utterance=0):
    if test_type not in ['all', 'short', 'long']:
        raise ValueError(':test_type should be either "all", "short" or "long".')

    if max_files_per_speaker > 0 and max_segments_per_utterance == 0:
        raise ValueError(":max_files_per_speaker and :max_segments_per_utterance should both be either zero or non-zero, given: {}, {}".format(max_files_per_speaker, max_segments_per_utterance))
    if max_segments_per_utterance > 0 and max_files_per_speaker == 0:    
        raise ValueError(":max_files_per_speaker and :max_segments_per_utterance should both be either zero or non-zero, given: {}, {}".format(max_files_per_speaker, max_segments_per_utterance))

    # Load the speakers(-list) used for testing
    # 
    all_speakers = np.array(dataset.get_test_speaker_list())
    num_speakers = all_speakers.shape[0]

    # Calculate the amount of spectrograms in total and the
    # total length of all those spectrograms according to the statistics
    # 
    total_test_length = np.sum(list(map(lambda x: np.sum(x), map(lambda y: dataset.get_test_file()['statistics'][y][dataset.get_test_statistics()[y][test_type]], all_speakers))))
    num_spectrograms = dataset.get_test_num_segments(test_type)

    # Get an upper bound estimate for the amount of segments actually produced
    # in terms of spectrogram slices of length :segment_size
    # 
    if max_files_per_speaker == 0 and max_segments_per_utterance == 0:
        num_segments = total_test_length // segment_size
    else:
        num_segments = num_speakers * max_files_per_speaker * max_segments_per_utterance

    X_test = np.zeros((num_segments, segment_size, spectrogram_height))
    y_test = []

    data_file = dataset.get_test_file()

    # Iterate over all speakers and extract spectrograms of length segment_size
    # 
    pos = 0
    for i in range(num_speakers):
        speaker_name = all_speakers[i]

        speaker_utterances_indices = dataset.get_test_statistics()[speaker_name][test_type]

        if max_files_per_speaker > 0:
            speaker_utterances_indices = np.random.choice(speaker_utterances_indices, max_files_per_speaker)

        for utterance_index in speaker_utterances_indices:
            # Extract the full spectrogram
            #  
            full_spect = data_file['data/'+speaker_name][utterance_index]

            # lehl@2019-12-03: Spectrogram reshaped to get the format (time_length, spec_height) 
            # 
            spect = full_spect.reshape((full_spect.shape[0] // spectrogram_height, spectrogram_height))

            # Standardize
            mu = np.mean(spect, 0, keepdims=True)
            stdev = np.std(spect, 0, keepdims=True)
            spect = (spect - mu) / (stdev + 1e-5)

            # Extract as many slices from each spectrogram-utterance
            # as there would be space, as in how many full windows (segment_size) would fit in the
            # length of this utterance if they'd be placed consecutively without allowing partial segments
            # 
            if max_segments_per_utterance > 0:
                segments_to_extract = range(0, max_segments_per_utterance)
            else:
                segments_to_extract = range(spect.shape[0] // segment_size)

            for segment in segments_to_extract:
                seg_idx = randint(0, spect.shape[1] - segment_size)
                X_test[pos] = spect[seg_idx:seg_idx + segment_size, :]
                y_test.append(i)

                pos += 1

    # Close h5py Connection
    data_file.close()

    return X_test[0:len(y_test)].reshape((len(y_test), segment_size, spectrogram_height, 1)), np.asarray(y_test, dtype=np.int)

# lehl@2019-12-02: Batch generator using h5py dataset and speaker list
# 
# Params:
# batch_type        ['train', 'al', 'val'], for statistics access
# dataset           DeepVoiceDataset instance containing references to the datasets and statistics
# 
def batch_generator_h5(batch_type, dataset, batch_size=100, segment_size=40, spectrogram_height=128):
    # Calculates the amount of indices used for the given :batch_type across all
    # speakers, such that this equals to the amount of spectrograms available for
    # training
    #
    num_segments = dataset.get_train_num_segments(batch_type)
    all_speakers = np.array(dataset.get_train_speaker_list())
    num_speakers = all_speakers.shape[0]

    # build as many batches as the training set is big
    # 
    while 1:
        for i in range((num_segments // batch_size) + 1):
            Xb = np.zeros((batch_size, segment_size, spectrogram_height), dtype=np.float32)
            yb = np.zeros(batch_size, dtype=np.int32)

            data_file = dataset.get_train_file()

            for j in range(0, batch_size):
                # TODO: lehl@2019-12-07: Check with batch_generator_divergence_optimised implementation
                # (see ZHAW_deep_voice Version before VT1)
                # 
                speaker_index = randint(0, num_speakers - 1)
                speaker_name = all_speakers[speaker_index]

                # Extract Spectrogram
                # Choose from all the utterances of the given speaker randomly
                # 
                utterance_index = np.random.choice(dataset.get_train_statistics()[speaker_name][batch_type])
                
                # lehl@2019-12-13:
                # If batches are generated as the statistics are updated due to active learning,
                # it might be possible to draw from incides that are not really available, this
                # is not a great solution, but quicker than ensuring these processes lock each other
                #
                full_spect = data_file['data/' + speaker_name][utterance_index]

                # lehl@2019-12-03: Spectrogram needs to be reshaped with (time_length, 128) and then
                # transposed as the expected ordering is (128, time_length)
                # TODO: Can this be improved (check how it's refactored again in return statement)
                # 
                spect = full_spect.reshape((full_spect.shape[0] // spectrogram_height, spectrogram_height))

                # Standardize
                mu = np.mean(spect, 0, keepdims=True)
                stdev = np.std(spect, 0, keepdims=True)
                spect = (spect - mu) / (stdev + 1e-5)

                # Extract random :segment_size long part of the spectrogram
                # 
                seg_idx = randint(0, spect.shape[1] - segment_size)
                Xb[j] = spect[seg_idx:seg_idx + segment_size, :]

                # Set label
                # 
                yb[j] = speaker_index

            # Close h5py Connection
            data_file.close()

            yield Xb.reshape((batch_size, segment_size, spectrogram_height, 1)), np.eye(num_speakers)[yb]
