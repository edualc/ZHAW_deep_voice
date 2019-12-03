"""
    The batch creation file to create generators that yield the batches.

    Work of Gerber and Glinski.
"""
from random import randint, sample, choice

import numpy as np
import time

from common.spectrogram.spectrogram_extractor import extract_spectrogram
from . import settings


def transform(Xb, yb):
    return Xb, yb


# Shuffles both array along the first Dimensions, in a way that they end up at the same position.
def shuffle_data(Xb, yb):
    rng_state = np.random.get_state()
    np.random.shuffle(Xb)
    np.random.set_state(rng_state)
    np.random.shuffle(yb)


# Extracts the Spectorgram and discards all padded Data
def extract(spectrogram, segment_size):
    return extract_spectrogram(spectrogram, segment_size, settings.FREQ_ELEMENTS)


# generates the data for testing the network, with the specified segment_size (timewindow)
def generate_test_data(X, y, segment_size):
    segments = X.shape[0] * 3 * (800 // segment_size)
    X_test = np.zeros((segments, 1, settings.FREQ_ELEMENTS, segment_size), dtype=np.float32)
    y_test = []

    pos = 0
    for i in range(len(X)):
        spect = extract(X[i, 0], segment_size)

        for j in range(int(spect.shape[1] / segment_size)):
            y_test.append(y[i])
            seg_idx = j * segment_size
            X_test[pos, 0] = spect[:, seg_idx:seg_idx + segment_size]
            pos += 1

    return X_test[0:len(y_test)], np.asarray(y_test, dtype=np.int32)


# # Batch generator for CNNs
# def batch_generator(X, y, batch_size=100, segment_size=100):
#     segments = X.shape[0]
#     bs = batch_size
#     speakers = np.amax(y) + 1
#     # build as much batches as fit into the training set
#     while 1:
#         for i in range((segments + bs - 1) // bs):
#             Xb = np.zeros((bs, 1, settings.FREQ_ELEMENTS, segment_size), dtype=np.float32)
#             yb = np.zeros(bs, dtype=np.int32)
#             # here one batch is generated
#             for j in range(0, bs):
#                 speaker_idx = randint(0, len(X) - 1)
#                 if y is not None:
#                     yb[j] = y[speaker_idx]
#                 spect = extract(X[speaker_idx, 0], segment_size)
#                 seg_idx = randint(0, spect.shape[1] - segment_size)
#                 Xb[j, 0] = spect[:, seg_idx:seg_idx + segment_size]
#             yield Xb, transformy(yb, bs, speakers)


# '''creates the a batch for CNN networks, with Pariwise Labels, 
# for use with core.pairwise_kl_divergence_full_labels'''


# def batch_generator_v2(X, y, batch_size=100, segment_size=100):
#     segments = X.shape[0]
#     bs = batch_size
#     speakers = np.amax(y) + 1
#     # build as much batches as fit into the training set
#     while 1:
#         for i in range((segments + bs - 1) // bs):
#             Xb = np.zeros((bs, 1, settings.FREQ_ELEMENTS, segment_size), dtype=np.float32)
#             yb = np.zeros(bs, dtype=np.int32)
#             # here one batch is generated
#             for j in range(0, bs):
#                 speaker_idx = randint(0, len(X) - 1)
#                 if y is not None:
#                     yb[j] = y[speaker_idx]
#                 spect = extract(X[speaker_idx, 0], segment_size)
#                 seg_idx = randint(0, spect.shape[1] - segment_size)
#                 Xb[j, 0] = spect[:, seg_idx:seg_idx + segment_size]
#             yield Xb, create_pairs(yb)


# Batch generator von LSTMS
def batch_generator_lstm(X, y, batch_size=100, segment_size=15):
    segments = X.shape[0]
    speakers = np.amax(y) + 1
    
    # build as much batches as fit into the training set
    while 1:
        for i in range((segments + batch_size - 1) // batch_size):
            Xb = np.zeros((batch_size, 1, settings.FREQ_ELEMENTS, segment_size), dtype=np.float32)
            yb = np.zeros(batch_size, dtype=np.int32)
            # here one batch is generated
            for j in range(0, batch_size):
                speaker_idx = randint(0, len(X) - 1)

                if y is not None:
                    yb[j] = y[speaker_idx]
                spect = extract(X[speaker_idx, 0], segment_size)
                seg_idx = randint(0, spect.shape[1] - segment_size)
                Xb[j, 0] = spect[:, seg_idx:seg_idx + segment_size]

            yield Xb.reshape(batch_size, segment_size, settings.FREQ_ELEMENTS), transformy(yb, batch_size, speakers)

# Optimized version of :batch_generator_lstm to adress matching issues when used with
# high number of classes (>100) to find reasonable comparisons
#
def batch_generator_divergence_optimised(X, y, batch_size=100, segment_size=15, spectrogram_height=128):
    segments = X.shape[0]
    bs = batch_size
    speakers = np.amax(y) + 1
    
    # build as much batches as fit into the training set
    while 1:
        for i in range((segments + bs - 1) // bs):
            # prepare arrays
            Xb = np.zeros((bs, 1, spectrogram_height, segment_size), dtype=np.float32)
            yb = np.zeros(bs, dtype=np.int32)
            #choose max. 100 speakers from all speakers contained in X (no duplicates!)
            population = set(y)
            n_speakers = min(len(population), 100)
            samples = sample(population, n_speakers)
            # here one batch is generated
            for j in range(0, bs):
                # choose random sentence of one speaker out of the 100 sampled above (duplicates MUST be allowed here!)
                # calculate the index of the sentence in X and y to access the data
                speaker_id = randint(0, len(samples) - 1)

                indices_of_speaker = np.where(y == speaker_id)[0]
                speaker_idx = choice(indices_of_speaker)                

                if y is not None:
                    yb[j] = y[speaker_idx]
                spect = extract(X[speaker_idx, 0], segment_size)
                seg_idx = randint(0, spect.shape[1] - segment_size)
                Xb[j, 0] = spect[:, seg_idx:seg_idx + segment_size]
            yield Xb.reshape(bs, segment_size, spectrogram_height), transformy(yb, bs, speakers)


# '''creates the a batch for LSTM networks, with Pairwise Labels, 
# for use with core.pairwise_kl_divergence_full_labels'''


# def batch_generator_lstm_v2(X, y, batch_size=100, segment_size=15):
#     segments = X.shape[0]
#     bs = batch_size
#     speakers = np.amax(y) + 1
#     # build as much batches as fit into the training set
#     while 1:
#         for i in range((segments + bs - 1) // bs):
#             Xb = np.zeros((bs, 1, settings.FREQ_ELEMENTS, segment_size), dtype=np.float32)
#             yb = np.zeros(bs, dtype=np.int32)
#             # here one batch is generated
#             for j in range(0, bs):
#                 speaker_idx = randint(0, len(X) - 1)
#                 if y is not None:
#                     yb[j] = y[speaker_idx]
#                 spect = extract(X[speaker_idx, 0], segment_size)
#                 seg_idx = randint(0, spect.shape[1] - segment_size)
#                 Xb[j, 0] = spect[:, seg_idx:seg_idx + segment_size]
#             yield Xb.reshape(bs, segment_size, settings.FREQ_ELEMENTS), create_pairs(yb)


def transformy(y, batch_size, nb_classes):
    yn = np.zeros((batch_size, int(nb_classes)))
    k = 0

    for v in y:
        yn[k][v] = 1
        k += 1
    return yn


def create_pairs(l):
    pair_list = []
    for i in range(len(l)):
        j = i + 1
        for j in range(i + 1, len(l)):
            print(j)
            if (l[i] == l[j]):
                pair_list.append((int(i), int(j), 1))
            else:
                pair_list.append((int(i), int(j), 0))
    return np.vstack(pair_list)


# Extracts the provided amount of samples from a Sentence
# The Samples are randomly chosen for each sentence.
# Example if there are 8 Sentences for 10 speakers each and the amount of samples is 3
# The function will return a numpy array Xb with shape 240, 1, 128, 100 and a Numpy array yb with shape 240
# this is for use in Keras model.fit function (does not yield good Training results)
def createData(X, y, samples, segment_size=15):
    segments = X.shape[0]
    idx = 0
    Xb = np.zeros((segments * samples, 1, settings.FREQ_ELEMENTS, segment_size), dtype=np.float32)
    yb = np.zeros(segments * samples, dtype=np.int32)
    for i in range(segments):
        # here one batch is generated
        for j in range(0, samples):
            speaker_idx = y[i]
            yb[idx] = speaker_idx
            spect = extract(X[i, 0], segment_size)
            seg_idx = randint(0, spect.shape[1] - segment_size)
            Xb[idx, 0] = spect[:, seg_idx:seg_idx + segment_size]
            idx += 1
    shuffle_data(Xb, yb)
    return Xb, yb


# lehl@2019-12-02: Batch generator using h5py dataset and speaker list
# 
# Params:
# statistics        Dict containing information for which ids can be used per speaker
# batch_type        'train' or 'val', for statistics access
# dataset           H5py handle of the dataset file
# 
def batch_generator_h5(statistics_fun, batch_type, dataset_fun, batch_size=100, segment_size=40):
    # Calculates the amount of indices used for the given :batch_type across all
    # speakers, such that this equals to the amount of spectrograms available for
    # training
    #
    # TODO: use method from bilstm layer
    #
    num_segments =  sum(map(lambda x: x.shape[0], (map(lambda x: statistics_fun()[x][batch_type], statistics_fun()))))
    speakers = np.array(list(statistics_fun().keys()))

    num_speakers = speakers.shape[0]

    # build as many batches as the training set is big
    # 
    while 1:
        for i in range((num_segments // batch_size) + 1):
            Xb = np.zeros((batch_size, 1, settings.FREQ_ELEMENTS, segment_size), dtype=np.float32)
            yb = np.zeros(batch_size, dtype=np.int32)

            # batch_start = time.time()

            for j in range(0, batch_size):
                # DEBUG CODE
                # import code; code.interact(local=dict(globals(), **locals()))
                # [batch_size, segment_size, batch_type, num_speakers, num_segments]

                speaker_index = randint(0, num_speakers - 1)
                speaker_name = speakers[speaker_index]

                # Extract Spectrogram
                # Choose from all the utterances of the given speaker randomly
                # 
                utterance_index = np.random.choice(statistics_fun()[speaker_name][batch_type])
                full_spect = dataset_fun()['data/' + speaker_name][utterance_index]
                
                # TODO: lehl@2019-12-03: Is the reshape ordering correct?
                # 
                spect = full_spect.reshape(settings.FREQ_ELEMENTS, full_spect.shape[0] // settings.FREQ_ELEMENTS)

                # Extract random :segment_size long part of the spectrogram
                # 
                seg_idx = randint(0, spect.shape[1] - segment_size)
                Xb[j, 0] = spect[:, seg_idx:seg_idx + segment_size]

                # Set label
                # 
                yb[j] = speaker_index
                
            # batch_end = time.time()
            # print("batch_generator_h5 single batch {} took {}".format(i, batch_end - batch_start))

            yield Xb.reshape(batch_size, segment_size, settings.FREQ_ELEMENTS), np.eye(num_speakers)[yb]
