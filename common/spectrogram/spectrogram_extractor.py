"""
The SpectrogramExtractor crawls the base folder and all its sub folder for wav files that
correspond with the valid speakers and extracts the spectrogram's of those files.

Based on previous work of Gerber, Lukic and Vogt.
"""
import os

import common.spectrogram.spectrogram_converter as spectrogram_converter


class SpectrogramExtractor:

    def extract_speaker_data(self, X, y, speaker_files):
        """
        Extract spectrogram and speaker names from given folder.

        :param X: return Array that saves the mel spectrogram's
        :param y: return Array that saves the speaker numbers
        :param speaker_files: dict with speaker names as keys and audio files in an array as values
        :return: the filled X, y
        """

        global_idx = 0
        curr_speaker_num = 0
        max_speakers = len(speaker_files.keys())

        # Crawl the base and all sub folders
        for speaker in speaker_files.keys():
            speaker_uid = curr_speaker_num

            print('Extraction progress: %d/%d' % (curr_speaker_num+1, max_speakers))

            # Extract files
            for full_path in speaker_files[speaker]:
                extract_mel_spectrogram(full_path, X, y, global_idx, speaker_uid)
                global_idx += 1

            curr_speaker_num += 1

        return X[0:global_idx], y[0:global_idx]

def extract_mel_spectrogram(wav_path, X, y, index, speaker_uid):
    """
    Extracts the mel spectrogram into the X array and saves the speaker into y.

    :param wav_path: the path to the wav file
    :param X: return Array that saves the mel spectrogram
    :param y: return Array that saves the speaker numbers
    :param index: the index in X and y this is stored in
    :param speaker_uid: the speaker number of the current speaker (integer hash of his identifier)
    :return: a one (1) to increase the index
    """
    #print('processing ', wav_path)
    Sxx = spectrogram_converter.mel_spectrogram(wav_path)
    for i in range(Sxx.shape[0]):
        for j in range(Sxx.shape[1]):
            # In case the file is longer than the :max_audio_length defined in the speaker_factory.py,
            # we only use the spectrogram up to that cut off point at :max_audio_length
            if j >= X.shape[3]:
                continue
            X[index, 0, i, j] = Sxx[i, j]
    y[index] = speaker_uid

    # # TODO: lehl@2019-11-26
    # # preprocessing, subtract mean, divided by time-wise var
    # mu = np.mean(spec_mag, 0, keepdims=True)
    # std = np.std(spec_mag, 0, keepdims=True)

# Extracts the spectrogram and discards all padded data
def extract_spectrogram(spectrogram, segment_size, frequency_elements):
    zeros = 0

    for x in spectrogram[0]:
        if x == 0.0:
            zeros += 1
        else:
            zeros = 0

    while spectrogram.shape[1] - zeros < segment_size:
        zeros -= 1

    return spectrogram[0:frequency_elements, 0:spectrogram.shape[1] - zeros]
