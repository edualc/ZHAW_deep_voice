"""

"""
from common.extrapolation.speaker_factory import create_all_speakers

import h5py
import librosa
import numpy as np
import os
import pickle
from tqdm import tqdm

TARGET_SAMPLING_RATE = 16000

def setup_suite(dataset):
    """
    Can be called whenever the project must be setup on a new machine. It automatically
    generates all not yet generated speaker pickles in the right place.
    """
    for speaker in create_all_speakers(dataset):
        if speaker.is_pickle_saved():
            print('{} already exists.'.format(speaker.output_name))
        else:
            speaker.safe_to_pickle()

# NEW Setup using DeepVoiceDataset
# (see common/data/dataset.py)
# 
# lehl@2019-12-23:
# This preprocessing setup is taken from Daniel Neururer (neurudan@students.zhaw.ch)
# and used for comparability between our codebases/experiments.
# ==================================================================================
# 
# Definition of the datasets that are being setup.
# Expects list of lists with:
# - the dict generation method
# - the path to the data
# - the prefix for the finished h5py file
# 
def _dataset_to_setup():
    return [
        # [prepare_timit_dict, '/mnt/all1/timit/', 'timit_'],
        # [prepare_vox1_dict, '/mnt/all1/voxceleb1/dev/', 'vox1_dev_'],
        # [prepare_vox1_dict, '/mnt/all1/voxceleb1/test/', 'vox1_test_'],
        # [prepare_vox2_dict, '/mnt/all1/voxceleb2/complete/dev/', 'vox2_dev_'],
        # [prepare_vox2_dict, '/mnt/all1/voxceleb2/test/', 'vox2_test_'],
        # [prepare_vox2_dict, '/mnt/all1/voxceleb2/test/', 'musan_']
        # [prepare_zeros_dict, '/mnt/all1/experiments/datasets/', 'debug_zeros_'],
        # [prepare_overfit_dict, '/mnt/all1/experiments/datasets/', 'debug_overfit_']
    ]

# Returns the different dataset types that are generated. Originally only
# Mel Spectrograms are used, but if different versions of the raw dataset
# are needed, this is the place to add the conversion.
# 
# Entries are lists with:
# [0]:  The function (expecting the extracted data from librosa using i.e. librosa.load())
#         that returns the extracted and converted utterance
# [1]:  The naming suffix for this dataset version
# [2]:  The dtype für saving it in the h5py file
#
def _dataset_function_types():
    return [
        # [_original, 'original.h5', h5py.special_dtype(vlen=np.dtype('float32'))],
        [_mel_spectrogram, 'mel.h5', h5py.special_dtype(vlen=np.dtype('float32'))]
    ]

def setup_datasets():
    for [dict_function, base_path, dataset_base_name] in _dataset_to_setup():
        destination_file_path = base_path + dataset_base_name

        # File paths for progress files, that are used in case the process aborts
        # 
        full_struct_file_path = base_path + 'full_structure.p'
        progress_file_path = base_path + 'progress.p'
        working_dict = None

        # Check if structure has already been stored or generated it using
        # the dataset dict function is needed
        # 
        if not os.path.isfile(full_struct_file_path):
            working_dict = dict_function(base_path)
            pickle.dump(working_dict, open(full_struct_file_path, 'wb'))
        else:
            working_dict = pickle.load(open(full_struct_file_path, 'rb'))

        if os.path.isfile(progress_file_path):
            progress_dict = pickle.load(open(progress_file_path, 'rb'))

            # Check if the process has to be resumed or the dataset
            # was extracted (success-)fully
            # 
            if len(progress_dict.keys()) > 0:
                create_h5_file(destination_file_path, working_dict, progress_file_path, dataset_base_name)
            else:
                print("The '{}' dataset has already been fully extracted.".format(dataset_base_name))
            
        else:
            create_h5_file(destination_file_path, working_dict, progress_file_path, dataset_base_name)

def create_h5_file(h5_path, audio_dict, progress_file, base_name):
    print("Extracting {} corpus...".format(base_name))

    # Setup h5py File with the correct hierarchical structure
    # 
    if not os.path.isfile(progress_file):
        for _, name, data_type in _dataset_function_types():
            with h5py.File(h5_path + name, 'w') as f:

                data = f.create_group('data')
                audio_names = f.create_group('audio_names')
                statistics = f.create_group('statistics')

                for speaker in audio_dict:
                    shape = (len(audio_dict[speaker]), )
                    data.create_dataset(speaker, shape, dtype=data_type)
                    audio_names.create_dataset(speaker, shape, dtype=h5py.string_dtype(encoding='utf-8'))
                    statistics.create_dataset(speaker, shape, dtype='long')

    else:
        audio_dict = pickle.load(open(progress_file, 'rb'))

    # Evaluate Data Statistics
    # 
    total_files = 0
    total_speakers = 0

    for speaker in audio_dict:
        total_speakers += 1
        total_files += len(audio_dict[speaker])

    # Extract Speaker Data
    # 
    speaker_progress_bar = tqdm(total=total_speakers, desc='speaker extraction', ncols=100, ascii=True)
    total_progress_bar = tqdm(total=total_files, desc='audio_extraction', ncols=100, ascii=True)

    progress_dict = audio_dict.copy()

    for speaker in audio_dict:
        for i, [audio_file, audio_name] in enumerate(audio_dict[speaker]):
            x, fs = librosa.core.load(audio_file, sr=TARGET_SAMPLING_RATE)

            for audio_function, name, _ in _dataset_function_types():
                with h5py.File(h5_path + name, 'a') as f:
                    x_new = audio_function(x, fs)
                    length = len(x_new)

                    x_new = x_new.reshape((np.prod(x_new.shape)))

                    f['data/'+speaker][i] = x_new
                    f['statistics/'+speaker][i] = length
                    f['audio_names/'+speaker][i] = audio_name

            total_progress_bar.update(1)

        progress_dict.pop(speaker, None)
        pickle.dump(progress_dict, open(progress_file, 'wb'))
        speaker_progress_bar.update(1)

    speaker_progress_bar.close()
    total_progress_bar.close()
    print("{} extraction finished!".format(base_name))

# Conversion method for the identity function (keeping the original wave form data as-is)
#
def _original(x, fs=TARGET_SAMPLING_RATE):
    return x

# Conversion method for mel spectrograms, in accordance with previous ZHAW papers
# 
def _mel_spectrogram(x, fs=TARGET_SAMPLING_RATE):
    nperseg = int(10 * fs / 1000)
    mel_spectrogram = librosa.feature.melspectrogram(y=x, sr=fs, n_fft=1024, hop_length=nperseg)
    mel_spectrogram = np.log10(1 + 10000 * mel_spectrogram).T
    return mel_spectrogram

# Dataset dict generation functions
# (how to find all files in the dataset provided)
# ==================================================================================
# 

def prepare_timit_dict(dataset_path):
    extension = '_RIFF.WAV'
    data_dict = dict()

    if dataset_path[-1] != '/':
        dataset_path += '/'

    for sub_folder in ['TRAIN/', 'TEST/']:
        for subset in os.listdir(dataset_path + sub_folder):
            for speaker in os.listdir(timit_path+sub_folder+subset):
                audio_files = list()

                for audio_file in os.listdir(timit_path+sub_folder+subset+'/'+speaker):
                    if extension in audio_file:
                        audio_files.append([timit_path+sub_folder+subset+'/'+speaker+'/'+audio_file, audio_file])

                if speaker not in data_dict:
                    data_dict[speaker] = audio_files
                else:
                    data_dict[speaker].extend(audio_files)

    return data_dict

def prepare_vox1_dict(dataset_path):
    extension = '.wav'
    data_dict = dict()

    if dataset_path[-1] != '/':
        dataset_path += '/'

    sub_folder = 'wav/'

    for speaker in tqdm(os.listdir(dataset_path + sub_folder), ncols=100, ascii=True, desc='Vox1 reading ' + sub_folder):
        audio_files = list()

        for video in os.listdir(dataset_path + sub_folder + speaker):
            for audio in os.listdir(dataset_path + sub_folder + speaker + '/' + video):

                if extension in audio:
                    audio_files.append([dataset_path + sub_folder + speaker + '/' + video + '/' + audio, video + '/' + audio])

        data_dict[speaker] = audio_files

    return data_dict

def prepare_vox2_dict(dataset_path):
    extension = '.m4a'
    data_dict = dict()

    if dataset_path[-1] != '/':
        dataset_path += '/'

    sub_folder = 'aac/'

    for speaker in tqdm(os.listdir(dataset_path + sub_folder), ncols=100, ascii=True, desc='Vox2 reading ' + sub_folder):
        audio_files = list()

        for video in os.listdir(dataset_path + sub_folder + speaker):
            for audio in os.listdir(dataset_path + sub_folder + speaker + '/' + video):

                if extension in audio:
                    audio_files.append([dataset_path + sub_folder + speaker + '/' + video + '/' + audio, video + '/' + audio])

        data_dict[speaker] = audio_files

    return data_dict
