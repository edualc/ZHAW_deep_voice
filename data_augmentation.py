import librosa
import numpy as np
import os
from tqdm import tqdm
import h5py

musan_path = '/mnt/all1/experiments/datasets/musan'
datasets_path = '/mnt/all1/experiments/datasets'
vox2_sample_path = '/mnt/all1/voxceleb2/complete/dev/aac/id00012/_raOc3-IRsw/00110.m4a'

TARGET_SAMPLING_RATE = 16000

def _mel_spectrogram(x, fs=TARGET_SAMPLING_RATE):
    nperseg = int(10 * fs / 1000)
    mel_spectrogram = librosa.feature.melspectrogram(y=x, sr=fs, n_fft=1024, hop_length=nperseg)
    mel_spectrogram = np.log10(1 + 10000 * mel_spectrogram).T
    return mel_spectrogram

# Taken from Scipy history
# see https://stackoverflow.com/questions/51413068/calculate-signal-to-noise-ratio-in-python-scipy-version-1-1
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def musan_dict():
    extension = '.wav'
    stats = dict()

    for category in tqdm(os.listdir(musan_path), ncols=100, ascii=True, desc='Reading MUSAN...'):
        audio_files = list()
        category_path = musan_path + '/' + category

        if os.path.isdir(category_path):
            for subfolder in os.listdir(category_path):

                if os.path.isdir(category_path + '/' + subfolder):
                    for file in os.listdir(category_path + '/' + subfolder):

                        if extension in file:
                            audio_files.append(category_path + '/' + subfolder + '/' + file)

        if category != 'README':
            stats[category] = audio_files
    return stats

d = musan_dict()

# for cat in d.keys():
#     print(cat, len(d[cat]))

# with h5py.File(datasets_path + '/augmentation.h5', 'wb') as f:
#     for category in d.keys():
#         f.create_group(category)

y = d['noise'][0]
x,fs = librosa.core.load(y,sr=TARGET_SAMPLING_RATE)
x2 = _mel_spectrogram(x)
x3 = librosa.core.power_to_db(x2.T)

import code; code.interact(local=dict(globals(), **locals()))

