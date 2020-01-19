import h5py
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

SPEAKER_ID = 'id09262'
SPEAKER_SAMPLE_ID = 4

# n_fft = 1024
# window_length = n_fft = 1024
# hop_length = nperseg = int(10 * sample_rate / 1000)
# 
# librosa.core.stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, dtype=<class 'numpy.complex64'>, pad_mode='reflect')
# librosa.feature.melspectrogram(y=None, sr=22050, S=None, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='reflect', power=2.0, **kwargs)
# 
# frame_length = window_length = 1024
# frame_step = hop_length = nperseg = int(10 * sample_rate / 1000)
# fft_length = 
# tf.signal.stft(signals, frame_length, frame_step, fft_length=None, window_fn=tf.signal.hann_window, pad_end=False, name=None)

SAMPLE_RATE = 16000
N_FFT = 1024
WINDOW_LENGTH = 1024
HOP_LENGTH = int(10 * SAMPLE_RATE / 1000)
SPECTROGRAM_HEIGHT = 128

LOWER_EDGE_HERTZ = 1.0
UPPER_EDGE_HERTZ = 8000.0

def save_spect(S, name):
    print("Saving spectrogram [{}] for {}".format(S.shape, name))
    librosa.display.specshow(librosa.power_to_db(S), x_axis='time', y_axis='mel', fmax=8000, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    plt.colorbar(format='%+2.0f dB')

    file_name = 'vox2_' + str(SPEAKER_ID) + '_' + str(SPEAKER_SAMPLE_ID) + name
    plt.savefig(file_name)
    plt.close()

def normalize_spect(S):
    mu = np.mean(S, 0, keepdims=True)
    stdev = np.std(S, 0, keepdims=True)
    return (S - mu) / (stdev + 1e-5)

def data():
    return h5py.File('/mnt/all1/voxceleb2/complete/dev/vox2_dev_mel.h5', 'r')

def full_path():
    return '/mnt/all1/voxceleb2/complete/dev/aac/' + str(SPEAKER_ID) + '/' + data()['audio_names/' + SPEAKER_ID][SPEAKER_SAMPLE_ID]

def deepvoice():
    # DeepVoice Processing (from .h5 mel)
    # ================================
    #
    full_spect = data()['data/' + SPEAKER_ID][SPEAKER_SAMPLE_ID]
    spect = full_spect.reshape((full_spect.shape[0] // SPECTROGRAM_HEIGHT, SPECTROGRAM_HEIGHT))
    save_spect(librosa.power_to_db(spect.T), '__deepvoice__melspect.png')
    
    norm_spect = normalize_spect(spect)
    save_spect(librosa.power_to_db(norm_spect.T), '__deepvoice__melspect_normalized.png')
    # return librosa.power_to_db(norm_spect.T)

    tf_mfcc = tf.signal.mfccs_from_log_mel_spectrograms(norm_spect)[...,:24]
    save_spect(tf_mfcc.eval(session=tf.Session()).T, '__deepvoice__mfcc.png')
    # return tf_mfcc.eval(session=tf.Session()).T

def deepvoice_from_file():
    # DeepVoice Processing (from audiofile)
    # ================================
    #
    x, _sr = librosa.core.load(full_path(), sr=SAMPLE_RATE)
    mel_dv = librosa.feature.melspectrogram(y=x, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    mel_dv_logged = np.log10(1 + 10000 * mel_dv)
    save_spect(librosa.power_to_db(mel_dv_logged), '__dv__melspect.png')

    mel_dv_logged_norm = normalize_spect(mel_dv_logged.T).T
    save_spect(librosa.power_to_db(mel_dv_logged_norm), '__dv__melspect_normalized.png')

    tf_dv_mfcc = tf.signal.mfccs_from_log_mel_spectrograms(mel_dv_logged_norm.T)[...,:24]
    save_spect(tf_dv_mfcc.eval(session=tf.Session()).T, '__dv__mfcc.png')

def autodl():
    # AutoDL Processing (from audiofile)
    # ================================
    #
    pcm, _sr = librosa.core.load(full_path(), sr=SAMPLE_RATE)

    stfts = tf.signal.stft(pcm, frame_length=WINDOW_LENGTH, frame_step=HOP_LENGTH, fft_length=N_FFT)
    spectrograms = tf.abs(stfts)
    save_spect(librosa.power_to_db(spectrograms.eval(session=tf.Session()).T), '__autodl__spect.png')

    num_spectrogram_bins = stfts.shape[-1].value
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(SPECTROGRAM_HEIGHT, num_spectrogram_bins, SAMPLE_RATE, LOWER_EDGE_HERTZ, UPPER_EDGE_HERTZ)

    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    mel_autodl = librosa.power_to_db(mel_spectrograms.eval(session=tf.Session()).T)
    save_spect(mel_autodl, '__autodl__melspect.png')

    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    save_spect(librosa.power_to_db(log_mel_spectrograms.eval(session=tf.Session()).T), '__autodl__log_melspect.png')

    log_mel_spectrograms_normalized = normalize_spect(log_mel_spectrograms.eval(session=tf.Session()))
    save_spect(librosa.power_to_db(log_mel_spectrograms_normalized.T), '__autodl__log_melspect_normalized.png')
    # return librosa.power_to_db(log_mel_spectrograms_normalized.T)
    
    tf_autodl_mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms_normalized)[...,:24]
    save_spect(tf_autodl_mfcc.eval(session=tf.Session()).T, '__autodl__mfcc.png')
    # return tf_autodl_mfcc.eval(session=tf.Session()).T



    # tmp = mel_spectrograms.eval(session=tf.Session()).T
    # mel_autodl_dyn = np.log10(1 + 10000 * tmp)
    # save_spect(librosa.power_to_db(mel_autodl_dyn), '__autodl__melspect2.png')

    # # CONVERSION TO dB
    # mel_autodl_logged = 10 * np.log10(mel_spectrograms.eval(session=tf.Session()).T)

    # # DYNAMIC RANGE COMPRESSION:
    # # mel_autodl_logged = np.log10(1 + 10000 * mel_spectrograms.eval(session=tf.Session()).T)
    # save_spect(mel_autodl_logged, 'vox2_' + str(SPEAKER_ID) + '_' + str(SPEAKER_SAMPLE_ID) + '__autodl__melspect_log.png')

    # # OFFICIAL LOG MEL SPECTROGRAMS (TF)
    # mel_autodl_logged_tf = librosa.power_to_db(tf.math.log(mel_spectrograms + 1e-6).eval(session=tf.Session()).T)
    # save_spect(mel_autodl_logged_tf, 'vox2_' + str(SPEAKER_ID) + '_' + str(SPEAKER_SAMPLE_ID) + '__autodl__melspect_log_tf.png')

    # mel_autodl_normalized = normalize_spect(mel_autodl_logged)
    # save_spect(mel_autodl_normalized, 'vox2_' + str(SPEAKER_ID) + '_' + str(SPEAKER_SAMPLE_ID) + '__autodl__melspect_normalized.png')

def main():
    # lehl = deepvoice()
    # lehl = lehl[:,:564]
    # amir = autodl()

    deepvoice()
    deepvoice_from_file()
    autodl()
    
    # import code; code.interact(local=dict(globals(), **locals()))

if __name__ == '__main__':
    main()
