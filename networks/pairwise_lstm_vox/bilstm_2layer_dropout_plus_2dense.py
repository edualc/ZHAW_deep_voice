import pickle
import numpy as np
import h5py
from tqdm import tqdm

import keras
from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.layers.wrappers import Bidirectional
from keras.wrappers.scikit_learn import KerasClassifier
from math import ceil

from .core import data_gen as dg
from .core import plot_saver as ps
from .core.callbacks import PlotCallback, ActiveLearningModelCheckpoint, ActiveLearningEpochLogger

import common.spectrogram.speaker_train_splitter as sts
from common.utils.logger import *
from common.utils.paths import *
from common.utils.pickler import load_speaker_pickle_or_h5

from common.active_learning import active_learner as al
from networks.losses import get_loss, add_final_layers

import wandb
from wandb.keras import WandbCallback

'''This Class Trains a Bidirectional LSTM with 2 Layers, and 2 Denselayer and a Dropout Layers
    Parameters:
    name: Network/experiment name to save artifacts
    training_data: name of the Training data file, expected to be train_xxxx.pickle
    n_hidden1: Units of the first LSTM Layer
    n_hidden2: Units of the second LSTM Layer
    dense_factor: Amount of output classes (Speakers in Trainingset)
    epochs: Number of Epochs to train the Network per ActiveLearningRound
    activeLearnerRounds: Number of learning rounds to requery the pool for new data
    segment_size: Segment size that is used as input 100 equals 1 second with current Spectrogram extraction
    frequency: size of the frequency Dimension of the Input Spectrogram

    Work of Gerber and Glinski.
'''


class bilstm_2layer_dropout(object):
    def __init__(self, name, segment_size, config, dataset):
        self.network_name = name
        self.config = config
        self.dataset = dataset
        
        # Setup Logging
        self.logger = get_logger('lstm_vox', logging.INFO)
        self.logger.info(self.network_name)
        
        # Network configuration
        self.n_speakers = config.getint('train', 'n_speakers')
        self.epochs = config.getint('train', 'n_epochs')
        self.segment_size = segment_size
        self.frequency = self.config.getint('pairwise_lstm','spectrogram_height')
        self.input = (self.segment_size, self.frequency)
        
        # Initialize WandB
        self.wandb_run = wandb.init(
            group=config.get('wandb','group'),
            project=config.get('wandb','project_name'),
            job_type=config.get('wandb','job_type')
        )

        # Initializes Active Learning if necessary
        if self.config.getboolean('active_learning','enabled'):
            self.epochs_before_active_learning = config.getint('active_learning', 'epochs_before_al')
            self.active_learning_rounds = config.getint('active_learning', 'al_rounds')
            self.active_learner = al.active_learner(
                logger=self.logger,
                config=self.config,
                segment_size=segment_size,
                spectrogram_height=config.getint('pairwise_lstm','spectrogram_height'),
                dataset=self.dataset,
                epochs=self.epochs,
                epochs_per_round=ceil((self.epochs - self.epochs_before_active_learning) / self.active_learning_rounds)
            )
        else:
            self.epochs_before_active_learning = self.epochs

        wandb.config.update({
            'n_hidden1': config.getint('pairwise_lstm', 'n_hidden1'),
            'n_hidden2': config.getint('pairwise_lstm', 'n_hidden2'),
            'dense_factor': config.getint('pairwise_lstm', 'dense_factor'),
            'epochs': self.epochs,
            'epochs_before_al': self.epochs_before_active_learning,
            'segment_size': self.segment_size,
            'learning_rate': self.config.getfloat('pairwise_lstm', 'adam_lr'),
            'beta_1': self.config.getfloat('pairwise_lstm', 'adam_beta_1'),
            'beta_2': self.config.getfloat('pairwise_lstm', 'adam_beta_2'),
            'epsilon': self.config.getfloat('pairwise_lstm', 'adam_epsilon'),
            'decay': self.config.getfloat('pairwise_lstm', 'adam_decay')
        })

        self.run_network()

    def create_net(self):
        model = Sequential()

        model.add(Bidirectional(LSTM(wandb.config.n_hidden1, return_sequences=True), input_shape=self.input))
        model.add(Dropout(0.50))
        model.add(Bidirectional(LSTM(wandb.config.n_hidden2)))

        model.add(Dense(wandb.config.dense_factor * 10))
        model.add(Dropout(0.25))
        model.add(Dense(wandb.config.dense_factor * 5))
        add_final_layers(model, self.config)

        loss_function = get_loss(self.config)

        adam = keras.optimizers.Adam(
            lr=wandb.config.learning_rate,
            beta_1=wandb.config.beta_1,
            beta_2=wandb.config.beta_2,
            epsilon=wandb.config.epsilon,
            decay=wandb.config.decay
        )

        model.compile(loss=loss_function, optimizer=adam, metrics=['accuracy'])
        model.summary()

        return model

    # def split_train_val_data(self, X, y):
    #     splitter = sts.SpeakerTrainSplit(0.2)
    #     X_t, X_v, y_t, y_v = splitter(X, y)
    #     return X_t, y_t, X_v, y_v

    def create_callbacks(self):
        csv_logger = keras.callbacks.CSVLogger(
            get_experiment_logs(self.network_name + '.csv'))
        info_logger = ActiveLearningEpochLogger(self.logger, self.epochs)
        net_saver = keras.callbacks.ModelCheckpoint(
            get_experiment_nets(self.network_name + "_best.h5"),
            monitor='val_loss', verbose=1, save_best_only=True)
        net_checkpoint = ActiveLearningModelCheckpoint(
            get_experiment_nets(self.network_name + "_{epoch:05d}.h5"),
            period=int(self.epochs / 3)
        )
        plot_callback_instance = PlotCallback(self.network_name)
        wandb_callback = WandbCallback(save_model=False)

        return [csv_logger, info_logger, net_saver, net_checkpoint, plot_callback_instance, wandb_callback]

    def fit(self, model, callbacks, epochs_to_run):
        train_gen = dg.batch_generator_h5('train', self.dataset, batch_size=100, segment_size=self.segment_size)
        val_gen = dg.batch_generator_h5('val', self.dataset, batch_size=100, segment_size=self.segment_size)

        # # Original Batch Generator
        # train_gen = dg.batch_generator_lstm(X_t, y_t, 100, segment_size=self.segment_size)
        # val_gen = dg.batch_generator_lstm(X_v, y_v, 100, segment_size=self.segment_size)

        # # Alternative Batch Generator
        # train_gen = dg.batch_generator_divergence_optimised(X_t, y_t, 100, segment_size=self.segment_size)
        # val_gen = dg.batch_generator_divergence_optimised(X_v, y_v, 100, segment_size=self.segment_size)

        history = model.fit_generator(
            train_gen,
            steps_per_epoch=10,
            epochs=epochs_to_run,
            callbacks=callbacks,
            validation_data=val_gen,
            validation_steps=2,
            class_weight=None,
            max_queue_size=10,
            workers=1,
            verbose=1,
            use_multiprocessing=False
        )

    def run_network(self):
        # base keras network
        model = self.create_net()
        callbacks = self.create_callbacks()

        # # initial train set
        # speaker_pickle = get_speaker_pickle(self.training_data, format='.h5')
        # X_pool, y_pool, _ = self._read_speaker_data(speaker_pickle)
        # X_t, y_t, X_v, y_v = self.split_train_val_data(X_pool, y_pool)

        # del X_pool
        # del y_pool

        # initial train
        if self.epochs_before_active_learning > 0:
            self.fit(model, callbacks, self.epochs_before_active_learning)

        # active learning
        if self.config.getboolean('active_learning','enabled'):
            model = self.active_learner.perform_active_learning(
                active_learning_rounds=self.active_learning_rounds,
                epochs_trained= self.epochs_before_active_learning,
                model=model,
                callbacks=callbacks,
                network=self
            )

        self.logger.info("saving model")
        model.save(get_experiment_nets(self.network_name + ".h5"))

    # # Reads the speaker data from a given speaker pickle or h5
    # # 
    # # Paramters:
    # # speaker_pickle: Location of speaker pickle/h5 file
    # #
    # # Returns: The unpacked file (Features, Labels, Speakerident List)
    # #
    # def _read_speaker_data(self, speaker_pickle):
    #     self.logger.info('create_train_data ' + speaker_pickle)
    #     (X, y, _) = load_speaker_pickle_or_h5(speaker_pickle)
    #     return (X, y, speaker_pickle)