import pickle
import numpy as np
import h5py
import math

import tensorflow as tf
from tensorflow.keras.backend import expand_dims

import keras
from keras import backend, optimizers
from keras.activations import relu, softmax
from keras.models import Sequential
from keras.layers import Convolution2D, BatchNormalization, Activation, MaxPool2D, GlobalMaxPooling2D, Dense, Lambda
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.layers.wrappers import Bidirectional
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras import regularizers
from math import ceil

from .core import data_gen as dg
from .core.parallel_train_data_gen import ParallelTrainingDataGenerator
from .core.parallel_val_data_gen import ParallelValidationDataGenerator
from .core import plot_saver as ps
from .core.callbacks import PlotCallback, ActiveLearningModelCheckpoint, ActiveLearningEpochLogger, EERCallback, ActiveLearningUncertaintyCallback

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
    epochs: Number of Epochs to train the Network per ActiveLearningRound
    activeLearnerRounds: Number of learning rounds to requery the pool for new data
    segment_size: Segment size that is used as input 100 equals 1 second with current Spectrogram extraction
    spectrogram_height: size of the frequency Dimension of the Input Spectrogram

    Work of Gerber and Glinski.
'''

class vggish(object):
    def __init__(self, name, segment_size, spectrogram_height, config, dataset):
        self.network_name = name
        self.config = config
        self.dataset = dataset
        
        # Setup Logging
        self.logger = get_logger('pairwise_lstm', logging.INFO)
        self.logger.info(self.network_name)
        
        # Network configuration
        self.n_speakers = config.getint('train', 'n_speakers')
        self.epochs = config.getint('train', 'n_epochs')
        self.segment_size = segment_size
        self.spectrogram_height = spectrogram_height
        self.input = (self.segment_size, self.spectrogram_height)
        
        # Initialize WandB
        self.wandb_run = wandb.init(
            group=config.get('wandb','group'),
            project=config.get('wandb','project_name')
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
            'n_dense1': config.getint('pairwise_lstm', 'n_dense1'),
            'n_dense2': config.getint('pairwise_lstm', 'n_dense2'),
            'epochs': self.epochs,
            'epochs_before_al': self.epochs_before_active_learning,
            'segment_size': self.segment_size,
            'learning_rate': self.config.getfloat('pairwise_lstm', 'adam_lr'),
            'beta_1': self.config.getfloat('pairwise_lstm', 'adam_beta_1'),
            'beta_2': self.config.getfloat('pairwise_lstm', 'adam_beta_2'),
            'epsilon': self.config.getfloat('pairwise_lstm', 'adam_epsilon'),
            'decay': self.config.getfloat('pairwise_lstm', 'adam_decay'),
            'batch_size': self.config.getint('train','batch_size'),
            'am_margin_cosface': self.config.getfloat('angular_loss','margin_cosface'),
            'am_margin_arcface': self.config.getfloat('angular_loss','margin_arcface'),
            'am_margin_sphereface': self.config.getfloat('angular_loss','margin_sphereface'),
            'am_scale': self.config.getint('angular_loss','scale')
        })

        self.run_network()



    def create_net(self):
        def compute_mfcc_from_melspectrogram(samples, input_shape, sample_rate=16000.0, number_of_features=24):
            return tf.signal.mfccs_from_log_mel_spectrograms(samples)[..., :number_of_features]

        from keras import backend as K
        K.set_image_dim_ordering('tf')

        # TODO: is this correct?
        # actual_input_shape = (None, 40, 128, 1)
        data_input_shape = (None, self.segment_size, self.spectrogram_height, 1)

        model = Sequential()
        # if preproc == "no":
        #     pass
        # elif preproc == "mfcc":
        #     model.add(Lambda(lambda x: compute_mfcc(x, input_shape)))
        # elif preproc == "melspec":
        #     model.add(Lambda(lambda x: compute_melspectrogram(x, input_shape)))
        # else:
        #     raise ValueError("Unknown preprocessing method: "+preproc)

        model.add(Lambda(lambda x: compute_mfcc_from_melspectrogram(x, data_input_shape)))

        # # expand last dim
        # model.add(Lambda(lambda x: expand_dims(x, -1)))

        filter_depth = 128

        # input_shape=(40, 128, 1), ? 
        model.add(Convolution2D(filter_depth//2, (4, 10), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPool2D())

        model.add(Convolution2D(filter_depth//2, (4, 10), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPool2D())

        model.add(Convolution2D(filter_depth, (4, 10), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPool2D())

        model.add(Convolution2D(filter_depth, (4, 10), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(GlobalMaxPooling2D())

        model.add(Dense(wandb.config.n_dense1))
        model.add(Dense(wandb.config.n_dense2))

        model.add(Dense(self.n_speakers, activation=softmax))

        adam = keras.optimizers.Adam(
            lr=wandb.config.learning_rate,
            beta_1=wandb.config.beta_1,
            beta_2=wandb.config.beta_2,
            epsilon=wandb.config.epsilon,
            decay=wandb.config.decay
        )

        model.build(input_shape=data_input_shape)
        
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
        
        model.summary()

        # import code; code.interact(local=dict(globals(), **locals()))

        return model

    def create_callbacks(self):
        net_saver = keras.callbacks.ModelCheckpoint(
            get_experiment_nets(self.network_name + "_best.h5"),
            monitor='val_loss', verbose=1, save_best_only=True)
        
        net_checkpoint = ActiveLearningModelCheckpoint(
            get_experiment_nets(self.network_name + "_{epoch:05d}.h5"),
            period=5
        )
        
        callbacks = [net_saver, net_checkpoint]

        callbacks.append(EERCallback(self.dataset, self.config, self.logger, self.segment_size, self.spectrogram_height))
        callbacks.append(ActiveLearningEpochLogger(self.logger, self.epochs))
        # callbacks.append(ActiveLearningUncertaintyCallback(self.dataset, self.config, self.logger, self.segment_size, self.spectrogram_height))
        callbacks.append(WandbCallback(save_model=False))
        # callbacks.append(LearningRateScheduler(self.lr_decay_fun(), verbose=1))
        callbacks.append(EarlyStopping(monitor='val_loss', patience=3, verbose=1))
        # callbacks.append(keras.callbacks.CSVLogger(get_experiment_logs(self.network_name + '.csv')))
        # callbacks.append(PlotCallback(self.network_name))

        return callbacks

    def lr_decay_fun(self):
        def lr_scheduler(epoch, lr):
            if epoch < 10:
                return self.config.getfloat('pairwise_lstm', 'adam_lr')
            else:
                return self.config.getfloat('pairwise_lstm', 'adam_lr') * math.exp(0.2 * (10 - epoch))

        return lr_scheduler

    def fit(self, model, callbacks, epochs_to_run):
        # Calculate the steps per epoch for training and validation
        # 
        train_steps = self.dataset.get_train_num_segments('train') // wandb.config.batch_size + 1
        val_steps = self.dataset.get_train_num_segments('val') // wandb.config.batch_size + 1
        print("Train Steps:",train_steps,"\tVal Steps:",val_steps)

        # Use multithreaded data generator
        # 
        tg = ParallelTrainingDataGenerator(batch_size=wandb.config.batch_size, segment_size=self.segment_size, spectrogram_height=self.spectrogram_height, config=self.config, dataset=self.dataset)
        train_gen = tg.get_generator()

        vg = ParallelValidationDataGenerator(batch_size=wandb.config.batch_size, segment_size=self.segment_size, spectrogram_height=self.spectrogram_height, config=self.config, dataset=self.dataset)
        val_gen = vg.get_generator()

        # Start training using the generators defined above
        # 
        history = model.fit_generator(train_gen,steps_per_epoch=train_steps,epochs=epochs_to_run,callbacks=callbacks,validation_data=val_gen,validation_steps=val_steps,verbose=1)

        tg.terminate_queue()
        vg.terminate_queue()

    def run_network(self):
        # base keras network
        model = self.create_net()
        callbacks = self.create_callbacks()

        # import code; code.interact(local=dict(globals(), **locals()))

        # # initial train
        # if self.epochs_before_active_learning > 0:
        self.fit(model, callbacks, self.epochs_before_active_learning)

        # # active learning
        # if self.config.getboolean('active_learning','enabled'):
        #     model = self.active_learner.perform_active_learning(
        #         active_learning_rounds=self.active_learning_rounds,
        #         epochs_trained= self.epochs_before_active_learning,
        #         model=model,
        #         callbacks=callbacks,
        #         network=self
        #     )

        self.logger.info("saving model")
        model.save(get_experiment_nets(self.network_name + ".h5"))
