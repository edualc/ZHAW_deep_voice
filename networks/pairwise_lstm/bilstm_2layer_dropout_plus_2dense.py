import pickle
import numpy as np
import h5py

import keras
from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.layers.wrappers import Bidirectional
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers
from math import ceil

from .core import data_gen as dg
# from .core.parallel_data_gen import ParallelDataGenerator
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

class bilstm_2layer_dropout(object):
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
            'l1_regularization': self.config.getfloat('pairwise_lstm', 'l1_regularization'),
            'l2_regularization': self.config.getfloat('pairwise_lstm', 'l2_regularization'),
            'batch_size': self.config.getint('train','batch_size')
        })

        self.run_network()

    def create_net(self):
        model = Sequential()

        from keras import backend as K
        list_of_gpus_available = K.tensorflow_backend._get_available_gpus()

        if len(list_of_gpus_available) > 0:
            self.logger.info("NETWORK IS USING GPU!")

            # GPUs are available
            # 
            model.add(Bidirectional(CuDNNLSTM(wandb.config.n_hidden1, return_sequences=True,
                kernel_regularizer=regularizers.l1_l2(l1=wandb.config.l1_regularization, l2=wandb.config.l2_regularization)), input_shape=self.input))
            model.add(Dropout(0.50))
            model.add(Bidirectional(CuDNNLSTM(wandb.config.n_hidden2,
                kernel_regularizer=regularizers.l1_l2(l1=wandb.config.l1_regularization, l2=wandb.config.l2_regularization))))
            
        else:
            self.logger.info("NETWORK IS USING CPU!")

            # running on CPU
            # 
            model.add(Bidirectional(LSTM(wandb.config.n_hidden1, return_sequences=True,
                kernel_regularizer=regularizers.l1_l2(l1=wandb.config.l1_regularization, l2=wandb.config.l2_regularization)), input_shape=self.input))
            model.add(Dropout(0.50))
            model.add(Bidirectional(LSTM(wandb.config.n_hidden2,
                kernel_regularizer=regularizers.l1_l2(l1=wandb.config.l1_regularization, l2=wandb.config.l2_regularization))))

        model.add(Dense(wandb.config.n_dense1))
        model.add(Dropout(0.25))
        
        model.add(Dense(wandb.config.n_dense2))

        # This adds the final (Dense) layer
        # 
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

    def create_callbacks(self):
        net_saver = keras.callbacks.ModelCheckpoint(
            get_experiment_nets(self.network_name + "_best.h5"),
            monitor='val_loss', verbose=1, save_best_only=True)
        
        net_checkpoint = ActiveLearningModelCheckpoint(
            get_experiment_nets(self.network_name + "_{epoch:05d}.h5"),
            period=100
        )
        
        callbacks = [net_saver, net_checkpoint]

        callbacks.append(EERCallback(self.dataset, self.config, self.logger, self.segment_size, self.spectrogram_height))
        callbacks.append(ActiveLearningEpochLogger(self.logger, self.epochs))
        callbacks.append(ActiveLearningUncertaintyCallback(self.dataset, self.config, self.logger, self.segment_size, self.spectrogram_height))
        callbacks.append(WandbCallback(save_model=False))
        # 
        # callbacks.append(keras.callbacks.CSVLogger(get_experiment_logs(self.network_name + '.csv')))
        # callbacks.append(PlotCallback(self.network_name))

        return callbacks

    def fit(self, model, callbacks, epochs_to_run):
        # Calculate the steps per epoch for training and validation
        # 
        train_steps = self.dataset.get_train_num_segments('train') // wandb.config.batch_size
        val_steps = self.dataset.get_test_num_segments('all') // wandb.config.batch_size
        print("Train Steps:",train_steps,"\tVal Steps:",val_steps)

        # Use multithreaded data generator
        # 
        tg = ParallelTrainingDataGenerator(batch_size=wandb.config.batch_size, segment_size=self.segment_size,
            spectrogram_height=self.spectrogram_height, config=self.config, dataset=self.dataset)
        train_gen = tg.get_generator()

        vg = ParallelValidationDataGenerator(batch_size=wandb.config.batch_size, segment_size=self.segment_size,
            spectrogram_height=self.spectrogram_height, config=self.config, dataset=self.dataset)
        val_gen = vg.get_generator()

        # # Use single_threaded data generator (legacy)
        # # 
        # train_gen = dg.batch_generator_h5('train', self.dataset, batch_size=100, segment_size=self.segment_size, spectrogram_height=self.spectrogram_height)
        # val_gen = dg.batch_generator_h5('val', self.dataset, batch_size=100, segment_size=self.segment_size, spectrogram_height=self.spectrogram_height)

        history = model.fit_generator(
            train_gen,
            steps_per_epoch=train_steps,
            epochs=epochs_to_run,
            callbacks=callbacks,
            validation_data=val_gen,
            validation_steps=val_steps,
            verbose=1
        )

        tg.terminate_queue()
        vg.terminate_queue()

    def run_network(self):
        # base keras network
        model = self.create_net()
        callbacks = self.create_callbacks()

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
