import pickle
import numpy as np
import h5py
import math

import keras
from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.layers.wrappers import Bidirectional
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import LearningRateScheduler, EarlyStopping
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

    def initialize_wandb(self):
        self.wandb_run = wandb.init(
            group=config.get('wandb','group'),
            project=config.get('wandb','project_name')
        )
        
        wandb.config.update({
            'n_hidden1': self.config.getint('pairwise_lstm', 'n_hidden1'),
            'n_hidden2': self.config.getint('pairwise_lstm', 'n_hidden2'),
            'n_dense1': self.config.getint('pairwise_lstm', 'n_dense1'),
            'n_dense2': self.config.getint('pairwise_lstm', 'n_dense2'),
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

    def create_net(self):
        from keras import backend as K
        list_of_gpus_available = K.tensorflow_backend._get_available_gpus()

        if len(list_of_gpus_available) > 0:
            gpu_model = self.create_net__gpu_component()
            gpu_model, gpu_loss, gpu_adam = self.create_net__classification_component(gpu_model)
            
            gpu_model.compile(loss=gpu_loss, optimizer=gpu_adam, metrics=['accuracy'])
            gpu_model.summary()

            # self.create_net__save_model(gpu_model, 'gpu')
            return gpu_model

        else:
            cpu_model = self.create_net__cpu_component()
            cpu_model, cpu_loss, cpu_adam = self.create_net__classification_component(cpu_model)
            
            cpu_model.compile(loss=cpu_loss, optimizer=cpu_adam, metrics=['accuracy'])
            cpu_model.summary()

            # self.create_net__save_model(cpu_model, 'cpu')
            return cpu_model

    def create_net__classification_component(self, model):
        model.add(Dense(self.config.getint('pairwise_lstm', 'n_dense1')))
        model.add(Dropout(0.25))
        
        model.add(Dense(self.config.getint('pairwise_lstm', 'n_dense2')))

        # This adds the final (Dense) layer
        # 
        add_final_layers(model, self.config)

        loss_function = get_loss(self.config)

        adam = keras.optimizers.Adam(
            lr=self.config.getfloat('pairwise_lstm', 'adam_lr'),
            beta_1=self.config.getfloat('pairwise_lstm', 'adam_beta_1'),
            beta_2=self.config.getfloat('pairwise_lstm', 'adam_beta_2'),
            epsilon=self.config.getfloat('pairwise_lstm', 'adam_epsilon'),
            decay=self.config.getfloat('pairwise_lstm', 'adam_decay')
        )

        return model, loss_function, adam

    def create_net__cpu_component(self):
        self.logger.info("NETWORK IS USING GPU!")
        model = Sequential()

        model.add(Bidirectional(CuDNNLSTM(self.config.getint('pairwise_lstm', 'n_hidden1'), return_sequences=True), input_shape=self.input))
        model.add(Dropout(0.50))
        model.add(Bidirectional(CuDNNLSTM(self.config.getint('pairwise_lstm', 'n_hidden2'))))

        return model

    def create_net__cpu_component(self):
        self.logger.info("NETWORK IS USING CPU!")
        model = Sequential()

        # needs activation/recurrent_activation defined with
        # CuDNNLSTM defaults such that the model can be loaded with compatible weights
        # 
        model.add(Bidirectional(LSTM(self.config.getint('pairwise_lstm', 'n_hidden1'), activation='tanh',recurrent_activation='sigmoid', return_sequences=True), input_shape=self.input))
        model.add(Dropout(0.50))
        model.add(Bidirectional(LSTM(self.config.getint('pairwise_lstm', 'n_hidden2'), activation='tanh',recurrent_activation='sigmoid')))

        return model

    def create_net__save_model(self, model, suffix):
        model_file = get_experiment_nets(self.network_name + '__' + suffix + '_model.json')

        with open(model_file, 'w') as json_file:
            json_file.write(model.to_json())

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
        callbacks.append(ActiveLearningUncertaintyCallback(self.dataset, self.config, self.logger, self.segment_size, self.spectrogram_height))
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
        batch_size = self.config.getint('train','batch_size')
        train_steps = self.dataset.get_train_num_segments('train') // batch_size + 1
        val_steps = self.dataset.get_train_num_segments('val') // batch_size + 1
        print("Train Steps:",train_steps,"\tVal Steps:",val_steps)

        # import code; code.interact(local=dict(globals(), **locals()))

        # Use multithreaded data generator
        # 
        print("Setting up ParallelTrainingDataGenerator...", end='')
        tg = ParallelTrainingDataGenerator(batch_size=batch_size, segment_size=self.segment_size,
            spectrogram_height=self.spectrogram_height, config=self.config, dataset=self.dataset)
        train_gen = tg.get_generator()
        print('done')

        print("Setting up ParallelValidationDataGenerator...", end='')
        vg = ParallelValidationDataGenerator(batch_size=batch_size, segment_size=self.segment_size,
            spectrogram_height=self.spectrogram_height, config=self.config, dataset=self.dataset)
        val_gen = vg.get_generator()
        print('done')

        # Start training using the generators defined above
        # 
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

        self.initialize_wandb()

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
