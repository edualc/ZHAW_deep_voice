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
from keras.callbacks import LearningRateScheduler

from .core import data_gen as dg
from .core.callbacks import ActiveLearningModelCheckpoint, ActiveLearningEpochLogger, EERCallback, ActiveLearningUncertaintyCallback

import common.spectrogram.speaker_train_splitter as sts
from common.utils.logger import *
from common.utils.paths import *
from common.utils.pickler import load_speaker_pickle_or_h5

from common.active_learning import active_learner as al
from networks.losses import get_loss, add_final_layers

import wandb
from wandb.keras import WandbCallback

class resnet_network(object):
    def __init__(self, name, segment_size, spectrogram_height, config, dataset):
        self.network_name = name
        self.config = config
        self.dataset = dataset

        # Setup Logging
        self.logger = get_logger('resnet50', logging.INFO)
        self.logger.info(self.network_name)

        # Network configuration
        self.n_speakers = config.getint('train', 'n_speakers')
        self.epochs = config.getint('train', 'n_epochs')
        self.segment_size = segment_size
        self.spectrogram_height = spectrogram_height
        self.input = (self.segment_size, self.spectrogram_height, 1)
        self.decay_step = self.config.getint('resnet50','decay_step')
        self.decay_rate = self.config.getfloat('resnet50','decay_rate')

        # Initialize WandB
        self.wandb_run = wandb.init(
            group=config.get('wandb','group'),
            project=config.get('wandb','project_name')
        )

        if self.config.getboolean('active_learning','enabled'):
            self.epochs_before_active_learning = config.getint('active_learning', 'epochs_before_al')
            self.active_learning_rounds = config.getint('active_learning', 'al_rounds')
            self.active_learner = al.active_learner(
                logger=self.logger,
                config=self.config,
                segment_size=segment_size,
                spectrogram_height=config.getint('resnet50','spectrogram_height'),
                dataset=self.dataset,
                epochs=self.epochs,
                epochs_per_round=ceil((self.epochs - self.epochs_before_active_learning) / self.active_learning_rounds)
            )
        else:
            self.epochs_before_active_learning = self.epochs

        wandb.config.update({
            'epochs': self.epochs,
            'epochs_before_al': self.epochs_before_active_learning,
            'segment_size': self.segment_size,
            'learning_rate': self.config.getfloat('resnet50', 'adam_lr'),
            'beta_1': self.config.getfloat('resnet50', 'adam_beta_1'),
            'beta_2': self.config.getfloat('resnet50', 'adam_beta_2'),
            'epsilon': self.config.getfloat('resnet50', 'adam_epsilon'),
            'decay': self.config.getfloat('resnet50', 'adam_decay'),
            'decay_step': self.decay_step,
            'decay_rate': self.decay_rate
        })

        self.run_network()

    def create_net(self):
        from .vgg_thin import resnet_2D_v1
        import keras
        from keras.layers import Conv2D, AveragePooling2D, Reshape, Flatten, Dense
        from keras.models import Model, Sequential
        
        bottleneck_dim = 512
        l2_regularization = 0.01

        # Import Thin ResNet34
        # 
        inputs, x = resnet_2D_v1(self.input, mode='train')
        
        from keras import backend
        backend.set_image_data_format('channels_first')

        x_fc = Conv2D(bottleneck_dim, (7, 1),
            strides=(1, 1),
            activation='relu',
            kernel_initializer='orthogonal',
            use_bias=True, trainable=True,
            padding='same',
            kernel_regularizer=keras.regularizers.l2(l2_regularization),
            bias_regularizer=keras.regularizers.l2(l2_regularization),
            name='x_fc')(x)

        x = AveragePooling2D((1, 5), strides=(1, 1), name='avg_pool')(x)
        x = Flatten()(x)

        x = keras.layers.Dense(bottleneck_dim, activation='relu',
                               kernel_initializer='orthogonal',
                               use_bias=True, trainable=True,
                               kernel_regularizer=keras.regularizers.l2(l2_regularization),
                               bias_regularizer=keras.regularizers.l2(l2_regularization),
                               name='fc6')(x)

        dense_model = Sequential()
        add_final_layers(dense_model, self.config)

        x = dense_model(x)
        model = Model(inputs, x)

        adam = keras.optimizers.Adam(
            lr=wandb.config.learning_rate, # 0.0001 @ VGG
            beta_1=wandb.config.beta_1,
            beta_2=wandb.config.beta_2,
            epsilon=wandb.config.epsilon,
            decay=wandb.config.decay
        )

        loss_function = get_loss(self.config)
        model.compile(loss=loss_function, optimizer=adam, metrics=['accuracy'])
        model.summary()

        return model

    def create_callbacks(self):
        net_saver = keras.callbacks.ModelCheckpoint(
            get_experiment_nets(self.network_name + "_best.h5"),
            monitor='val_loss', verbose=1, save_best_only=True)
        
        net_checkpoint = ActiveLearningModelCheckpoint(
            get_experiment_nets(self.network_name + "_{epoch:05d}.h5"),
            period=int(self.epochs / 3)
        )
        
        callbacks = [net_saver, net_checkpoint]

        callbacks.append(EERCallback(self.dataset, self.config, self.logger, self.segment_size, self.spectrogram_height))
        callbacks.append(ActiveLearningEpochLogger(self.logger, self.epochs))
        callbacks.append(ActiveLearningUncertaintyCallback(self.dataset, self.config, self.logger, self.segment_size, self.spectrogram_height))
        callbacks.append(WandbCallback(save_model=False))
        callbacks.append(LearningRateScheduler(self.step_decay, verbose=1))

        return callbacks

    def step_decay(self, epoch, lr):
        if epoch > 0 and epoch % self.decay_step == 0:
            return lr * self.decay_rate
        else:
            return lr

    def fit(self, model, callbacks, epochs_to_run):
        train_gen = dg.batch_generator_h5('train', self.dataset, batch_size=100, segment_size=self.segment_size, spectrogram_height=self.spectrogram_height)
        val_gen = dg.batch_generator_h5('val', self.dataset, batch_size=100, segment_size=self.segment_size, spectrogram_height=self.spectrogram_height)

        history = model.fit_generator(
            train_gen,
            steps_per_epoch=10,
            epochs=epochs_to_run,
            callbacks=callbacks,
            validation_data=val_gen,
            validation_steps=2,
            verbose=1
        )

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
