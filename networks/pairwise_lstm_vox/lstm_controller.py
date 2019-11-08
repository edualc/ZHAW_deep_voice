"""
The controller to train and test the pairwise_lstm network
"""

import numpy as np
from keras.models import Model
from keras.models import load_model

from common.clustering.generate_embeddings import generate_embeddings
from common.network_controller import NetworkController
from common.utils import TimeCalculator
from common.utils.logger import *
from common.utils.paths import *
from .bilstm_2layer_dropout_plus_2dense import bilstm_2layer_dropout
from .core.data_gen import generate_test_data
from common.spectrogram.speaker_train_splitter import SpeakerTrainSplit
from networks.losses import get_custom_objects, get_loss
import common.utils.pickler as pickler


class LSTMVOX2Controller(NetworkController):
    def __init__(self, name, config, dev, best):
        super().__init__(name, config, dev) #"vox2_speakers_120_test_cluster"

        # Currently prepared speaker_lists have the following datasets:
        #
        # 'vox2_speakers_5994_dev_cluster', # _train suffix for train/test split, _cluster otherwise
        # 'vox2_speakers_5994_dev_600_base', # _train suffix for train/test split, _cluster otherwise
        # 'vox2_speakers_120_test_cluster', # _train suffix for train/test split, _cluster otherwise
        # 'vox2_speakers_10_test_cluster', # _train suffix for train/test split, _cluster otherwise
        #
        self.train_data = config.get('train', 'pickle')
        self.dense_factor = config.getint('pairwise_lstm_vox2', 'dense_factor')
        self.n_speakers = config.getint('train', 'n_speakers')
        self.epochs = config.getint('pairwise_lstm_vox2', 'num_epochs')
        self.epochs_before_active_learning = config.getint('pairwise_lstm_vox2', 'epochs_before_al')
        self.active_learning_rounds = config.getint('pairwise_lstm_vox2', 'al_rounds')
        self.seg_size = config.getint('pairwise_lstm_vox2', 'seg_size')
        self.out_layer = config.getint('pairwise_lstm_vox2', 'out_layer')
        self.vec_size = config.getint('pairwise_lstm_vox2', 'vec_size')
        self.best = best

    def get_validation_data(self):
        return get_speaker_pickle(self.val_data, ".h5")

    def train_network(self):
        bilstm_2layer_dropout(
            self.name,
            self.train_data,
            n_hidden1=256,
            n_hidden2=256,
            dense_factor=self.dense_factor,
            n_speakers=self.n_speakers,
            epochs=self.epochs,
            epochs_before_active_learning=self.epochs_before_active_learning,
            active_learning_rounds=self.active_learning_rounds,
            segment_size=self.seg_size,
            config=self.config
        )

    # Loads the validation dataset as '_cluster' and splits it for further use
    #
    def get_validation_datasets(self):
        train_test_splitter = SpeakerTrainSplit(0.2)
        X, speakers = load_and_prepare_data(self.get_validation_data(), self.seg_size)

        X_train, X_test, y_train, y_test = train_test_splitter(X, speakers)

        return X_train, y_train, X_test, y_test

    def get_embeddings(self):
        # Passed seg_size parameter is ignored
        # because it is already used during training and must stay equal

        logger = get_logger('lstm_vox', logging.INFO)
        logger.info('Run pairwise_lstm test')
        logger.info('out_layer -> ' + str(self.out_layer))
        logger.info('seg_size -> ' + str(self.seg_size))
        logger.info('vec_size -> ' + str(self.vec_size))

        # Load and prepare train/test data
        x_train, speakers_train, x_test, speakers_test = self.get_validation_datasets()

        # Prepare return values
        set_of_embeddings = []
        set_of_speakers = []
        speaker_numbers = []

        if self.best:
            file_regex = self.name + "*_best.h5"
        else:
            file_regex = self.name + "*.h5"

        checkpoints = list_all_files(get_experiment_nets(), file_regex)

        # Values out of the loop
        metrics = ['accuracy', 'categorical_accuracy', ]
        loss = get_loss(self.config)
        custom_objects = get_custom_objects(self.config)
        optimizer = 'rmsprop'
        set_of_total_times = []

        # Fill return values
        for checkpoint in checkpoints:
            logger.info('Running checkpoint: ' + checkpoint)

            # Check if checkpoint is already processed and stored in intermediate results
            checkpoint_result_pickle = get_results_intermediate_test(checkpoint)

            # Add out_layer to checkpoint name
            checkpoint_result_pickle = checkpoint_result_pickle.split('.')[0] + '__ol' + str(self.out_layer) + '.' + checkpoint_result_pickle.split('.')[1]

            if os.path.isfile(checkpoint_result_pickle):
                embeddings, speakers, num_embeddings = pickler.load(checkpoint_result_pickle)
            else:
                # Load and compile the trained network
                model_full = load_model(get_experiment_nets(checkpoint), custom_objects=custom_objects)
                model_full.compile(loss=loss, optimizer=optimizer, metrics=metrics)

                # Get a Model with the embedding layer as output and predict
                model_partial = Model(inputs=model_full.input, outputs=model_full.layers[self.out_layer].output)

                logger.info('running predict on test set')
                test_output = np.asarray(model_partial.predict(x_test))
                logger.info('running predict on train set')
                train_output = np.asarray(model_partial.predict(x_train))
                logger.info('test_output len -> ' + str(test_output.shape))
                logger.info('train_output len -> ' + str(train_output.shape))

                embeddings, speakers, num_embeddings = generate_embeddings(
                    [train_output, test_output], [speakers_train,
                    speakers_test], self.vec_size
                )

                pickler.save((embeddings, speakers, num_embeddings), checkpoint_result_pickle)

            # Fill the embeddings and speakers into the arrays
            set_of_embeddings.append(embeddings)
            set_of_speakers.append(speakers)
            speaker_numbers.append(num_embeddings)

            # Calculate the time per utterance
            time = TimeCalculator.calc_time_all_utterances([speakers_train, speakers_test], self.seg_size)
            set_of_total_times.append(time)

        # Add out_layer to checkpoint names
        checkpoints = list(map(lambda x: x.split('.')[0] + '__ol' + str(self.out_layer) + '.' + x.split('.')[1], checkpoints))
        print("checkpoints: {}".format(checkpoints))

        logger.info('Pairwise_lstm test done.')
        return checkpoints, set_of_embeddings, set_of_speakers, speaker_numbers, set_of_total_times


def load_and_prepare_data(data_path, segment_size):
    # Load and generate test data
    (X, y, _) = pickler.load_speaker_pickle_or_h5(data_path)
    X, speakers = generate_test_data(X, y, segment_size)

    # Reshape test data because it is an lstm
    return X.reshape(X.shape[0], X.shape[3], X.shape[2]), speakers
