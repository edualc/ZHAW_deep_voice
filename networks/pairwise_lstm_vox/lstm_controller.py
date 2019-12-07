"""
The controller to train and test the pairwise_lstm network
"""

import numpy as np
from keras.models import Model
from keras.models import load_model
import logging

from common.clustering.generate_embeddings import generate_embeddings
from common.network_controller import NetworkController
from common.utils import TimeCalculator
from common.utils.logger import *
from common.utils.paths import *
from common.data.dataset import DeepVoiceDataset
from .bilstm_2layer_dropout_plus_2dense import bilstm_2layer_dropout
from .core.data_gen import generate_test_data
from common.spectrogram.speaker_train_splitter import SpeakerTrainSplit
from networks.losses import get_custom_objects, get_loss
import common.utils.pickler as pickler


class LSTMVOX2Controller(NetworkController):
    def __init__(self, name, config, dev, best):
        super().__init__(name, config, dev)
        # self.train_data = config.get('train', 'pickle')

        self.seg_size = config.getint('pairwise_lstm', 'seg_size')
        self.out_layer = config.getint('pairwise_lstm', 'out_layer')
        self.vec_size = config.getint('pairwise_lstm', 'vec_size')
        
        self.best = best
        self.dataset = DeepVoiceDataset(self.config)

    # def get_validation_data(self):
    #     return get_speaker_pickle(self.val_data, ".h5")

    def train_network(self):
        bilstm_2layer_dropout(
            self.name,
            segment_size=self.seg_size,
            config=self.config,
            dataset=self.dataset
        )

    # # Loads the validation dataset as '_cluster' and splits it for further use
    # #
    # def get_validation_datasets(self):
    #     train_test_splitter = SpeakerTrainSplit(0.2)
    #     X, speakers = load_and_prepare_data(self.get_validation_data(), self.seg_size)

    #     X_train, X_test, y_train, y_test = train_test_splitter(X, speakers)

    #     return X_train, y_train, X_test, y_test

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
        set_of_utterance_embeddings = []

        if self.best:
            file_regex = self.name + ".*_best\.h5"
        else:
            file_regex = self.name + ".*\.h5"

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
                embeddings, speakers, num_embeddings, utterance_embeddings = pickler.load(checkpoint_result_pickle)
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

                embeddings, speakers, num_embeddings, utterance_embeddings = generate_embeddings(
                    [train_output, test_output], [speakers_train,
                    speakers_test], self.vec_size
                )

                pickler.save((embeddings, speakers, num_embeddings, utterance_embeddings), checkpoint_result_pickle)

            # Fill the embeddings and speakers into the arrays
            set_of_embeddings.append(embeddings)
            set_of_speakers.append(speakers)
            speaker_numbers.append(num_embeddings)
            set_of_utterance_embeddings.append(utterance_embeddings)

            # Calculate the time per utterance
            time = TimeCalculator.calc_time_all_utterances([speakers_train, speakers_test], self.seg_size)
            set_of_total_times.append(time)

        # Add out_layer to checkpoint names
        checkpoints = list(map(lambda x: x.split('.')[0] + '__ol' + str(self.out_layer) + '.' + x.split('.')[1], checkpoints))
        print("checkpoints: {}".format(checkpoints))

        logger.info('Pairwise_lstm test done.')
        return checkpoints, set_of_embeddings, set_of_speakers, speaker_numbers, set_of_total_times, set_of_utterance_embeddings


# def load_and_prepare_data(data_path, segment_size):
#     # Load and generate test data
#     (X, y, _) = pickler.load_speaker_pickle_or_h5(data_path)
#     X, speakers = generate_test_data(X, y, segment_size)

#     # Reshape test data because it is an lstm
#     return X.reshape(X.shape[0], X.shape[3], X.shape[2]), speakers
