import numpy as np
import random

import wandb
from tqdm import tqdm

from networks.pairwise_lstm.core import plot_saver as ps
from common.spectrogram import speaker_train_splitter as sts
from common.utils.pickler import load_speaker_pickle_or_h5
from common.utils.paths import *


'''This Class performs active learning rounds
    
    Parameters:
    # logger: The logger used to pass messages
    # config: The current run config
    # spectrogram_height: The height of the spectrogram during training
    #                     (input is segment_size x spectrogram_height)
    # segment_size: The segment length used for training
    # epochs: Number of epochs to train the network in total
    # epochs_per_round: Amount of epochs to be run per active learning round

    Work of Lauener and Lehmann.
'''

class active_learner(object):
    def __init__(self, logger, config, segment_size, spectrogram_height, dataset, epochs, epochs_per_round):
        self.logger = logger
        self.config = config
        self.dataset = dataset
        self.epochs_per_round = epochs_per_round
        self.epochs = epochs
        self.segment_size = segment_size
        self.spectrogram_height = spectrogram_height

        self.n_instances = config.getint('active_learning','n_instances')
        self.n_speakers_per_al_round = config.getint('active_learning','n_speakers_per_al_round')

        self.al_rounds = config.getint('active_learning', 'al_rounds')
        if self.al_rounds <= 0:
            raise ValueError("Active Learning needs to have more than 0 al_rounds set, was {}".format(self.al_rounds))
        
    # This methods performs the active learning rounds and needs the following parameters:
    # 
    # model: The current model that is used to train
    # network: The network that implements the :fit method
    # callbacks: The callbacks used by the network during the :run_network "pre active learning" training
    # epochs_trained: The amount of epochs already run
    # active_learning_rounds: How many rounds of active learning are performed
    # 
    # Returns: The model used in training
    # 
    def perform_active_learning(self, model, network, callbacks, epochs_trained, active_learning_rounds):

        for i in range(active_learning_rounds):
            self.logger.info("Active learning round " + str(i) + "/" + str(active_learning_rounds))

            # Check how many epochs are to be run
            # 
            if self.epochs >= (epochs_trained + self.epochs_per_round):
                epochs_to_run = self.epochs_per_round
            else:
                epochs_to_run = self.epochs - epochs_trained

            # if max epochs to train already reached before all
            # rounds processed we can end the training
            # 
            if epochs_to_run <= 0:
                self.logger.info("Max epoch of " + str(self.epochs) + " reached, end of training")
                break

            if epochs_trained > 0:
                # query for uncertainty based on pool and append to numpy X_t, X_v, ... arrays
                # 
                self.perform_round(model)

            network.fit(model, callbacks, epochs_to_run)
            epochs_trained += epochs_to_run

            # # TODO: Change ALR Plot?
            # ps.save_alr_shape_x_plot(network.network_name, [ X_t_shapes, X_v_shapes ])

        return model

    # This method performs ONE round of active learning by itself
    # 
    # The parameters are:
    # model: The model used for training
    # 
    # Returns: The new training and validation sets, with new smaples added from the speaker pools
    # 
    def perform_round(self, model):
        # do uncertainty sampling
        # ==========================================
        # 
        uncertainty_results = np.empty((0,3), dtype=np.float32)

        # TODO: Do we need to check if the speaker has files left (after excessive extractions)?
        # 
        # Iterate over a set of speakers and perform uncertainty sampling on their samples
        #
        for speaker in tqdm(random.sample(self.dataset.get_train_speaker_list(), self.n_speakers_per_al_round), ncols=100, desc='Performing uncertainty sampling...'):
            uncertainty_results = np.append(uncertainty_results, self.perform_uncertainty_sampling(model, speaker), axis=0)

        # enlarge training set
        # ==========================================
        # 
        if uncertainty_results.shape[0] > self.n_instances:
            # The only the worst :n_instances of utterances to add to the training
            # dataset, picking higher uncertainties
            # 
            utterances_to_add = np.sort(uncertainty_results, axis=0)[-self.n_instances:]
        else:
            utterances_to_add = uncertainty_results

        wandb.log({
            'al_utterances_added': utterances_to_add.shape[0]
        }, commit=False)

        self.dataset.update_active_learning_share(utterances_to_add)

    # Uncertainty sampling query strategy. Selects the least sure instances for labelling.
    #
    # Parameters:
    # model: The model for which the labels are to be queried.
    # X: The pool of samples to query from.
    # n_instances: Number of samples to be queried.
    # 
    # Returns: The indices of the instances from X chosen to be labelled;
    #
    def perform_uncertainty_sampling(self, model, speaker):
        al_indices = self.dataset.get_train_statistics()[speaker]['al']

        if al_indices.shape[0] == 0:
            return np.empty((0,3), dtype=np.float32)

        # Limit the number of utterances per speaker with an upper bound such that the imbalance is 
        # reduced during active learning probing
        # 
        # TODO: lehl@2019-12-12: Should this be a parameter?
        # 
        max_num_utterances_per_speaker = int(self.n_instances / 8)
        if al_indices.shape[0] > max_num_utterances_per_speaker:
            al_indices = np.sort(np.random.choice(al_indices, max_num_utterances_per_speaker, replace=False))

        spectrograms = self.dataset.get_train_file()['data/'+speaker][al_indices][:]

        Xb = np.zeros((spectrograms.shape[0], 1, self.spectrogram_height, self.segment_size))
        indices = np.zeros((spectrograms.shape[0]),dtype=np.int)

        for i in range(spectrograms.shape[0]):
            spect = spectrograms[i].reshape(spectrograms[i].shape[0] // self.spectrogram_height, self.spectrogram_height).T
            seg_idx = random.randint(0, spect.shape[1] - self.segment_size)

            Xb[i, 0] = spect[:, seg_idx:seg_idx + self.segment_size]
            indices[i] = al_indices[i]

        ys = model.predict(Xb.reshape(spectrograms.shape[0], self.segment_size, self.spectrogram_height))

        uncertainties = (1 - np.max(ys,axis=1)).reshape(ys.shape[0],1)
        speaker_ids = np.full(uncertainties.shape, speaker)
        indices = indices.reshape(uncertainties.shape)

        return np.hstack((uncertainties, speaker_ids, indices))
