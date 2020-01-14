from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.models import Model

from common.utils.paths import get_experiment_nets, get_evaluation_list
from common.analysis.metrics.eer import equal_error_rate, eer_direct
from common.clustering.generate_embeddings import generate_utterance_embeddings

from networks.vggish.core.data_gen import generate_test_data_h5, generate_evaluation_data
from . import plot_saver as ps

import h5py
import numpy as np
import random
import time
import wandb

CALLBACK_PERIOD = 1

# ModelCheckpoint taking active learning rounds (epoch resets) into account
# -------------------------------------------------------------------
class ActiveLearningModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, period):
        super().__init__(
            filepath=filepath,
            period=period
        )

        self.alr_epoch = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.alr_epoch += 1
        super().on_epoch_begin(self.alr_epoch, logs)

    def on_epoch_end(self, epoch, logs={}):
        super().on_epoch_end(self.alr_epoch, logs)


class ActiveLearningUncertaintyCallback(Callback):
    def __init__(self, dataset, config, logger, segment_size, spectrogram_height):
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.logger = logger
        self.segment_size = segment_size
        self.spectrogram_height = spectrogram_height
        self.speakers_to_sample = 32
        self.current_epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        super().on_epoch_end(epoch, logs)
        
        if self.current_epoch % CALLBACK_PERIOD == 0:
            self._calculate_and_log_uncertainties()

        self.current_epoch += 1

    def _calculate_and_log_uncertainties(self):
        start = time.time()

        # do uncertainty sampling
        # ==========================================
        # 
        uncertainty_results = np.empty((0,3))

        # Iterate over a set of speakers and perform uncertainty sampling on their samples
        #
        for speaker in random.sample(self.dataset.get_train_speaker_list(), self.speakers_to_sample):
            uncertainty_results = np.append(uncertainty_results, self._perform_uncertainty_sampling(speaker), axis=0)

        uncertainties = uncertainty_results[:,0].astype(np.float)
        
        end = time.time()
        uncertainties_time_taken = end - start
        self.logger.info("Calculating Uncertainties took {}".format(uncertainties_time_taken))

        wandb.log({
            'al_uncertainty_min': np.min(uncertainties),
            'al_uncertainty_max': np.max(uncertainties),
            'al_uncertainty_mean': np.mean(uncertainties),
            'uncertainties_time_taken': uncertainties_time_taken
        }, commit=False)

    def _perform_uncertainty_sampling(self, speaker):
        if self.config.getboolean('active_learning','enabled'):
            al_indices = self.dataset.get_train_statistics()[speaker]['al']
        else:
            al_indices = self.dataset.get_train_statistics()[speaker]['train']

        if len(al_indices) == 0:
            return np.empty((0,3))

        spectrograms = self.dataset.get_train_file()['data/'+speaker][al_indices][:]

        Xb = np.zeros((spectrograms.shape[0], self.segment_size, self.spectrogram_height, 1))
        indices = np.zeros((spectrograms.shape[0]),dtype=np.int)

        for i in range(spectrograms.shape[0]):
            spect = spectrograms[i].reshape((spectrograms[i].shape[0] // self.spectrogram_height, self.spectrogram_height))

            if spect.shape[0] < self.segment_size:
                # In case the sample is shorter than the segment_length,
                # we need to artificially prolong it
                # 
                num_repeats = self.segment_size // spect.shape[0] + 1
                spect = np.tile(spect, (num_repeats,1))

            seg_idx = random.randint(0, spect.shape[0] - self.segment_size)
            Xb[i,:,:,0] = spect[seg_idx:seg_idx + self.segment_size, :]
            indices[i] = al_indices[i]

        ys = self.model.predict(Xb)

        uncertainties = (1 - np.max(ys, axis=1)).reshape(ys.shape[0], 1)
        speaker_ids = np.full(uncertainties.shape, speaker)
        indices = indices.reshape(uncertainties.shape)

        return np.hstack((uncertainties, speaker_ids, indices))

class EERCallback(Callback):
    def __init__(self, dataset, config, logger, segment_size, spectrogram_height):
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.logger = logger
        self.segment_size = segment_size
        self.spectrogram_height = spectrogram_height
        self.current_epoch = 0 

    def on_epoch_end(self, epoch, logs={}):
        super().on_epoch_end(epoch, logs)

        if self.current_epoch % CALLBACK_PERIOD == 0:
            self._calculate_and_log_eer()

        self.current_epoch += 1

    def _calculate_eer__eval_lists(self):
        return {
            'vox1': get_evaluation_list('list_vox1'),
            'vox1-cleaned': get_evaluation_list('list_vox1_c')
        }

    def _calculate_eer__validation_lists(self):
        eval_lists = self._calculate_eer__eval_lists()
        utterances = list()

        for key in eval_lists:
            for line in open(eval_lists[key], 'r'):
                label, file1, file2 = line[:-1].split(' ')

                utterances.append(file1)
                utterances.append(file2)

        return set(utterances)

    def _calculate_eer__load_data(self):
        utterances = self._calculate_eer__validation_lists()

        X, y = generate_evaluation_data(
            utterances,
            self.config.get('test','voxceleb1_test'),
            self.segment_size,
            self.spectrogram_height
        )

        return X, y

    def _calculate_and_log_eer(self):
        # Load Data
        # ================================================================================
        # 
        start = time.time()
        # X, y = generate_test_data_h5('all', self.dataset, self.segment_size, self.spectrogram_height, max_files_per_speaker=16, max_segments_per_utterance=1)
        X, y = self._calculate_eer__load_data()
        end = time.time()
        data_time_taken = end - start
        self.logger.info("Preparing EER data took {}\t{}\t{}".format(data_time_taken, X.shape, y.shape))

        # Prepare Partial Model
        # ================================================================================
        # 
        start = time.time()
        out_layer = self.config.getint('pairwise_lstm', 'out_layer')
        model_partial = Model(
            inputs=self.model.input,
            outputs=self.model.layers[out_layer].output
        )
        end = time.time()
        model_time_taken = end - start
        self.logger.info("Loading Model took {}".format(model_time_taken))

        # Predict Embeddings
        # ================================================================================
        start = time.time()
        output = np.asarray(model_partial.predict(X))        
        end = time.time()
        prediction_time_taken = end - start
        self.logger.info("Calculating Predictions took {}".format(prediction_time_taken))

        # # Generate Utterance Embeddings
        # # ================================================================================
        # # 
        # start = time.time()
        # embeddings = generate_utterance_embeddings(output, y)
        # end = time.time()
        # embeddings_time_taken = end - start
        # self.logger.info("Generating Embeddings took {}".format(embeddings_time_taken))

        # Prepare Vox1-List Pairs
        # ================================================================================
        start = time.time()
        num_pairs = sum(1 for line in open(self._calculate_eer__eval_lists()['vox1']))

        true_scores = np.zeros((num_pairs,))
        scores = np.zeros((num_pairs,))
        cosine_scores = np.zeros((num_pairs,))

        index = 0
        for line in open(self._calculate_eer__eval_lists()['vox1']):
            line = line.rstrip() # remove ('\n')
            label, file1, file2 = line.split(' ')
            
            true_scores[index] = label

            embedding1 = output[np.where(y == file1)[0][0]]
            embedding2 = output[np.where(y == file2)[0][0]]

            # "Simple" multiplicative and cosine similarity scores
            # 
            scores[index] = np.sum(embedding1 * embedding2)
            cosine_scores[index] = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

            index += 1

        end = time.time()
        embeddings_time_taken = end - start
        self.logger.info("Generating Embedding-Pair Scores took {}".format(embeddings_time_taken))

        # Calculate EER
        # ================================================================================
        # 
        start = time.time()
        # eer = equal_error_rate(embeddings)
        eer_sum = eer_direct(true_scores, scores)
        eer_cosine = eer_direct(true_scores, cosine_scores)

        end = time.time()
        eer_time_taken = end - start
        self.logger.info("Calculating EER took {}".format(eer_time_taken))

        # Log to WandB
        # ================================================================================
        # 
        wandb.log({
            'eer_sum': eer_sum,
            'eer_cosine': eer_cosine,
            'data_time_taken': data_time_taken,
            'model_time_taken': model_time_taken,
            'prediction_time_taken': prediction_time_taken,
            'eer_time_taken': eer_time_taken
        }, commit=False)
        # 'eer': eer,
        # 'embeddings_time_taken': embeddings_time_taken,


# Custom callback for own plots
# -------------------------------------------------------------------
class PlotCallback(Callback):
    def __init__(self, network_name, reset_train_begin=False):
        super().__init__()
        self.network_name = network_name
        self.reset_train_begin = reset_train_begin
        self.reset()

    def reset(self):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.logs = []

    def on_train_begin(self, logs={}):
        if self.reset_train_begin:
            self.reset()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        
        self.i += 1
        
        ps.save_accuracy_plot_direct(self.network_name, self.acc, self.val_acc)
        ps.save_loss_plot_direct(self.network_name, self.losses, self.val_losses)


# ActiveLearningLogCallback, active learning round aware epoch number logger
# -------------------------------------------------------------------
class ActiveLearningEpochLogger(Callback):
    def __init__(self, logger, total_epochs):
        super().__init__()

        self.alr_epoch = 0
        self.logger = logger
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs={}):
        self.alr_epoch += 1
        self.logger.info("Total Epoch {}/{}".format(self.alr_epoch, self.total_epochs))
