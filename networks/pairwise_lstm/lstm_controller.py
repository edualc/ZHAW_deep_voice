"""
The controller to train and test the pairwise_lstm network
"""

import numpy as np
from keras.models import Model
from keras.models import load_model
import logging
import h5py
from tqdm import tqdm

from common.clustering.generate_embeddings import generate_embeddings
from common.network_controller import NetworkController
from common.utils import TimeCalculator
from common.utils.logger import *
from common.utils.paths import *
from common.data.dataset import DeepVoiceDataset
from .bilstm_2layer_dropout_plus_2dense import bilstm_2layer_dropout
from .core.data_gen import generate_test_data_h5
from common.spectrogram.speaker_train_splitter import SpeakerTrainSplit
from networks.losses import get_custom_objects, get_loss
import common.utils.pickler as pickler

from keras.models import model_from_json

class LSTMController(NetworkController):
    def __init__(self, name, config, dev, best):
        super().__init__(name, config, dev)

        self.spectrogram_height = config.getint('pairwise_lstm','spectrogram_height')
        self.seg_size = config.getint('pairwise_lstm', 'seg_size')
        self.out_layer = config.getint('pairwise_lstm', 'out_layer')
        self.vec_size = config.getint('pairwise_lstm', 'vec_size')
        
        self.best = best
        self.dataset = DeepVoiceDataset(self.config, initialized=False)

        self.network_component = bilstm_2layer_dropout(
            self.name,
            segment_size=self.seg_size,
            spectrogram_height=self.spectrogram_height,
            config=self.config,
            dataset=self.dataset
        )

    def train_network(self):
        self.network_component.run_network()

    # Contains the VoxCeleb1 Evaluation Lists
    def eval_network__lists(self):
        # return {
        #     'vox1-cleaned': get_evaluation_list('list_vox1_c')
        # }
        
        return {
            'vox1': get_evaluation_list('list_vox1'),
            'vox1-cleaned': get_evaluation_list('list_vox1_c'),
            'vox1-E': get_evaluation_list('list_vox1_e'),
            'vox1-E-cleaned': get_evaluation_list('list_vox1_ec'),
            'vox1-H': get_evaluation_list('list_vox1_h'),
            'vox1-H-cleaned': get_evaluation_list('list_vox1_hc')
        }

    def eval_network(self):
        logger = get_logger('pairwise_lstm', logging.INFO)
        logger.info('Run pairwise_lstm eval')

        base_network = self.network_component.create_net()
        
        # Calculate which utterances are needed (unique counts)
        # for the evaluation lists provided
        # ========================================================================
        # 
        logger.info('Gathering utterances')
        utterances = set()
        eval_lists = self.eval_network__lists()

        for key in eval_lists.keys():
            for line in open(eval_lists[key], 'r'):
                label, file1, file2 = line[:-1].split(' ')

                utterances.add(file1)
                utterances.add(file2)

        if self.best:
            file_regex = self.name + ".*_best\.h5"
        else:
            file_regex = self.name + ".*\.h5"

        checkpoints = list_all_files(get_experiment_nets(), file_regex)

        for checkpoint in checkpoints:
            logger.info('Running checkpoint: ' + checkpoint)

            checkpoint_eval_file = get_results(checkpoint + '__eval_data.h5')

            if not os.path.isfile(checkpoint_eval_file):
                # Create h5py File if it does not exist yet
                with h5py.File(checkpoint_eval_file, 'a') as f:
                    pass

                # Prepare Embedding Network
                # ========================================================================
                # Load weights into network
                base_network.load_weights(get_experiment_nets(checkpoint))

                # Get a Model with the embedding layer as output and predict
                model_partial = Model(inputs=base_network.input, outputs=base_network.layers[self.out_layer].output)

                # Prepare Data
                # ========================================================================
                logger.info('Spectrogram data not present, extracting spectrograms.')

                test_data = h5py.File(self.config.get('test','voxceleb1_test') + '.h5', 'r')
                dev_data = h5py.File(self.config.get('test','voxceleb1_dev') + '.h5', 'r')
                
                # shift by 50% of seg_size
                sliding_window_shift = self.seg_size // 2

                for idx, utterance in enumerate(tqdm(list(utterances), ascii=True, ncols=100, desc='preparing spectrogram windows for predictions')):
                    spectrogram_data = None
                    spectrogram_labels = list()

                    split = utterance.split('/')
                    speaker = split.pop(0)
                    file = '/'.join(split)

                    if speaker in test_data['audio_names'].keys():
                        # found in test dataset
                        file_index = np.where(test_data['audio_names'][speaker][:] == file)[0][0]
                        full_spect = test_data['data'][speaker][file_index]
                    else:
                        # fallback dev dataset
                        file_index = np.where(dev_data['audio_names'][speaker][:] == file)[0][0]
                        full_spect = dev_data['data'][speaker][file_index]
                    
                    spect = full_spect.reshape((full_spect.shape[0] // self.spectrogram_height, self.spectrogram_height))

                    # Standardize
                    mu = np.mean(spect, 0, keepdims=True)
                    stdev = np.std(spect, 0, keepdims=True)
                    spect = (spect - mu) / (stdev + 1e-5)

                    if spect.shape[0] < self.seg_size + 4 * sliding_window_shift:
                        # In case the sample is shorter than the segment_length,
                        # we need to artificially prolong it
                        # 
                        num_repeats = ((self.seg_size + 4 * sliding_window_shift) // spect.shape[0]) + 1
                        spect = np.tile(spect, (num_repeats,1))

                    offset = 0
                    sample_spects = list()

                    # Extract spectrograms with offset of :sliding_window_shift
                    while offset < spect.shape[0] - self.seg_size:
                        sample_spects.append(spect[offset:offset + self.seg_size,:])
                        spectrogram_labels.append(utterance)
                        offset += sliding_window_shift

                    spectrogram_data = np.asarray(sample_spects)
                    spectrogram_labels = np.string_(spectrogram_labels)
                
                    with h5py.File(checkpoint_eval_file, 'a') as f:
                        # if 'spectrograms' not in f.keys():
                        #     f.create_dataset('spectrograms', data=spectrogram_data, maxshape=(None, spectrogram_data.shape[1], spectrogram_data.shape[2]))
                        # else:
                        #     f['spectrograms'].resize((f['spectrograms'].shape[0] + spectrogram_data.shape[0]), axis=0)
                        #     f['spectrograms'][-spectrogram_data.shape[0]:] = spectrogram_data
                        
                        if 'labels' not in f.keys():
                            f.create_dataset('labels', data=spectrogram_labels, maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'))
                        else:
                            f['labels'].resize((f['labels'].shape[0] + spectrogram_labels.shape[0]), axis=0)
                            f['labels'][-spectrogram_labels.shape[0]:] = spectrogram_labels
                        
                        embeddings = model_partial.predict(spectrogram_data)

                        if 'embeddings' not in f.keys():
                            f.create_dataset('embeddings', data=embeddings, maxshape=(None, embeddings.shape[1]))
                        else:
                            f['embeddings'].resize((f['embeddings'].shape[0] + embeddings.shape[0]), axis=0)
                            f['embeddings'][-embeddings.shape[0]:] = embeddings

                    del spectrogram_data
                    del spectrogram_labels
                    del embeddings

            # Perform Evaluations
            # ========================================================================
            logger.info('Running evaluation on speaker embeddings.')

            data_file = h5py.File(checkpoint_eval_file, 'r')
            eer_values = dict()

            checkpoint_eval_result_file = get_results(checkpoint + '__eval_result.h5')
            
            # Result file already exists
            if os.path.isfile(checkpoint_eval_result_file):
                result_file = h5py.File(checkpoint_eval_result_file, 'r')

            # No Result file is present, we need to calculate the values
            else:
                result_file = h5py.File(checkpoint_eval_result_file, 'w')

                for list_key in self.eval_network__lists():
                    print("\tEvaluating {}...".format(list_key))
                    eer_values[list_key] = dict()

                    num_similarity_types = 9
                    num_lines = sum(1 for line in open(self.eval_network__lists()[list_key]))
                    result_file.create_dataset(list_key, data=np.zeros((num_lines, num_similarity_types), dtype=np.float32))

                    all_labels = data_file['labels'][:]

                    for i, line in enumerate(tqdm(open(self.eval_network__lists()[list_key], 'r'), total=num_lines, desc='Calculating similarities over evaluation pairs...', ascii=True, ncols=100)):
                        y_true, id1, id2 = line[:-1].split(' ')
                        y_true = int(y_true)

                        idx1 = np.where(all_labels == id1)
                        idx2 = np.where(all_labels == id2)

                        embeddings1 = data_file['embeddings'][idx1]
                        embeddings2 = data_file['embeddings'][idx2]
                        embeddings_stacked = np.vstack((embeddings1, embeddings2))

                        y_values_for_current_pair = list()
                        y_values_for_current_pair.append(y_true)
                        
                        # Mean-Cosine EER
                        # =====================================================================================
                        # 
                        embedding1 = np.mean(embeddings1, axis=0)
                        embedding2 = np.mean(embeddings2, axis=0)

                        # Calculation Cosine Distance for mean of both embeddings
                        # 
                        y_pred_mean = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
                        y_values_for_current_pair.append(y_pred_mean)

                        # Complete Linkage EER
                        # =====================================================================================
                        # 
                        from scipy.cluster.hierarchy import fcluster, linkage
                        from scipy.spatial.distance import cdist

                        # Calculation of distances between all the embeddings
                        # 
                        embeddings_distance = cdist(embeddings_stacked, embeddings_stacked, 'cosine')
                        embeddings_distance_condensed = embeddings_distance[np.triu_indices(embeddings_distance.shape[0],1)]
                        # distance_between_utterances = embeddings_distance[:embeddings1.shape[0],-embeddings2.shape[0]:]
                    
                        # Linkage Calculations for different linkage methods
                        # 
                        embeddings_single_linkage = linkage(embeddings_distance_condensed, 'single', 'cosine')
                        embeddings_complete_linkage = linkage(embeddings_distance_condensed, 'complete', 'cosine')
                        embeddings_average_linkage = linkage(embeddings_distance_condensed, 'average', 'cosine')
                        embeddings_weighted_linkage = linkage(embeddings_distance_condensed, 'weighted', 'cosine')
                        embeddings_centroid_linkage = linkage(embeddings_distance_condensed, 'centroid', 'cosine')
                        embeddings_median_linkage = linkage(embeddings_distance_condensed, 'median', 'cosine')
                        embeddings_ward_linkage = linkage(embeddings_distance_condensed, 'ward', 'cosine')

                        # Extraction of the distance for between the last two remaining clusters
                        # according to the linkage(s) defined above
                        # 
                        y_pred_single_linkage = embeddings_single_linkage[-1:,2][0]
                        y_values_for_current_pair.append(y_pred_single_linkage)

                        y_pred_complete_linkage = embeddings_complete_linkage[-1:,2][0]
                        y_values_for_current_pair.append(y_pred_complete_linkage)

                        y_pred_average_linkage = embeddings_average_linkage[-1:,2][0]
                        y_values_for_current_pair.append(y_pred_average_linkage)

                        y_pred_weighted_linkage = embeddings_weighted_linkage[-1:,2][0]
                        y_values_for_current_pair.append(y_pred_weighted_linkage)
                        
                        y_pred_centroid_linkage = embeddings_centroid_linkage[-1:,2][0]
                        y_values_for_current_pair.append(y_pred_centroid_linkage)
                        
                        y_pred_median_linkage = embeddings_median_linkage[-1:,2][0]
                        y_values_for_current_pair.append(y_pred_median_linkage)
                        
                        y_pred_ward_linkage = embeddings_ward_linkage[-1:,2][0]
                        y_values_for_current_pair.append(y_pred_ward_linkage)

                        # TODO: Dominant Sets EER
                        # =====================================================================================
                        # 
                        import common.dominant_sets.dominantset as ds

                        labels1 = data_file['labels'][idx1]
                        labels2 = data_file['labels'][idx2]
                        labels_stacked = np.hstack((labels1, labels2))

                        # import code; code.interact(local=dict(globals(), **locals()))

                        unique_labels, labels_categorical = np.unique(labels_stacked, return_inverse=True)

                        # eps = 0.01
                        # phi = 1e-7
                        
                        # mr_results = dict()

                        # for eps in [1e-3, 1e-2, 1e-1]:
                        #     mr_results[eps] = dict()

                        #     for phi in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]:
                        #         mr_results[eps][phi] = 0

                        #         dos = ds.DominantSetClustering(
                        #             feature_vectors=np.array(embeddings_stacked),
                        #             speaker_ids=labels_categorical,
                        #             metric='cosine',
                        #             # hungarian method, True: max
                        #             dominant_search=False,
                        #             reassignment='noise',
                        #             # eps 1e-6 default
                        #             epsilon=eps,
                        #             # phi 0.1 default
                        #             cutoff=phi
                        #         )
                        #         dos.apply_clustering()

                        #         mr, randi, acp = dos.evaluate()

                        #         if mr > 0.0:
                        #             mr_results[eps][phi] += 1
                                

                        #         print("\teps: {}\tphi: {}\t\tmr: {}".format(eps, phi, mr))
                        # import code; code.interact(local=dict(globals(), **locals()))

                        # Write calculated similarities to result file
                        # =======================================================================================
                        #
                        result_file[list_key][i] = np.asarray(y_values_for_current_pair)
                    
                    # import code; code.interact(local=dict(globals(), **locals()))

            for list_key in self.eval_network__lists():
                # Calculate EER with similarities
                # =======================================================================================
                #
                from common.analysis.metrics.eer import eer_direct

                eer_values[list_key] = {
                    'mean': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,1]) * 100, 5),
                    'linkage_single': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,2], 0) * 100, 5),
                    'linkage_complete': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,3], 0) * 100, 5),
                    'linkage_average': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,4], 0) * 100, 5),
                    'linkage_weighted': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,5], 0) * 100, 5),
                    'linkage_centroid': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,6], 0) * 100, 5),
                    'linkage_median': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,7], 0) * 100, 5),
                    'linkage_ward': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,8], 0) * 100, 5)
                }

                print("====================================================================================")
                print()
                print("EER for {}:".format(list_key))
                for k in sorted(eer_values[list_key], key=eer_values[list_key].get):
                    print("\t{}\t\t{}".format(eer_values[list_key][k], k))
                print()
                print("====================================================================================")

                # import code; code.interact(local=dict(globals(), **locals()))

    def get_embeddings(self):
        logger = get_logger('pairwise_lstm', logging.INFO)
        logger.info('Run pairwise_lstm test')
        logger.info('out_layer -> ' + str(self.out_layer))
        logger.info('seg_size -> ' + str(self.seg_size))
        logger.info('vec_size -> ' + str(self.vec_size))

        base_network = self.network_component.create_net()

        X_long, speakers_long = generate_test_data_h5('long', self.dataset, self.seg_size, self.spectrogram_height)
        logger.info('X_long -> ' + str(X_long.shape))

        X_short, speakers_short = generate_test_data_h5('short', self.dataset, self.seg_size, self.spectrogram_height)
        logger.info('X_short -> ' + str(X_short.shape))

        # Prepare return values
        set_of_embeddings = []
        set_of_speakers = []
        speaker_numbers = []
        set_of_utterance_embeddings = []
        set_of_total_times = []

        if self.best:
            file_regex = self.name + ".*_best\.h5"
        else:
            file_regex = self.name + ".*\.h5"

        checkpoints = list_all_files(get_experiment_nets(), file_regex)

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
                # Load weights into network
                base_network.load_weights(get_experiment_nets(checkpoint))

                # Get a Model with the embedding layer as output and predict
                model_partial = Model(inputs=base_network.input, outputs=base_network.layers[self.out_layer].output)

                logger.info('running predict on train set')
                output_long = np.asarray(model_partial.predict(X_long))
                logger.info('output_long len -> ' + str(output_long.shape))
                
                logger.info('running predict on test set')
                output_short = np.asarray(model_partial.predict(X_short))
                logger.info('output_short len -> ' + str(output_short.shape))

                embeddings, speakers, num_embeddings, utterance_embeddings = generate_embeddings(
                    [output_long, output_short], [speakers_long,
                    speakers_short], self.vec_size
                )

                pickler.save((embeddings, speakers, num_embeddings, utterance_embeddings), checkpoint_result_pickle)

            # Fill the embeddings and speakers into the arrays
            set_of_embeddings.append(embeddings)
            set_of_speakers.append(speakers)
            speaker_numbers.append(num_embeddings)
            set_of_utterance_embeddings.append(utterance_embeddings)

            # Calculate the time per utterance
            calculated_time = TimeCalculator.calc_time_all_utterances([speakers_long, speakers_short], self.seg_size)
            set_of_total_times.append(calculated_time)

        # Add out_layer to checkpoint names
        checkpoints = list(map(lambda x: x.split('.')[0] + '__ol' + str(self.out_layer) + '.' + x.split('.')[1], checkpoints))
        print("checkpoints: {}".format(checkpoints))

        logger.info('Pairwise_lstm test done.')
        return checkpoints, set_of_embeddings, set_of_speakers, speaker_numbers, set_of_total_times, set_of_utterance_embeddings
