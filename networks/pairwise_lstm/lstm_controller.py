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

# Evaluation Stuff
from common.analysis.metrics.eer import eer_direct
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist
import common.dominant_sets.dominantset as ds
from sklearn.metrics import homogeneity_score

# Mean: Cosine+Sum
# Linkage: 7
# DominantSet: HS, MR, RANDI, ACP
# 
# NUM_SIMILARITY_TYPES = 2 + 7 + 4 
# NUM_SIMILARITY_TYPES = 2 + 10 + 4 # w/ h-min stuff
NUM_SIMILARITY_TYPES = 2 + 7 + 84 # w/ dominant set as clustering augmentation step

class LSTMController(NetworkController):
    def __init__(self, name, config, dev, best):
        super().__init__(name, config, dev)

        self.spectrogram_height = config.getint('pairwise_lstm','spectrogram_height')
        self.seg_size = config.getint('pairwise_lstm', 'seg_size')
        self.out_layer = config.getint('pairwise_lstm', 'out_layer')
        self.vec_size = config.getint('pairwise_lstm', 'vec_size')
        
        self.best = best
        self.dataset = DeepVoiceDataset(self.config, initialized=True)

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
        return {
            'vox1-cleaned': get_evaluation_list('list_vox1_c')
        }
        
        # return {
        #     'vox1': get_evaluation_list('list_vox1'),
        #     'vox1-cleaned': get_evaluation_list('list_vox1_c'),
        #     'vox1-E': get_evaluation_list('list_vox1_e'),
        #     'vox1-E-cleaned': get_evaluation_list('list_vox1_ec'),
        #     'vox1-H': get_evaluation_list('list_vox1_h'),
        #     'vox1-H-cleaned': get_evaluation_list('list_vox1_hc')
        # }

    def eval_network__checkpoint_eval_data_file(self, checkpoint):
        return get_results(checkpoint + '__eval_data.h5')

    def eval_network__checkpoint_eval_result_file(self, checkpoint):
        return get_results(checkpoint + '__eval_result.h5')

    def eval_network__create_embeddings_file_for_checkpoint(self, checkpoint, base_network, utterances):
        checkpoint_eval_file = self.eval_network__checkpoint_eval_data_file(checkpoint)

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

                # del spectrogram_data
                # del spectrogram_labels
                # del embeddings

    def eval_network__checkpoints(self):
        if self.best:
            file_regex = self.name + ".*_best\.h5"
        else:
            file_regex = self.name + ".*\.h5"

        return list_all_files(get_experiment_nets(), file_regex)

    def eval_network__unique_utterances(self):
        utterances = set()
        eval_lists = self.eval_network__lists()

        for key in eval_lists.keys():
            for line in open(eval_lists[key], 'r'):
                label, file1, file2 = line[:-1].split(' ')

                utterances.add(file1)
                utterances.add(file2)

        return utterances

    def eval_network__compare_pairs__mean_cosine(self, embeddings1, embeddings2):
        embedding1 = np.mean(embeddings1, axis=0)
        embedding2 = np.mean(embeddings2, axis=0)

        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    def eval_network__compare_pairs__mean_sum(self, embeddings1, embeddings2):
        embedding1 = np.mean(embeddings1, axis=0)
        embedding2 = np.mean(embeddings2, axis=0)

        return np.sum(embedding1 * embedding2)

    def eval_network__compare_pairs__linkages(self, embeddings_stacked, y_true=None):
        # Calculation of distances between all the embeddings
        # 
        embeddings_distance = cdist(embeddings_stacked, embeddings_stacked, 'cosine')
        embeddings_distance_condensed = embeddings_distance[np.triu_indices(embeddings_distance.shape[0],1)]
        # distance_between_utterances = embeddings_distance[:embeddings1.shape[0],-embeddings2.shape[0]:]

        # try:
        #     lavg = linkage(embeddings_distance_condensed, 'average', 'cosine')

        #     left_cluster_index = int(lavg[-1, 0])
        #     right_cluster_index = int(lavg[-1, 1])

        #     # left = lavg[left_cluster_index - embeddings_stacked.shape[0], :]
        #     # right = lavg[right_cluster_index - embeddings_stacked.shape[0], :]

        #     if left_cluster_index < embeddings_stacked.shape[0]:
        #         left_cluster_num = 1
        #         left_cluster_height = 0
        #         # right_cluster_num = embeddings_stacked.shape[0] - 1
        #         # right_cluster_height = 
        #     else:
        #         left_cluster_num = lavg[left_cluster_index - embeddings_stacked.shape[0], 3]
        #         left_cluster_height = lavg[left_cluster_index - embeddings_stacked.shape[0], 2]

        #     if right_cluster_index < embeddings_stacked.shape[0]:
        #         right_cluster_num = 1
        #         right_cluster_height = 0
        #     else:
        #         right_cluster_num = lavg[right_cluster_index - embeddings_stacked.shape[0], 3]
        #         right_cluster_height = lavg[right_cluster_index - embeddings_stacked.shape[0], 2]
            
        #     ce = np.abs(left_cluster_num - right_cluster_num) / ((left_cluster_num + right_cluster_num) / 2)
        #     H = lavg[-1,2]
        #     hmax = H - max(left_cluster_height, right_cluster_height)
        #     hmin = H - min(left_cluster_height, right_cluster_height)

        #     # from scipy.cluster.hierarchy import dendrogram
        #     # import matplotlib.pyplot as plt

        #     # if (int(y_true) == 1 and H >= 0.6) or (int(y_true) == 0 and H <= 0.4):
        #     #     plt.figure()
        #     #     dendrogram(lavg)
                
        #     #     # import os
        #     #     # if os.path.isfile('test.png'):
        #     #     #     os.remove('test.png')
                
        #     #     if y_true is not None:
        #     #         if int(y_true) == 0:
        #     #             plt.title('DIFFERENT speakers')
        #     #         elif int(y_true) == 1:
        #     #             plt.title('SAME speakers')

        #     #     import time
        #     #     plt.ylim([0,1])
        #     #     plt.savefig('dendrograms/' + str(time.time()) + '.png')
        #     #     plt.close()
        #     #     # import code; code.interact(local=dict(globals(), **locals()))

        # except IndexError:
        #     import code; code.interact(local=dict(globals(), **locals()))

        # tmp = list()
        # tmp.append(H)
        # tmp.append(hmin)
        # tmp.append(hmax)
        # tmp.append(ce)
        # tmp.append(ce + H)
        # tmp.append(ce + hmin)
        # tmp.append(ce + hmax)
        # tmp.append(ce * H)
        # tmp.append(ce * hmin)
        # tmp.append(ce * hmax)

        tmp = list()
        for linkage_method in ['single','complete','average','weighted','centroid','median','ward']:
            tmp.append(linkage(embeddings_distance_condensed, linkage_method, 'cosine')[-1:,2][0])
        
        return tmp

    def eval_network__compare_pairs__dominant_set(self, embeddings_stacked, labels_stacked, y_true=None):
        # return [0., 0., 0., 0.]

        unique_labels, labels_categorical = np.unique(labels_stacked, return_inverse=True)

        # eps = 1e-6
        # phi = 0.1

        tmp = list()

        for eps in [1e-5, 1e-6, 1e-7, 1e-8]:
            for phi in [1e-2, 1e-3, 1e-4]:

                dos = ds.DominantSetClustering(feature_vectors=np.array(embeddings_stacked), speaker_ids=labels_categorical, metric='cosine', dominant_search=False, reassignment='noise', epsilon=eps, cutoff=phi)
                dos.apply_clustering()

                mr, randi, acp = dos.evaluate()
                hs = homogeneity_score(labels_categorical, dos.ds_result)
                

                num_clusters = dos.cluster_counter
                new_embeddings = np.zeros((num_clusters, embeddings_stacked.shape[1]))
                # new_labels = np.zeros((num_clusters,))

                # Use dominant sets to get new clusters
                # 
                for i in range(num_clusters):
                    ids = np.where(dos.ds_result == i)

                    cluster_vals = dos.ds_vals[ids]
                    cluster_results = dos.ds_result[ids]
                    cluster_embeddings = np.copy(embeddings_stacked[ids,:][0])

                    for j in range(cluster_vals.shape[0]):
                        influence_multiplier = cluster_vals[j]
                        cluster_embeddings[j,:] *= cluster_vals[j]

                    new_embeddings[i,:] = np.sum(cluster_embeddings, axis=0)

                    # uniques, unique_counts = np.unique(dos.get_most_participating()[ids], return_counts=True)

                    # if uniques.shape[0] > 1:
                    #     print('')
                    #     print(dos.ds_result)
                    #     print(uniques)
                    #     print(unique_counts)
                    #     import code; code.interact(local=dict(globals(), **locals()))

                    #     new_labels[i] = 0
                    # else:
                    #     new_labels[i] = uniques[0]

                # Use new clusters for hierarchical clustering
                # 
                embeddings_distance = cdist(new_embeddings, new_embeddings, 'cosine')
                embeddings_distance_condensed = embeddings_distance[np.triu_indices(embeddings_distance.shape[0],1)]

                # for linkage_method in ['single','complete','average','weighted','centroid','median','ward']:
                for linkage_method in ['complete']:
                    tmp.append(linkage(embeddings_distance_condensed, linkage_method, 'cosine')[-1:,2][0])
                
        return tmp


        # print('')
        # print('********************')
        # print("\t{}\t{}".format('hs',hs))
        # print("\t{}\t{}".format('mr',mr))
        # print("\t{}\t{}".format('randi',randi))
        # print("\t{}\t{}".format('acp',acp))
        # print(homogeneity_score(labels_categorical, dos.ds_result), dos.evaluate())
        
        




        # # eps = 0.01
        # # phi = 1e-7

        # mr_results = dict()
        # print('')

        # # for eps in [1e-3, 1e-2, 1e-1]:
        # for eps in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
        #     mr_results[eps] = dict()

        #     # for phi in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]:
        #     for phi in [1e-2, 1e-3, 1e-4]:
        #         mr_results[eps][phi] = 0

        # # DEFAULT: phi=0.1, eps=1e6, dominant_search=False(hungarian)
        #         dos = ds.DominantSetClustering(feature_vectors=np.array(embeddings_stacked), speaker_ids=labels_categorical, metric='cosine', dominant_search=False, reassignment='noise', epsilon=eps, cutoff=phi)
        #         dos.apply_clustering()

        #         mr, randi, acp = dos.evaluate()

        #         if mr > 0.0:
        #             mr_results[eps][phi] += 1
                

        #         if dos.cluster_counter == 2:
        #             print("\t*[2/{0}] eps: {1:1.4f}\tphi: {2}\t\tmr: {3:2.5f}\trandi: {4:2.5f}\tacp: {5:2.5f}".format(y_true, eps, phi, mr, randi, acp))
        #         else:
        #             print("\t [{0}/{1}] eps: {2:1.4f}\tphi: {3}\t\tmr: {4:2.5f}\trandi: {5:2.5f}\tacp: {6:2.5f}".format(dos.cluster_counter, y_true, eps, phi, mr, randi, acp))

        #         # if dos.cluster_counter == 2:
        #         #     print("\t*[2] eps: {}\tphi: {}\t\tmr: {}".format(eps, phi, mr))
        #         # else:
        #         #     print("\t [{}] eps: {}\tphi: {}\t\tmr: {}".format(dos.cluster_counter, eps, phi, mr))

        # import code; code.interact(local=dict(globals(), **locals()))


        return [hs, mr, randi, acp]

    def eval_network__compare_pairs(self, checkpoint):
        data_file = h5py.File(self.eval_network__checkpoint_eval_data_file(checkpoint), 'r')
        result_file_path = self.eval_network__checkpoint_eval_result_file(checkpoint)
        
        # Result file already exists
        if not os.path.isfile(result_file_path):
            result_file = h5py.File(result_file_path, 'w')

            for list_key in self.eval_network__lists():
                print("\tEvaluating {}...".format(list_key))

                num_lines = sum(1 for line in open(self.eval_network__lists()[list_key]))
                result_file.create_dataset(list_key, data=np.zeros((num_lines, NUM_SIMILARITY_TYPES + 1), dtype=np.float32))

                all_labels = data_file['labels'][:]

                for i, line in enumerate(tqdm(open(self.eval_network__lists()[list_key], 'r'), total=num_lines, desc='Calculating similarities over evaluation pairs...', ascii=True, ncols=100)):
                    y_true, id1, id2 = line[:-1].split(' ')
                    y_true = int(y_true)

                    idx1 = np.where(all_labels == id1)
                    idx2 = np.where(all_labels == id2)

                    embeddings1 = data_file['embeddings'][idx1]
                    embeddings2 = data_file['embeddings'][idx2]
                    embeddings_stacked = np.vstack((embeddings1, embeddings2))

                    y_pred_cosine_mean = self.eval_network__compare_pairs__mean_cosine(embeddings1, embeddings2)
                    y_pred_sum_mean = self.eval_network__compare_pairs__mean_sum(embeddings1, embeddings2)
                    y_preds_linkage = self.eval_network__compare_pairs__linkages(embeddings_stacked, y_true)

                    labels1 = data_file['labels'][idx1]
                    labels2 = data_file['labels'][idx2]
                    labels_stacked = np.hstack((labels1, labels2))

                    y_preds_dominant_set = self.eval_network__compare_pairs__dominant_set(embeddings_stacked, labels_stacked, y_true)

                    # Write calculated similarities to result file
                    # =======================================================================================
                    #
                    # result_file[list_key][i] = np.asarray(y_values_for_current_pair)
                    result_file[list_key][i] = np.concatenate([[y_true], [y_pred_cosine_mean], [y_pred_sum_mean], y_preds_linkage, y_preds_dominant_set])

            result_file.close()

    def eval_network(self):
        logger = get_logger('pairwise_lstm', logging.INFO)
        logger.info('Run pairwise_lstm eval')

        base_network = self.network_component.create_net()
        
        # Calculate which utterances are needed (unique counts)
        # for the evaluation lists provided
        # ========================================================================
        # 
        logger.info('Gathering utterances')
        utterances = self.eval_network__unique_utterances()
        checkpoints = self.eval_network__checkpoints()

        for checkpoint in checkpoints:
            logger.info('Generating embeddings for checkpoint: ' + checkpoint)
            self.eval_network__create_embeddings_file_for_checkpoint(checkpoint, base_network, utterances)

            # Perform Evaluations
            # ========================================================================
            logger.info('Running evaluation on speaker embeddings.')
            self.eval_network__compare_pairs(checkpoint)

            result_file = h5py.File(self.eval_network__checkpoint_eval_result_file(checkpoint), 'r')
            eer_values = dict()

            for list_key in self.eval_network__lists():
                eer_values[list_key] = {
                    'mean_cosine': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,1]) * 100, 5),
                    'mean_sum': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,2]) * 100, 5),
                    'linkage_single': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,3], 0) * 100, 5),
                    'linkage_complete': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,4], 0) * 100, 5),
                    'linkage_average': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,5], 0) * 100, 5),
                    'linkage_weighted': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,6], 0) * 100, 5),
                    'linkage_centroid': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,7], 0) * 100, 5),
                    'linkage_median': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,8], 0) * 100, 5),
                    'linkage_ward': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,9], 0) * 100, 5),

                    # 'H': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,3], 0) * 100, 5),
                    # 'hmin': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,4], 0) * 100, 5),
                    # 'hmax': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,5], 0) * 100, 5),
                    # 'ce': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,6], 0) * 100, 5),
                    # 'ce + H': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,7], 0) * 100, 5),
                    # 'ce + hmin': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,8], 0) * 100, 5),
                    # 'ce + hmax': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,9], 0) * 100, 5),
                    # 'ce * H': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,10], 0) * 100, 5),
                    # 'ce * hmin': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,11], 0) * 100, 5),
                    # 'ce * hmax': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,12], 0) * 100, 5)



                    # 'dominant_set_A_linkage_complete':     round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,10], 0) * 100, 5),
                    # 'dominant_set_B_linkage_complete':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,11], 0) * 100, 5),
                    # 'dominant_set_C_linkage_complete':    round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,12], 0) * 100, 5),
                    # 'dominant_set_D_linkage_complete':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,13], 0) * 100, 5),
                    # 'dominant_set_E_linkage_complete':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,14], 0) * 100, 5),
                    # 'dominant_set_F_linkage_complete':     round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,15], 0) * 100, 5),
                    # 'dominant_set_G_linkage_complete':       round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,16], 0) * 100, 5),
                    # 'dominant_set_H_linkage_complete':     round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,17], 0) * 100, 5),
                    # 'dominant_set_I_linkage_complete':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,18], 0) * 100, 5),
                    # 'dominant_set_J_linkage_complete':    round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,19], 0) * 100, 5),
                    # 'dominant_set_K_linkage_complete':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,20], 0) * 100, 5),
                    # 'dominant_set_L_linkage_complete':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,21], 0) * 100, 5)






                    'dominant_set_A_linkage_single':     round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,10], 0) * 100, 5),
                    'dominant_set_A_linkage_complete':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,11], 0) * 100, 5),
                    'dominant_set_A_linkage_average':    round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,12], 0) * 100, 5),
                    'dominant_set_A_linkage_weighted':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,13], 0) * 100, 5),
                    'dominant_set_A_linkage_centroid':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,14], 0) * 100, 5),
                    'dominant_set_A_linkage_median':     round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,15], 0) * 100, 5),
                    'dominant_set_A_linkage_ward':       round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,16], 0) * 100, 5),

                    'dominant_set_B_linkage_single':     round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,17], 0) * 100, 5),
                    'dominant_set_B_linkage_complete':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,18], 0) * 100, 5),
                    'dominant_set_B_linkage_average':    round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,19], 0) * 100, 5),
                    'dominant_set_B_linkage_weighted':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,20], 0) * 100, 5),
                    'dominant_set_B_linkage_centroid':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,21], 0) * 100, 5),
                    'dominant_set_B_linkage_median':     round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,22], 0) * 100, 5),
                    'dominant_set_B_linkage_ward':       round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,23], 0) * 100, 5),

                    'dominant_set_C_linkage_single':     round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,24], 0) * 100, 5),
                    'dominant_set_C_linkage_complete':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,25], 0) * 100, 5),
                    'dominant_set_C_linkage_average':    round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,26], 0) * 100, 5),
                    'dominant_set_C_linkage_weighted':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,27], 0) * 100, 5),
                    'dominant_set_C_linkage_centroid':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,28], 0) * 100, 5),
                    'dominant_set_C_linkage_median':     round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,29], 0) * 100, 5),
                    'dominant_set_C_linkage_ward':       round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,30], 0) * 100, 5),

                    'dominant_set_D_linkage_single':     round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,31], 0) * 100, 5),
                    'dominant_set_D_linkage_complete':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,32], 0) * 100, 5),
                    'dominant_set_D_linkage_average':    round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,33], 0) * 100, 5),
                    'dominant_set_D_linkage_weighted':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,34], 0) * 100, 5),
                    'dominant_set_D_linkage_centroid':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,35], 0) * 100, 5),
                    'dominant_set_D_linkage_median':     round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,36], 0) * 100, 5),
                    'dominant_set_D_linkage_ward':       round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,37], 0) * 100, 5),

                    'dominant_set_E_linkage_single':     round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,38], 0) * 100, 5),
                    'dominant_set_E_linkage_complete':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,39], 0) * 100, 5),
                    'dominant_set_E_linkage_average':    round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,40], 0) * 100, 5),
                    'dominant_set_E_linkage_weighted':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,41], 0) * 100, 5),
                    'dominant_set_E_linkage_centroid':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,42], 0) * 100, 5),
                    'dominant_set_E_linkage_median':     round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,43], 0) * 100, 5),
                    'dominant_set_E_linkage_ward':       round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,44], 0) * 100, 5),

                    'dominant_set_F_linkage_single':     round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,45], 0) * 100, 5),
                    'dominant_set_F_linkage_complete':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,46], 0) * 100, 5),
                    'dominant_set_F_linkage_average':    round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,47], 0) * 100, 5),
                    'dominant_set_F_linkage_weighted':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,48], 0) * 100, 5),
                    'dominant_set_F_linkage_centroid':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,49], 0) * 100, 5),
                    'dominant_set_F_linkage_median':     round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,50], 0) * 100, 5),
                    'dominant_set_F_linkage_ward':       round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,51], 0) * 100, 5),

                    'dominant_set_G_linkage_single':     round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,52], 0) * 100, 5),
                    'dominant_set_G_linkage_complete':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,53], 0) * 100, 5),
                    'dominant_set_G_linkage_average':    round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,54], 0) * 100, 5),
                    'dominant_set_G_linkage_weighted':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,55], 0) * 100, 5),
                    'dominant_set_G_linkage_centroid':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,56], 0) * 100, 5),
                    'dominant_set_G_linkage_median':     round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,57], 0) * 100, 5),
                    'dominant_set_G_linkage_ward':       round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,58], 0) * 100, 5),

                    'dominant_set_H_linkage_single':     round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,59], 0) * 100, 5),
                    'dominant_set_H_linkage_complete':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,60], 0) * 100, 5),
                    'dominant_set_H_linkage_average':    round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,61], 0) * 100, 5),
                    'dominant_set_H_linkage_weighted':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,62], 0) * 100, 5),
                    'dominant_set_H_linkage_centroid':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,63], 0) * 100, 5),
                    'dominant_set_H_linkage_median':     round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,64], 0) * 100, 5),
                    'dominant_set_H_linkage_ward':       round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,65], 0) * 100, 5),

                    'dominant_set_I_linkage_single':     round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,66], 0) * 100, 5),
                    'dominant_set_I_linkage_complete':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,67], 0) * 100, 5),
                    'dominant_set_I_linkage_average':    round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,68], 0) * 100, 5),
                    'dominant_set_I_linkage_weighted':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,69], 0) * 100, 5),
                    'dominant_set_I_linkage_centroid':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,70], 0) * 100, 5),
                    'dominant_set_I_linkage_median':     round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,71], 0) * 100, 5),
                    'dominant_set_I_linkage_ward':       round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,72], 0) * 100, 5),

                    'dominant_set_J_linkage_single':     round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,73], 0) * 100, 5),
                    'dominant_set_J_linkage_complete':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,74], 0) * 100, 5),
                    'dominant_set_J_linkage_average':    round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,75], 0) * 100, 5),
                    'dominant_set_J_linkage_weighted':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,76], 0) * 100, 5),
                    'dominant_set_J_linkage_centroid':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,77], 0) * 100, 5),
                    'dominant_set_J_linkage_median':     round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,78], 0) * 100, 5),
                    'dominant_set_J_linkage_ward':       round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,79], 0) * 100, 5),

                    'dominant_set_K_linkage_single':     round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,80], 0) * 100, 5),
                    'dominant_set_K_linkage_complete':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,81], 0) * 100, 5),
                    'dominant_set_K_linkage_average':    round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,82], 0) * 100, 5),
                    'dominant_set_K_linkage_weighted':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,83], 0) * 100, 5),
                    'dominant_set_K_linkage_centroid':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,84], 0) * 100, 5),
                    'dominant_set_K_linkage_median':     round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,85], 0) * 100, 5),
                    'dominant_set_K_linkage_ward':       round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,86], 0) * 100, 5),

                    'dominant_set_L_linkage_single':     round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,87], 0) * 100, 5),
                    'dominant_set_L_linkage_complete':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,88], 0) * 100, 5),
                    'dominant_set_L_linkage_average':    round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,89], 0) * 100, 5),
                    'dominant_set_L_linkage_weighted':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,90], 0) * 100, 5),
                    'dominant_set_L_linkage_centroid':   round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,91], 0) * 100, 5),
                    'dominant_set_L_linkage_median':     round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,92], 0) * 100, 5),
                    'dominant_set_L_linkage_ward':       round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,93], 0) * 100, 5)

                    # 'dominant_set_homogeneity_score': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,10]) * 100, 5),
                    # 'dominant_set_mr': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,11]) * 100, 5),
                    # 'dominant_set_randi': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,12]) * 100, 5),
                    # 'dominant_set_acp': round(eer_direct(result_file[list_key][:,0], result_file[list_key][:,13]) * 100, 5)
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
