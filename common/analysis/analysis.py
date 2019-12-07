from common.analysis.metrics.acp import average_cluster_purity
from common.analysis.metrics.ari import adjusted_rand_index
from common.analysis.metrics.der import diarization_error_rate
import numpy as np

from scipy.optimize import brentq
from scipy.interpolate import interp1d

from sklearn.metrics import roc_curve

from common.analysis.metrics.mr import misclassification_rate
from common.utils.logger import *
from common.utils.paths import *
from common.utils.pickler import load, save

metric_names = ["MR", "ACP", "ARI", "DER"]
metric_worst_values = [1,0,0,1]
MAX_NUM_PAIRS_PER_SPEAKER = 100


def analyse_results(network_name, checkpoint_names, set_of_predicted_clusters,
                    set_of_true_clusters, embedding_numbers, set_of_times, set_of_utterance_embeddings):
    """
    Analyses each checkpoint with the values of set_of_predicted_clusters and set_of_true_clusters.
    After the analysis the result are stored in the Pickle network_name.pickle and the best Result
    according to min MR is stored in network_name_best.pickle.
    :param network_name: The name for the result pickle.
    :param checkpoint_names: A list of names from the checkpoints. Later used as curvenames,
    :param set_of_predicted_clusters: A 2D array of the predicted Clusters from the Network. [checkpoint, clusters]
    :param set_of_true_clusters: A 2d array of the validation clusters. [checkpoint, validation-clusters]
    :param embedding_numbers: A list which represent the number of embeddings in each checkpoint.
    :param set_of_times: A 2d array of the time per utterance [checkpoint, times]
    """

    logger = get_logger('analysis', logging.INFO)
    logger.info('Run analysis')
    metric_sets = [[None] * len(set_of_predicted_clusters) for _ in range(len(metric_names))]

    for index, predicted_clusters in enumerate(set_of_predicted_clusters):
        checkpoint = checkpoint_names[index]
        logger.info('Analysing checkpoint:' + checkpoint)

        # Check if checkpoint is already stored
        analysis_pickle = get_results_intermediate_analysis(checkpoint)

        if os.path.isfile(analysis_pickle):
            (metric_results, eer_result) = load(analysis_pickle)
        else:
            metric_results = _calculate_analysis_values(predicted_clusters, set_of_true_clusters[index], set_of_times[index])
            eer_result = _calculate_eer_result(set_of_utterance_embeddings[index])
            save((metric_results, eer_result), analysis_pickle)

        print('\tModel: {}, EER: {}'.format(checkpoint, round(eer_result,5)))

        for m, metric_result in enumerate(metric_results):
            metric_sets[m][index] = metric_result

    _write_result_pickle(network_name, checkpoint_names, metric_sets, embedding_numbers)
    _save_best_results(network_name, checkpoint_names, metric_sets, embedding_numbers)

    logger.info('Clearing intermediate result checkpoints')
    
    for checkpoint in checkpoint_names:
        analysis_pickle = get_results_intermediate_analysis(checkpoint)
        test_pickle = get_results_intermediate_test(checkpoint)

        if os.path.exists(analysis_pickle):
            os.remove(analysis_pickle)

        if os.path.exists(test_pickle):
            os.remove(test_pickle)

    logger.info('Analysis done')

#   utterance_embeddings         <NumSpeakers, UtterancesPerSpeaker, EmbeddingLength>
def _calculate_eer_result(utterance_embeddings):
    # how many unique embeddings there are per speaker and how
    # many pairs can be created by combining all of them with each other
    # 
    embeddings_per_speaker = utterance_embeddings.shape[1]
    max_pairs_per_speaker = (embeddings_per_speaker * (embeddings_per_speaker - 1)) // 2
    num_speakers = utterance_embeddings.shape[0]

    possible_pairs = _calculate_err__calculate_possible_pairs(embeddings_per_speaker)
    num_pairs_to_create = _calculate_eer__get_num_pairs_to_create(max_pairs_per_speaker)

    true_scores = np.empty(1)
    scores = np.empty(1)

    # Generate pairs for speakers
    # 
    for true_speaker_id in range(num_speakers):
        speaker_embeddings = utterance_embeddings[true_speaker_id,:,:]

        # Generate POSITIVE Pairs
        # 
        pos_pairs = _calculate_eer__get_pairs(possible_pairs, num_pairs_to_create)
        pos_scores = _calculate_err__positive_scores_for_speaker(speaker_embeddings, pos_pairs)

        scores = np.concatenate([scores, pos_scores])
        true_scores = np.concatenate([true_scores, np.ones(pos_scores.shape)])

        # Generate NEGATIVE Pairs
        # 
        neg_pairs = _calculate_eer__get_pairs(possible_pairs, num_pairs_to_create)
        false_speaker_ids = _calculate_eer__get_false_speaker_ids(num_speakers, true_speaker_id, neg_pairs.shape[0])
        neg_scores = _calculate_err__negative_scores_for_speaker(utterance_embeddings, neg_pairs, true_speaker_id, false_speaker_ids)
        
        # Update Score Arrays (NEGATIVE)
        # 
        scores = np.concatenate([scores, neg_scores])
        true_scores = np.concatenate([true_scores, np.zeros(neg_scores.shape)])
        
    # EER Calculation
    fpr, tpr, thresholds = roc_curve(true_scores, scores, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    
    return eer

def _calculate_err__positive_scores_for_speaker(speaker_embeddings, pairs):
    emb_A = speaker_embeddings[pairs[:,0]]
    emb_B = speaker_embeddings[pairs[:,1]]

    return _calculate_err__calculate_similarity(emb_A, emb_B)

def _calculate_err__negative_scores_for_speaker(utterance_embeddings, pairs, true_speaker_id, false_speaker_ids):
    true_embedding = utterance_embeddings[true_speaker_id, pairs[:,0], :]
    false_embedding = utterance_embeddings[false_speaker_ids, pairs[:,1], :]

    return _calculate_err__calculate_similarity(true_embedding, false_embedding)

def _calculate_err__calculate_similarity(embedding_A, embedding_B):
    return np.sum(embedding_A * embedding_B, axis=1)

# Creates an array containing all the index unique index pairs of possible
# pairs for this speaker without identity pairs (A,A). This generates the lower
# diagonal matrix of utterance index combinations and creates the pairs from
# these indices of the lower triangular matrix
# 
def _calculate_err__calculate_possible_pairs(embeddings_per_speaker):
    lower_triangular_matrix = np.tril(np.ones((embeddings_per_speaker,embeddings_per_speaker)),-1)
    return np.dstack(np.where(lower_triangular_matrix==1))[0]

# Returns a numpy array containing the speaker_ids to be used for the negative pair
# creation, which can be any speaker id from 0 to num_speakers excluding the current
# speaker_id (=exclusion)
#  
def _calculate_eer__get_false_speaker_ids(num_speakers, exclusion, amount):
    return np.random.choice(np.delete(np.arange(num_speakers), exclusion), amount)

# How many pairs will be generated? As many as there are
# but not more than MAX_NUM_PAIRS_PER_SPEAKER
# 
def _calculate_eer__get_num_pairs_to_create(max_pairs_per_speaker):
    if max_pairs_per_speaker > MAX_NUM_PAIRS_PER_SPEAKER:
        return MAX_NUM_PAIRS_PER_SPEAKER
    else:
        return max_pairs_per_speaker

def _calculate_eer__get_pairs(possible_pairs, amount):
    indices = np.random.choice(np.arange(possible_pairs.shape[0]), amount, replace=False)
    return np.take(possible_pairs, indices, axis=0)


def _calculate_analysis_values(predicted_clusters, true_cluster, times):
    """
    Calculates the analysis values out of the predicted_clusters.

    :param predicted_clusters: The predicted Clusters of the Network.
    :param true_clusters: The validation clusters
    :return: the results of all metrics as a 2D array where i is the index of the metric and j is the index of a
        specific result

    """
    logger = get_logger('analysis', logging.INFO)
    logger.info('Calculate scores')

    # Initialize output
    metric_results = [None] * len(metric_names)
    for m, min_value in enumerate(metric_worst_values):
        if min_value == 1:
            metric_results[m] = np.ones(len(true_cluster))
        else:
            metric_results[m] = np.zeros((len(true_cluster)))

    # Loop over all possible clustering
    for i, predicted_cluster in enumerate(predicted_clusters):
        # logger.info('Calculated Scores for {}/{} predicted clusters'.format(i+1, len(predicted_clusters)))

        # Calculate different analysis's
        metric_results[0][i] = misclassification_rate(true_cluster, predicted_cluster)
        metric_results[1][i] = average_cluster_purity(true_cluster, predicted_cluster)
        metric_results[2][i] = adjusted_rand_index(true_cluster, predicted_cluster)
        metric_results[3][i] = diarization_error_rate(true_cluster, predicted_cluster, times)

    return metric_results


def _save_best_results(network_name, checkpoint_names, metric_sets, speaker_numbers):
    if len(metric_sets[0]) == 1:
        _write_result_pickle(network_name + "_best", checkpoint_names, metric_sets, speaker_numbers)
    else:
        # Find best result (according to the first metric in metrics)
        if metric_worst_values[0] == 1:
            best_results = []
            for results in metric_sets[0]:
                best_results.append(np.min(results))
            best_result_over_all = min(best_results)
        else:
            best_results = []
            for results in metric_sets[0]:
                best_results.append(np.max(results))
            best_result_over_all = max(best_results)

        best_checkpoint_name = []
        set_of_best_metrics = [[] for _ in metric_sets]
        best_speaker_numbers = []

        for index, best_result in enumerate(best_results):
            if best_result == best_result_over_all:
                best_checkpoint_name.append(checkpoint_names[index])
                for m, metric_set in enumerate(metric_sets):
                    set_of_best_metrics[m].append(metric_set[index])
                best_speaker_numbers.append(speaker_numbers[index])

        _write_result_pickle(network_name + "_best", best_checkpoint_name, set_of_best_metrics, best_speaker_numbers)


def _write_result_pickle(network_name, checkpoint_names, metric_sets, number_of_embeddings):
    logger = get_logger('analysis', logging.INFO)
    logger.info('Write result pickle')
    save((checkpoint_names, metric_sets, number_of_embeddings), get_result_pickle(network_name))
