from scipy.optimize import brentq
from scipy.interpolate import interp1d

from sklearn.metrics import roc_curve

import numpy as np

MAX_NUM_PAIRS_PER_SPEAKER = 100

def equal_error_rate(utterance_embeddings):
    # how many unique embeddings there are per speaker and how
    # many pairs can be created by combining all of them with each other
    # 
    embeddings_per_speaker = utterance_embeddings.shape[1]
    max_pairs_per_speaker = (embeddings_per_speaker * (embeddings_per_speaker - 1)) // 2
    num_speakers = utterance_embeddings.shape[0]

    possible_pairs = _calculate_err__calculate_possible_pairs(embeddings_per_speaker)
    num_pairs_to_create = _calculate_eer__get_num_pairs_to_create(max_pairs_per_speaker)

    true_scores = np.empty(0, dtype=np.float32)
    scores = np.empty(0, dtype=np.float32)

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
