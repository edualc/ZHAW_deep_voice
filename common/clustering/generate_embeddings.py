import numpy as np
from tqdm import tqdm

from common.utils.logger import *

def generate_embeddings(outputs, speakers_inputs, vector_size):
    """
    Combines the utterances of the speakers in the train- and testing-set and combines them into embeddings.
    :param outputs:     Tuple of train_output (The training output, 80%) and test_output (The testing output, 20%)
    :param speakers_inputs:     Tuple of train_speakers (The speakers used in training) and test_speakers (The speakers used in testing)
    :param vector_size: The size which the output will have
    :return: embeddings, the speakers and the number of embeddings
    """
    logger = get_logger('clustering', logging.INFO)
    logger.info('Generate embeddings')
    num_speakers = len(set(speakers_inputs[0]))

    # Prepare return variable
    number_embeddings = len(outputs) * num_speakers
    embeddings = []
    speakers = []

    for output, speakers_input in zip(outputs, speakers_inputs):
        embeddings_output, speakers_output = _create_speaker_embeddings(num_speakers, vector_size, output, speakers_input)
        embeddings.extend(embeddings_output)
        speakers.extend(speakers_output)

    utterance_embeddings = generate_utterance_embeddings(
        np.concatenate((outputs[0],outputs[1])), np.concatenate((speakers_inputs[0],speakers_inputs[1])))

    return embeddings, speakers, number_embeddings, utterance_embeddings

def generate_utterance_embeddings(X, y):
    """
    Creates one embedding per utterance
    """
    num_speakers = len(set(y))

    # Check how many times the least represented class is present
    # and use this to draw the samples from (giving every speaker
    # the same amount of files to draw embeddings from during
    # pair creation)
    # 
    min_occurance = np.min(np.unique(y, return_counts=True)[1])

    # TODO: lehl@2019-12-07:
    # Check if min_occurance should be clamped at i.e. 100?
    # (Check with Vox1/Vox2 dataset sizes)
    # 

    embeddings = np.zeros((num_speakers, min_occurance, X.shape[1]))

    # Speakers are numbered/labeled from 0 to n-1
    # 
    for i in range(num_speakers):
        indices = np.where(y == i)[0]
        sampled_indices = np.random.choice(indices, min_occurance, replace=False)
        embeddings[i,:] = np.take(X, sampled_indices, axis=0)

    # Shape: <NumSpeakers, UtterancesPerSpeaker, EmbeddingLength>
    return embeddings

def _create_speaker_embeddings(num_speakers, vector_size, vectors, y):
    """
    Creates one embedding for each speaker in the vectors.
    :param num_speakers: Number of distinct speakers in this vector
    :param vector_size: Number of data in utterance
    :param vectors: The unordered speaker data
    :param y: An array that tells which speaker (number) is in which place of the vectors array
    :return: the embeddings per speaker and the speakers (numbers)
    """

    # Prepare return variables
    embeddings = np.zeros((num_speakers, vector_size))
    speakers = set(y)

    # Fill embeddings with utterances
    for i in range(0, num_speakers):

        # Fetch correct utterance
        utterance = embeddings[i]

        # Fetch values where same speaker and add to utterance
        indices = np.where(y == i)[0]
        
        outputs = np.take(vectors, indices, axis=0)
        for value in outputs:
            utterance = np.add(utterance, value)

        # Add filled utterance to embeddings
        embeddings[i] = np.divide(utterance, len(outputs))

    return embeddings, speakers
