import h5py
import numpy as np
import time
import copy

from common.utils.paths import *
from tqdm import tqdm

class DeepVoiceDataset():
  def __init__(self, config):
      self.config = config
      self.data = dict()

      self.initialized = False
      self.initialize()

  def initialize(self):
    if not self.initialized:
      self.initialized = True
      
      # Initialize Statistics
      # 
      self.get_train_statistics()
      self.get_test_statistics()

  def get_train_speaker_list(self):
    return self.get_speaker_list('train')
    
  def get_test_speaker_list(self):
    return self.get_speaker_list('test')

  def get_speaker_list(self, file_type):
    self.__check_file_type(file_type)

    try:
      return self.data[file_type + '_speaker_list']
    except KeyError:
      return self.__get_speaker_liste__initialize(file_type)

  def __get_speaker_liste__initialize(self, file_type):
    speaker_list_path = get_speaker_list(self.config.get(file_type,'speaker_list'))
    self.data[file_type + '_speaker_list'] = [line.rstrip('\n') for line in open(speaker_list_path)] 

    return self.data[file_type + '_speaker_list']

  def get_train_file(self):
    return self.get_file('train')
    
  def get_test_file(self):
    return self.get_file('test')

  # Returns a NEW file handler to the dataset file
  # 
  def get_file(self, file_type):
    self.__check_file_type(file_type)
    return self.__get_file__by_type(file_type)

  # Due to a problem when forking the main process, the file handler
  # should only be open for as long as necessary and NOT constantly.
  # 
  # see https://groups.google.com/forum/#!topic/h5py/bJVtWdFtZQM
  # 
  def __get_file__by_type(self, file_type):
    dataset_path = self.config.get(file_type,'dataset') + '.h5'
    return h5py.File(dataset_path, 'r')

  def get_train_statistics(self):
    return self.get_statistics('train')
    
  def get_test_statistics(self):
    return self.get_statistics('test')

  def get_statistics(self, file_type):
    self.__check_file_type(file_type)

    try:
      return self.data[file_type + '_statistics']
    except KeyError:
      if file_type == 'train':
        return self.__get_statistics__initialize_train()
      elif file_type == 'test':
        return self.__get_statistics__initialize_test()
      else:
        pass

  def __get_statistics__initialize_train(self):
    data_file = self.get_file('train')
    statistics = dict()

    for speaker in tqdm(self.get_speaker_list('train'), ascii=True, ncols=100, desc='split training and validation data'):
      # Files for each speaker are split into training and validation,
      # while - with active learning enabled - the training is further
      # split into training and al, such that during the active learning
      # the training data is increased from the al-pool
      # 
      # The split is done according to the :validation_share and
      # :active_learning_share values of the configuration
      # 
      train_indices = list()
      al_indices = list()
      val_indices = list()

      # How much of the training dataset needs to be reserver for validation
      # (and from the leftover: how much for active learning?)
      # 
      val_share = self.config.getfloat('train','validation_share')
      al_share = self.__get_statistics__get_active_learning_share()

      # Calculate base statistics for speaker
      # 
      total_speaker_time = np.sum(data_file['statistics/'+ speaker])
      cumulative_sum = np.cumsum(data_file['statistics/'+ speaker])

      # Calculate which is the first index of files that is, an 
      # extra +1 is added to include the cut_off_index file in the training,
      # otherwise it would be "AT LEAST" validation_share of the dataset
      # in the validation set
      # 
      val_cut_off_index = np.where(cumulative_sum > total_speaker_time * (1 - val_share))[0][0] + 1

      train_indices = np.arange(val_cut_off_index)
      val_indices = np.arange(val_cut_off_index, data_file['statistics/' + speaker].shape[0]) 
      val_share = np.sum(data_file['statistics/'+ speaker][val_cut_off_index:] / total_speaker_time)

      # Do we need to further split the training data for active learning?
      if self.config.getboolean('active_learning','enabled'):
        al_cut_off_index = np.where(cumulative_sum > total_speaker_time * (1 - val_share) * (1 - al_share))[0][0] + 1

        train_indices = np.arange(al_cut_off_index)
        al_indices = np.arange(al_cut_off_index, val_cut_off_index)

        al_share = np.sum(data_file['statistics/'+ speaker][al_cut_off_index:val_cut_off_index] / total_speaker_time)
        train_share = 1.0 - val_share - al_share

      else:
        al_share = 0.0
        train_share = 1.0 - val_share

      # Collect the speaker statistics
      # 
      statistics[speaker] = {
        'train': train_indices, 'train_share': train_share,
        'al': al_indices, 'al_share': al_share,
        'val': val_indices, 'val_share': val_share
      }

    # Close h5py Connection
    data_file.close()

    # Set the local variable
    # 
    self.data['train_statistics'] = statistics
    return self.data['train_statistics']

  def __get_statistics__initialize_test(self):
    data_file = self.get_file('test')
    statistics = dict()

    for speaker in tqdm(self.get_speaker_list('test'), ascii=True, ncols=100, desc='prepare test data'):
      num_utterances = data_file['statistics/' + speaker].shape[0]
      cut_off = int(num_utterances * (1 - self.config.getfloat('test','short_split')))

      statistics[speaker] = {
        'all': np.arange(num_utterances),
        'long': np.arange(cut_off),
        'short': np.arange(cut_off, num_utterances)
      }

    # Close h5py Connection
    data_file.close()
    
    # Set the local variable
    # 
    self.data['test_statistics'] = statistics
    return self.data['test_statistics']

  def __get_statistics__get_active_learning_share(self):
    if self.__check_is_active_learning_enabled():
      return self.config.getfloat('active_learning','active_learning_share')
    else:
      return 0.0

  # Returns the amount of spectrograms available for the
  # given train_type (across all speakers)
  # 
  def get_train_num_segments(self, train_type):
    self.__check_train_type(train_type)

    return sum(
      map(
        lambda x: x.shape[0],
        map(
          lambda y: self.get_train_statistics()[y][train_type],
          self.get_train_statistics()
        )
      )
    )

  # Returns the amount of spectrograms available during testing
  # 
  def get_test_num_segments(self, test_type):
    self.__check_test_type(test_type)

    return sum(
      map(
        lambda x: x.shape[0],
        map(
          lambda y: self.get_test_statistics()[y][test_type],
          self.get_test_statistics()
        )
      )
    )

  def update_active_learning_share(self, utterances_change):
    statistics = copy.deepcopy(self.get_train_statistics())

    for row in utterances_change:
      speaker = row[1]
      utterance_id = int(row[2])

      # Add utterance to train set
      statistics[speaker]['train'] = np.append(statistics[speaker]['train'], utterance_id)

      # Remove utterance from active learning set
      statistics[speaker]['al'] = np.delete(statistics[speaker]['al'], np.where(statistics[speaker]['al'] == utterance_id), 0)

    # Overwrite existing statistics
    self.data['train_statistics'] = statistics

  def __check_is_active_learning_enabled(self):
    return self.config.getboolean('active_learning','enabled')

  # PRIVATE method to check if the given file_type is either
  # train or test
  # 
  def __check_file_type(self, file_type):
    if file_type in ['train','test']:
      return True

    else:
      raise AttributeError('Unknown :file_type, should be "train" or "test".')

  def __check_train_type(self, train_type):
    if train_type in ['train','val','al']:
      return True

    else:
      raise AttributeError('Unknown :train_type, should be "train", "al" or "val".')

  def __check_test_type(self, test_type):
    if test_type in ['all','short','long']:
      return True

    else:
      raise AttributeError('Unknown :test_type, should be "all", "short" or "long".')
