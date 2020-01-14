# from random import randint, sample, choice
# from multiprocessing import Process, Queue
# from tqdm import tqdm

# import h5py
# import numpy as np
# import time
# import os
# import signal
# import sys

# QUEUE_SIZE = 5
# NUM_PROCESSES = 4

# class ParallelDataGenerator:
#     def __init__(self, batch_size=100, segment_size=40, spectrogram_height=128, config=None, dataset=None):
#         self.batch_size = batch_size
#         self.segment_size = segment_size
#         self.spectrogram_height = spectrogram_height
#         self.config = config
#         self.dataset = dataset

#         # Create handlers to stop threads in case of abort
#         # 
#         signal.signal(signal.SIGTERM, self.signal_terminate_queue)
#         signal.signal(signal.SIGINT, self.signal_terminate_queue)

#         self.start_queue()

#     def start_queue(self):
#         self.exit_process = False
#         self.train_queue = Queue(QUEUE_SIZE)
#         self.val_queue = Queue(QUEUE_SIZE)

#         self.processes = list()

#         main_process_id = os.getpid()

#         for i in range(NUM_PROCESSES):
#             mp_process = Process(target=self.sample_queue)
#             self.processes.append(mp_process)
            
#         # Ensure all processes are spawned before starting them
#         # such that no read is openend on the h5py file before forking
#         # 
#         # see https://groups.google.com/forum/#!topic/h5py/bJVtWdFtZQM
#         # 
#         for mp_process in self.processes:
#             mp_process.start()

#         if os.getpid() == main_process_id:
#             self.dataset.initialize()

#     def terminate_queue(self):
#         print('Stopping threads...')
#         self.exit_process = True
#         time.sleep(5)
#         print("\t5 seconds elapsed... kill threads.")
#         for i, mp_process in enumerate(self.processes):
#             print("\tTerminating process {}/{}...".format(i+1, NUM_PROCESSES),end='')
#             mp_process.terminate()
#             print('done')

#     # Function overload for signal API
#     # 
#     def signal_terminate_queue(self, signum, frame):
#         self.terminate_queue()

#     def sample_queue(self):
#         while not self.exit_process:
#             with h5py.File(self.config.get('train','dataset') + '.h5', 'r') as data:
#                 if self.train_queue.qsize() > self.val_queue.qsize():
#                     # The train_queue has more samples prepared => add to val_queue
#                     #
#                     try:
#                         Xb, yb = self.__get_batch__('val', data)
#                         self.val_queue.put([Xb, yb], timeout=0.5)
#                     except:
#                         pass

#                 else:
#                     # The val_queue has more samples prepared => add to train_queue
#                     # 
#                     try:
#                         Xb, yb = self.__get_batch__('train', data)
#                         self.train_queue.put([Xb, yb], timeout=0.5)
#                     except:
#                         pass

#     def __get_batch__(self, batch_type, data):
#         if batch_type not in ['train', 'val']:
#             raise ValueError(":batch_type has to be :train or :val.")
        
#         all_speakers = np.array(self.dataset.get_train_speaker_list())
#         num_speakers = all_speakers.shape[0]

#         Xb = np.zeros((self.batch_size, self.segment_size, self.spectrogram_height), dtype=np.float32)
#         yb = np.zeros(self.batch_size, dtype=np.int32)

#         train_statistics = self.dataset.get_train_statistics().copy()

#         for j in range(0, self.batch_size):
#             speaker_index = randint(0, num_speakers - 1)
#             speaker_name = all_speakers[speaker_index]

#             # Extract Spectrogram
#             # Choose from all the utterances of the given speaker randomly
#             # 
#             utterance_index = np.random.choice(train_statistics[speaker_name][batch_type])
            
#             # lehl@2019-12-13:
#             # If batches are generated as the statistics are updated due to active learning,
#             # it might be possible to draw from incides that are not really available, this
#             # is not a great solution, but quicker than ensuring these processes lock each other
#             #
#             full_spect = data['data/' + speaker_name][utterance_index]

#             # lehl@2019-12-03: Spectrogram needs to be reshaped with (time_length, 128) and then
#             # transposed as the expected ordering is (128, time_length)
#             # 
#             spect = full_spect.reshape((full_spect.shape[0] // self.spectrogram_height, self.spectrogram_height))

#             if not np.isnan(spect).any():
#                 # Standardize
#                 mu = np.mean(spect, 0, keepdims=True)
#                 stdev = np.std(spect, 0, keepdims=True)
#                 spect = (spect - mu) / (stdev + 1e-5)

#             # Extract random :segment_size long part of the spectrogram
#             # 
#             seg_idx = randint(0, spect.shape[0] - self.segment_size)
#             Xb[j] = spect[seg_idx:seg_idx + self.segment_size, :]

#             # Set label
#             # 
#             yb[j] = speaker_index

#         return Xb, np.eye(num_speakers)[yb]

#     def batch_generator(self, dataset_type):
#         if dataset_type == 'train':
#             queue = self.train_queue
#         elif dataset_type == 'val':
#             queue = self.val_queue
#         else:
#             raise ValueError(":dataset_type has to be :train or :val.")

#         while True:
#             [Xb, yb] = queue.get()

#             while np.isnan(Xb).any():
#                 [Xb, yb] = queue.get()

#             yield Xb, yb

#     def get_generator(self, generator):
#         gen = self.batch_generator(generator)
#         gen.__next__()

#         return gen