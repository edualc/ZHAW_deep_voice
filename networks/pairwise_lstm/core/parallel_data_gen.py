from random import randint, sample, choice
from multiprocessing import Process, Queue
from tqdm import tqdm

import numpy as np
import time

QUEUE_SIZE = 20

class ParallelDataGenerator:
    def __init__(self, batch_size=100, segment_size=40, spectrogram_height=128, config, dataset, generator_type):
        self.batch_size = batch_size
        self.segment_size = segment_size
        self.spectrogram_height = spectrogram_height
        self.config = config
        self.dataset = dataset
        
        if generator_type not in ['train','val']:
            raise ValueError(":generator_type must be :train, :al or :val.")
        else:
            self.generator_type = generator_type

        self.start_queue()

    def start_queue(self):
        self.train_queue = Queue(QUEUE_SIZE)
        self.val_queue = Queue(QUEUE_SIZE)

        self.enqueuers = list()

        for i in range(8):
            thread = Process(target=self.sample_queue)
            self.enqueuers.append(thread)
            
            thread.start()

        self.exit_process = False

    def terminate_queue(self):
        print('Stopping queue...', end='')
        self.exit_process = True
        time.sleep(5)
        for i, thread in enumerate(self.enqueuers):
            print("Terminating thread {}...".format(i),end='')
            thread.terminate()
            print('done')
        print('done')

    def sample_queue(self):
        # data_file = self.dataset.get_train_file()
        
        while not self.exit_process:
            if self.train_queue.qsize() > self.val_queue.qsize():
                # The train_queue has more samples prepared => add to val_queue
                #
                Xb, yb = self.__get_batch__('val')
                self.val_queue.put([Xb, yb])

            else:
                # The val_queue has more samples prepared => add to train_queue
                # 
                Xb, yb = self.__get_batch__('train')
                self.train_queue.put([Xb, yb])

    def __get_batch__(self, batch_type):
        if batch_type not in ['train', 'val']:
            raise ValueError(":batch_type has to be :train or :val.")
        
        all_speakers = np.array(dataset.get_train_speaker_list())
        num_speakers = all_speakers.shape[0]

        Xb = np.zeros((self.batch_size, self.segment_size, self.spectrogram_height), dtype=np.float32)
        yb = np.zeros(self.batch_size, dtype=np.int32)

        for j in range(0, self.batch_size):
            speaker_index = randint(0, num_speakers - 1)
            speaker_name = all_speakers[speaker_index]

            # Extract Spectrogram
            # Choose from all the utterances of the given speaker randomly
            # 
            utterance_index = np.random.choice(dataset.get_train_statistics()[speaker_name][batch_type])
            
            # lehl@2019-12-13:
            # If batches are generated as the statistics are updated due to active learning,
            # it might be possible to draw from incides that are not really available, this
            # is not a great solution, but quicker than ensuring these processes lock each other
            #
            full_spect = dataset.get_train_file()['data/' + speaker_name][utterance_index]

            # lehl@2019-12-03: Spectrogram needs to be reshaped with (time_length, 128) and then
            # transposed as the expected ordering is (128, time_length)
            # 
            spect = full_spect.reshape((full_spect.shape[0] // spectrogram_height, spectrogram_height))

            # Standardize
            mu = np.mean(spect, 0, keepdims=True)
            stdev = np.std(spect, 0, keepdims=True)
            spect = (spect - mu) / (stdev + 1e-5)

            # Extract random :segment_size long part of the spectrogram
            # 
            seg_idx = randint(0, spect.shape[1] - segment_size)
            Xb[j] = spect[seg_idx:seg_idx + segment_size, :]

            # Set label
            # 
            yb[j] = speaker_index

        return Xb, yb

    def batch_generator(self, dataset_type):
        if dataset_type == 'train':
            queue = self.train_queue
        elif dataset_type == 'val':
            queue = self.val_queue
        else:
            raise ValueError(":dataset_type has to be :train or :val.")
        
        num_speakers = np.array(self.dataset.get_train_speaker_list()).shape[0]

        while True
            [Xb, yb] = queue.get()

            return Xb, np.eye(num_speakers)[yb]

    def get_generator(self, generator):
        gen = self.batch_generator(generator)
        gen.__next__()

        return gen