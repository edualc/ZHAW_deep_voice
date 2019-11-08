"""
The factory to create all used speaker pickles in the networks.

Based on previous work of Gerber, Lukic and Vogt, adapted by Heusser
"""
import sys

from common.extrapolation.speaker import Speaker


# lehmacl1: defines which speakers are being set up initially
#
def create_all_speakers(dataset):
    """
    A generator that yields all Speakers that are needed for the Speaker Clustering Suite to function
    :return: yields Speakers
    """
    if dataset == 'timit':
        yield Speaker(True, 40, 'timit_speakers_40_dev_test', 'timit')
        yield Speaker(True, 40, 'timit_speakers_40_final_test', 'timit')
        yield Speaker(True, 60, 'timit_speakers_60_dev_test', 'timit')
        yield Speaker(True, 60, 'timit_speakers_60_final_test', 'timit')
        yield Speaker(True, 80, 'timit_speakers_80_dev_test', 'timit')
        yield Speaker(True, 80, 'timit_speakers_80_final_test', 'timit')
        yield Speaker(False, 100, 'timit_speakers_100_50w_50m_not_reynolds', 'timit')
        yield Speaker(False, 470, 'timit_speakers_470_stratified', 'timit')
        yield Speaker(True, 590, 'timit_speakers_590_clustering_without_raynolds', 'timit')

    # VoxCeleb2 Speakers
    # lehmacl1@2019-04-16: Since some of the voxceleb2 files are >1min long, setting a high
    # max_audio_length might add huge amounts of "empty" spectrograms with useless information.
    # Trying ~8sec (800) to see if it helps reduce "nonsense audio"
    # 
    elif dataset == 'vox2':
        yield Speaker(False, 10, "vox2_speakers_10_test", dataset="voxceleb2", max_audio_length=800, max_files_per_partition=10)
        yield Speaker(False, 5994, "vox2_speakers_5994_dev", dataset="voxceleb2", max_audio_length=800, max_files_per_partition=12000)
        yield Speaker(False, 120, "vox2_speakers_120_test", dataset="voxceleb2", max_audio_length=800)

    elif dataset == 'synthetic':
        # ZEROS
        # This dataset is used as a baseline to check if the network is unable to learn from zeros, 
        # meaning no usable data. It is expected to randomly guess between two classes and achieve
        # about 50% accuracy.
        # 
        yield Speaker(False, 2, "synthetic_zeros", dataset="synthetic")
        
        # OVERFIT
        # This dataset consists of two classes which are easy to seperate and should quickly achieve
        # an accuracy of 100%. :overfit has actual "random" numbers while the :overfit_simple is comparing
        # numpy matrices of zeros and ones against each other.
        # 
        yield Speaker(False, 2, "synthetic_overfit", dataset="synthetic")
        yield Speaker(False, 2, "synthetic_overfit_simple", dataset="synthetic")
    
    else:
        print("Dataset " + dataset + " is not known.")
        sys.exit(1)
