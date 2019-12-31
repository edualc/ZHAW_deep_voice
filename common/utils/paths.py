"""
A bunch of path related convenience methods that avoid usage of relative paths for the application.
Uses the os.path library to use the correct path separators ( \\ or / ) automatically.
"""
import re
import os
import os.path as path

import common.path_helper as common_helper
import networks.path_helper as networks_helper
import configs.path_helper as configs_helper


def join(base, *args):
    for arg in args:
        base = path.join(base, arg)

    return base

def get_common(*args):
    return join(common_helper.get_common_path(), *args)

def get_networks(*args):
    return join(networks_helper.get_networks_path(), *args)

def get_configs(*args):
    return join(configs_helper.get_configs_path(), *args)

def get_data(*args):
    return join(get_common('data'), *args)

def get_experiments(*args):
    return join(get_data('experiments'), *args)

def get_experiment_logs(*args):
    return join(get_experiments('logs'), *args)

def get_experiment_tensorboard_logs(*args):
    return join(get_experiments('tensorboard_logs'), *args)

def get_experiment_nets(*args):
    return join(get_experiments('nets'), *args)

def get_experiment_plots(*args):
    return join(get_experiments('plots'), *args)

def get_experiment_cluster(*args):
    return join(get_experiments('cluster'), *args)

def get_speaker_list(speaker):
    """
    Gets the absolute path to the speaker list of that name.
    :param speaker: the name (without .txt) of the file
    :return: the absolute path of the speakerlist
    """
    return get_common('data', 'speaker_lists', speaker + '.txt')

def get_evaluation_list(ident):
    return get_common('data', 'evaluation_lists', ident + '.txt')


def get_training(*args):
    return join(get_common('data', 'training'), *args)


def get_speaker_pickle(speaker, format='.pickle'):
    """
    Gets the absolute path to the speaker pickle of that name.
    :param speaker: the name (without .pickle) of the file
    :return: the absolute path of the speakers pickle
    """
    return get_training('speaker_pickles', speaker + format)

def get_results(*args):
    return join(get_experiments('results'), *args)

def get_results_intermediate_test(*args):
    return get_results('intermediate_test', *args)

def get_results_intermediate_analysis(*args):
    return get_results('intermediate_analysis', *args)

def get_result_pickle(network, format='.pickle'):
    """
    Gets the absolute path to the result pickle of that network.
    :param network: the name (without .pickle) of the file
    :return: the absolute path of the resut pickle
    """
    return get_results(network + format)


def get_result_files(filename, best):
    """
    Gets the absolute path to all results files containing a specific name and ends with "best" if the best option is set.
    :param filename: The name that the files should contain
    :param best: A boolean. If set to False the files that are found do not end with "best". If True the files found end with "best"
    :return: All absolute paths to a file that matches the criteria.
    """
    if best:
        regex = '^{}.*best.pickle'.format(filename)
    else:
        regex = '^{}.*(?<!best)\.pickle'.format(filename)

    files = list_all_files(get_results(), regex)
    for index, file in enumerate(files):
        files[index] = get_results(file)
    return files

def get_result_png(network):
    """
    Gets the absolute path to the result pickle of that network.
    :param network: the name (without .pickle) of the file
    :return: the absolute path of the resut pickle
    """
    return get_experiment_plots(network)

def list_all_files(directory, file_regex):
    """
    returns the filenames of all files in specified directory. The values are only the filename and NOT the fully qualified path
    :param directory: the absolut path to de directory
    :param file_regex: a String that the files should match (fnmatch.fnmatch(file, file_regex))
    :return: returns the filenames of all files in specified directory. The values are only the filename and NOT the fully qualified path
    """
    files = []
    for file in os.listdir(directory):
        if re.match(file_regex, file):
            files.append(file)
    return sorted(files)


def get_ivec_feature_path(list):
    return get_training('i_vector', list)
