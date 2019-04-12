import matplotlib

from common.analysis.acp import average_cluster_purity
from common.analysis.ari import adjusted_rand_index

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import *
from theano.gradient import np

from common.analysis.mr import misclassification_rate
from common.utils.logger import *
from common.utils.paths import *
from common.utils.pickler import load, save


'''
This list contains all metrics as tuple, containing the name, the function and the worst expected value (either 0 or 1).
All functions in this file are influenced by this list and any metric can be added as long as they expect y_true and
y_pred as their input.
'''
metrics = [
    ("MR", misclassification_rate, 1),
    ("ACP", average_cluster_purity, 0),
    ("ARI", adjusted_rand_index, 0),
    ("completeness_score", completeness_score, 0),
    ("homogeneity_score", homogeneity_score, 0)
]

def plot_files(plot_file_name, files):
    """
    Plots the results stored in the files given and stores them in a file with the given name
    :param plot_file_name: the file name stored in common/data/results
    :param files: a set of full file paths that hold result data
    """
    curve_names, metric_sets, set_of_number_of_embeddings = _read_result_pickle(files)

    _plot_curves(plot_file_name, curve_names, metric_sets, set_of_number_of_embeddings)


def _read_result_pickle(files):
    """
    Reads the results of a network from these files.
    :param files: can be 1-n files that contain a result.
    :return: curve names, thresholds, metric scores as a list and number of embeddings
    """
    logger = get_logger('analysis', logging.INFO)
    logger.info('Read result pickle')
    curve_names = []

    # Initialize result sets
    metric_sets_all_files = [[] for _ in metrics]
    set_of_number_of_embeddings = []

    # Fill result sets
    for file in files:
        curve_name, metric_sets, number_of_embeddings = load(file)

        for index, curve_name in enumerate(curve_name):
            for m, _ in enumerate(metrics):
                metric_sets_all_files[m].append(metric_sets[m][index])

            set_of_number_of_embeddings.append(number_of_embeddings[index])
            curve_names.append(curve_name)

    return curve_names, metric_sets_all_files, set_of_number_of_embeddings


def _plot_curves(plot_file_name, curve_names, metric_sets, number_of_embeddings):
    """
    Plots all specified curves and saves the plot into a file.
    :param plot_file_name: String value of save file name
    :param curve_names: Set of names used in legend to describe this curve
    :param metric_sets: A list of 2D matrices, each row of a metrics 2D matrix describes one dataset for a curve
    :param number_of_embeddings: set of integers, each integer describes how many embeddings is in this curve
    """
    logger = get_logger('analysis', logging.INFO)
    logger.info('Plot results')

    best_results = [[] for _ in metrics]
    for m, metric in enumerate(metrics):
        if(metric[2] == 0):
            best_results[m].append(np.max(metric_sets[m]))
        else:
            best_results[m].append(np.min(metric_sets[m]))

    '''
    This code is used to sort the lines by min mr. Because we now use mutliple metrics and dont sort by a single
    metric, this code is not used anymore, but we keep it for now.
    min_mrs, curve_names, mrs, acps, aris, homogeneity_scores, completeness_scores, number_of_embeddings = \
        (list(t) for t in
         zip(*sorted(zip(min_mrs, curve_names, mrs, acps, aris, homogeneity_scores, completeness_scores, number_of_embeddings))))
    '''

    # How many lines to plot
    number_of_lines = len(curve_names)

    # Get various colors needed to plot
    color_map = plt.get_cmap('gist_rainbow')
    colors = [color_map(i) for i in np.linspace(0, 1, number_of_lines)]

    # Define number of figures
    fig1 = plt.figure(1)
    fig1.set_size_inches(16, 16)

    # Define Plots
    plot_grid = (len(metrics),3)

    plots = [None] * 5

    for m, metric in enumerate(metrics):
        plots[m] = _add_cluster_subplot(plot_grid, (m, 0), metric[0], 2)

    # Define curves and their values
    curves = [[] for _ in metrics]

    for m, metric in enumerate(metrics):
        curves[m] = [plots[m], metric_sets[m]]

    # Plot all curves
    for index in range(number_of_lines):
        label = curve_names[index]
        for m, metric in enumerate(metrics):
            label = label + '\n {} {}: {}'.format('Max' if metric[2]==0 else 'Min', metric[0],
                                                  str(best_results[m][index]))
        color = colors[index]
        number_of_clusters = np.arange(number_of_embeddings[index], 0, -1)

        for plot, value in curves:
            plot.plot(number_of_clusters, value[index], color=color, label=label)

    # Add legend and save the plot
    fig1.legend()
    # fig1.show()
    fig1.savefig(get_result_png(plot_file_name))
    fig1.savefig(get_result_png(plot_file_name + '.svg'), format='svg')


def _add_cluster_subplot(grid, position, y_label, colspan=1):
    """
    Adds a cluster subplot to the current figure.

    :param grid: a tuple that contains number of rows as the first entry and number of columns as the second entry
    :param position: the position of this subplot
    :param y_label: the label of the y axis
    :param colspan: number of columns for the x axis, default is 1
    :return: the subplot itself
    """
    subplot = plt.subplot2grid(grid, position, colspan=colspan)
    subplot.set_ylabel(y_label)
    subplot.set_xlabel('number of clusters')
    return subplot


def analyse_results(network_name, checkpoint_names, set_of_predicted_clusters, set_of_true_clusters, embedding_numbers):
    """
    Analyses each checkpoint with the values of set_of_predicted_clusters and set_of_true_clusters.
    After the analysis the result are stored in the Pickle network_name.pickle and the best Result
    according to min MR is stored in network_name_best.pickle.

    :param network_name: The name for the result pickle.
    :param checkpoint_names: A list of names from the checkpoints. Later used as curvenames,
    :param set_of_predicted_clusters: A 2D array of the predicted Clusters from the Network. [checkpoint, clusters]
    :param set_of_true_clusters: A 2d array of the validation clusters. [checkpoint, validation-clusters]
    :param embeddings_numbers: A list which represent the number of embeddings in each checkpoint.
    """
    logger = get_logger('analysis', logging.INFO)
    logger.info('Run analysis')
    metric_sets = [[None] * len(set_of_predicted_clusters) for _ in range(len(metrics))]

    for index, predicted_clusters in enumerate(set_of_predicted_clusters):
        logger.info('Analysing checkpoint:' + checkpoint_names[index])

        metric_results = _calculate_analysis_values(predicted_clusters, set_of_true_clusters[index])

        for m, _ in enumerate(metrics):
            metric_sets[m][index] = metric_results[m]

    _write_result_pickle(network_name, checkpoint_names, metric_sets, embedding_numbers)
    _save_best_results(network_name, checkpoint_names, metric_sets, embedding_numbers)
    logger.info('Analysis done')


def _calculate_analysis_values(predicted_clusters, true_cluster):
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
    metric_results = [None] * len(metrics)
    for m, metric in enumerate(metrics):
        if metric[2] == 1:
            metric_results[m] = np.ones(len(true_cluster))
        else:
            metric_results[m] = np.zeros((len(true_cluster)))

    # Loop over all possible clustering
    for i, predicted_cluster in enumerate(predicted_clusters):
        # Calculate different analysis's
        for m, metric in enumerate(metrics):
            metric_function = metric[1]
            metric_results[m][i] = metric_function(true_cluster, predicted_cluster)

    return metric_results


def _save_best_results(network_name, checkpoint_names, metric_sets, speaker_numbers):
    if len(metric_sets[0]) == 1:
        _write_result_pickle(network_name + "_best", checkpoint_names, metric_sets, speaker_numbers)
    else:
        # Find best result (according to the first metric in metrics)
        if(metrics[0][2] == 1):
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
        set_of_best_metrics = [[] for _ in metrics]
        best_speaker_numbers = []

        for index, best_result in enumerate(best_results):
            if best_result == best_result_over_all:
                best_checkpoint_name.append(checkpoint_names[index])
                for m, metric in enumerate(metrics):
                    set_of_best_metrics[m].append(metric_sets[m][index])
                best_speaker_numbers.append(speaker_numbers[index])

        _write_result_pickle(network_name + "_best", best_checkpoint_name, set_of_best_metrics, best_speaker_numbers)


def _write_result_pickle(network_name, checkpoint_names, metric_sets, number_of_embeddings):
    logger = get_logger('analysis', logging.INFO)
    logger.info('Write result pickle')
    save((checkpoint_names, metric_sets, number_of_embeddings), get_result_pickle(network_name))


def _read_and_safe_best_results():
    checkpoint_names, metric_sets, speaker_numbers = _read_result_pickle([get_result_pickle('flow_me')])
    _save_best_results('flow_me', checkpoint_names, metric_sets, speaker_numbers)


if __name__ == '__main__':
    _read_and_safe_best_results()
