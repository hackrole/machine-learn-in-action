# pylint: disable=missing-docstring,redefined-builtin,wildcard-import
# pylint: disable=unused-wildcard-import,
import operator

from numpy import *


def create_dataset():
    """
    create sample dataset
    """
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inx, dataset, labels, k):
    """
    use kNN math to classify data
    """
    dataset_size = dataset.shape[0]
    diffmat = tile(inx, (dataset_size, 1)) - dataset
    sq_diffmat = diffmat ** 2
    sq_distances = sq_diffmat.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_dist_indicies = distances.argsort()

    class_count = {}
    for i in range(k):
        votelabel = labels[sorted_dist_indicies[i]]
        class_count[votelabel] = class_count.get(votelabel, 0) + 1

    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]
