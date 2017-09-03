# pylint: disable=missing-docstring,redefined-builtin,wildcard-import
# pylint: disable=unused-wildcard-import,
import os
import operator

import matplotlib
from matplotlib import pyplot as plt
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


def file2matrix(filename):
    """
    read data from text file
    """
    fp = open(filename)
    lines = fp.readlines()
    line_num = len(lines)

    index = 0
    data = zeros((line_num, 3))
    labels = []
    for line in lines:
        line = line.strip()
        columns = line.split('\t')
        print('>>>', columns)
        data[index, :] = columns[0:3]
        labels.append(int(columns[-1]))
        index += 1

    return data, labels


def test_matplot():
    data, labels = file2matrix('./datingTestSet2.txt')
    figure = plt.figure()
    plot = figure.add_subplot(111)
    # plot.scatter(data[:, 1], data[:, 2])
    # use colorful plot
    plot.scatter(data[:, 1], data[:, 2], 15.0 * array(labels), 15.0 * array(labels))
    plt.show()


def norm(dataset):
    """norm dataset"""
    min_vals = dataset.min(0)
    max_vals = dataset.max(0)
    ranges = max_vals - min_vals

    norm_dataset = zeros(shape(dataset))
    shape1 = dataset.shape[0]
    norm_dataset = dataset - tile(min_vals, (shape1, 1))
    norm_dataset = norm_dataset / tile(ranges, (shape1, 1))

    return norm_dataset, ranges, min_vals


def test_dating():
    """
    test the dating ML
    """
    radio = 0.10
    data, labels = file2matrix('./datingTestSet2.txt')
    norm_data, _, _ = norm(data)

    shape1 = norm_data.shape[0]
    test_num = int(shape1 * radio)
    error_count = 0.0

    for i in range(test_num):
        result = classify0(norm_data[i, :],
                           norm_data[test_num:shape1, :],
                           labels[test_num:shape1], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (
            result, labels[i]
        ))
        if result != labels[i]:
            error_count += 1.0

    print("total error rate is %f" % (error_count / float(test_num)))


def classify_person():
    results = ['not', 'small', 'large']

    tats = float(input("time of video game?"))
    mile = float(input("filer mile per year?"))
    cream = float(input("ice cream per year?"))

    data, labels = file2matrix("./datingTestSet2.txt")
    norm_data, ranges, min_vals = norm(data)

    inarr = array([mile, tats, cream])
    result = classify0((inarr - min_vals) / ranges, norm_data, labels, 3)
    print("you will probably like this person: ", results[result - 1])


def img2vector(filename):
    """
    turn image to vector array
    """
    vect = zeros((1, 1024))
    fp = open(filename)
    for i in range(32):
        line = fp.readline()
        for j in range(32):
            vect[0, 32*i + j] = int(line[j])

    return vect


def test_handlewrite():
    """
    handle write ML
    """
    labels = []
    train_files = os.listdir("./trainingDigits")
    train_num = len(train_files)
    train_data = zeros((train_num, 1024))

    for i in range(train_num):
        filename = train_files[i]
        name = filename.split('.')[0]
        label = int(name.split('_')[0])

        labels.append(label)
        train_data[i, :]  = img2vector('./trainingDigits/%s' % filename)

    test_files = os.listdir("./testDigits")
    error_count = 0.0
    test_num = len(test_files)

    for i in range(test_num):
        filename = test_files[i]
        name = filename.split('.')[0]
        label = int(name.split('_')[0])
        data = img2vector('./testDigits/%s' % filename)

        result = classify0(data, train_data, labels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (
            result, label))
        if result != label:
            error_count += 1.0

    print("the total error num is %d" % error_count)
    print("the error rate is %f" % (error_count / test_num))
