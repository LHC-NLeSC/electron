import numpy as np
from ROOT import TFile, RDataFrame


def load_data(filename : str):
    kalman_file = TFile(filename)
    dataframe = RDataFrame("kalman_validator/kalman_ip_tree", kalman_file)
    dataframe = dataframe.Define("p", "abs(1.f/best_qop)")
    dataframe = dataframe.Define("ep", "ecal_energy/p")
    return dataframe.AsNumpy(), dataframe.GetColumnNames()


def shuffle_data(rng, data, labels):
    assert(len(data) == len(labels))
    permutation = rng.permutation(len(data))
    return data[permutation], labels[permutation]


def threshold_method(data, labels, threshold=0.7):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    for index in range(0, len(data)):
        positive = False
        if data[index][6] > threshold:
            positive = True
        if positive and labels[index][0] == 1:
            true_positives = true_positives + 1
        elif not positive and labels[index][0] == 0:
            true_negatives = true_negatives + 1
        elif positive and labels[index][0] == 0:
            false_positives = false_positives + 1
        elif not positive and labels[index][0] == 1:
            false_negatives = false_negatives + 1
    return (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)