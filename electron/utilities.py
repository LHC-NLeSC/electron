import numpy as np
from ROOT import TFile, RDataFrame


def load_data(filename : str):
    global columns
    kalman_file = TFile(filename)
    dataframe = RDataFrame("kalman_validator/kalman_ip_tree", kalman_file)
    return dataframe.AsNumpy(), dataframe.GetColumnNames()


def unpack_digit_indices(digit_indices):
    unpacked_digit_indices = []
    for index in range(6):
        column = np.ndarray(len(digit_indices), dtype=int)
        for column_index in range(len(digit_indices)):
            column[column_index] = digit_indices[column_index][index]
        unpacked_digit_indices.append(column)
    return unpacked_digit_indices


def shuffle_data(rng, data, labels):
    assert(len(data) == len(labels))
    permutation = rng.permutation(len(data))
    return data[permutation], labels[permutation]