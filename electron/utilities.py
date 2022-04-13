import numpy as np
from ROOT import TFile, RDataFrame


def load_data(filename : str):
    global columns
    kalman_file = TFile(filename)
    dataframe = RDataFrame("kalman_validator/kalman_ip_tree", kalman_file)
    return dataframe.AsNumpy(), dataframe.GetColumnNames()


def unpack_digit_indices(digit_indices, columns:int=6, filter:bool=False):
    unpacked_digit_indices = np.ndarray((columns, len(digit_indices)))
    for index in range(columns):
        for column_index in range(len(digit_indices)):
            unpacked_digit_indices[index][column_index] = digit_indices[column_index][index]
    if filter:
        to_filter = []
        for index in range(len(unpacked_digit_indices)):
            if len(np.unique(unpacked_digit_indices[index])) == 1:
                print(f"Column {index} is being filtered out.")
                to_filter.append(index)
        unpacked_digit_indices = np.delete(unpacked_digit_indices, to_filter, axis=0)
    return unpacked_digit_indices


def shuffle_data(rng, data, labels):
    assert(len(data) == len(labels))
    permutation = rng.permutation(len(data))
    return data[permutation], labels[permutation]