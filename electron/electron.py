import argparse
import numpy as np
import tensorflow as tf
from os import environ
from ROOT import TFile, RDataFrame

# reduce TensorFlow verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# columns in the dataframe
columns = None


def command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="ROOT file containing the training data set", type=str, required=True)
    return parser.parse_args()


def load_data(filename : str):
    global columns
    kalman_file = TFile(filename)
    dataframe = RDataFrame("kalman_validator/kalman_ip_tree", kalman_file)
    columns = dataframe.GetColumnNames()
    return dataframe.AsNumpy()


def shuffle_data(rng, data, labels):
    assert(len(data) == len(labels))
    permutation = rng.permutation(len(data))
    return data[permutation], labels[permutation]


def __main__():
    arguments = command_line()
    dataframe = load_data(arguments.filename)
    print(f"Columns in the table: {len(dataframe)}")
    if "mcp_electron" not in columns:
        print("Missing labels.")
        return
    labels = dataframe["mcp_electron"].astype(int)
    if "digit_indices" not in columns:
        print("Missing training data.")
        return
    data = dataframe["digit_indices"]
    print(f"Number of entries: {len(data)}")
    data_electron = data[labels == 1]
    data_other = data[labels == 0]
    print(f"Number of electrons ({len(data_electron)}) and other particles ({len(data_other)}) in data set")
    # shuffle and select the same number of other particles as there are electrons
    rng = np.random.default_rng()
    rng.shuffle(data_other)
    data_other = data_other[:len(data_electron)]
    # create training and testing data set
    data = np.concatenate((data_electron, data_other))
    labels_electron = np.ones((len(data_electron), 1), dtype=int)
    labels_other = np.zeros((len(data_other), 1), dtype=int)
    labels = np.vstack((labels_electron, labels_other))
    data, labels = shuffle_data(rng, data, labels)
    test_point = int(len(data) * 0.8)
    print(f"Training set size: {test_point}")
    print(f"Test set size: {len(data) - test_point}")


if __name__ == "__main__":
    __main__()
