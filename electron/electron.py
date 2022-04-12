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


if __name__ == "__main__":
    __main__()
