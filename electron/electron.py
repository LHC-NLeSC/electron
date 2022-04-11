import argparse
import numpy as np
import tensorflow as tf
from os import environ
from ROOT import TFile, RDataFrame

# reduce TensorFlow verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="ROOT file containing the training data set", type=str, required=True)
    return parser.parse_args()


def load_data(filename : str):
    kalman_file = TFile(filename)
    dataframe = RDataFrame("kalman_validator/kalman_ip_tree", kalman_file)
    return dataframe.AsNumpy()


def __main__():
    arguments = command_line()
    dataframe = load_data(arguments.filename)


if __name__ == "__main__":
    __main__()
