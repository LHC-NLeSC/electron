import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import onnx
import tf2onnx
from os import environ

from utilities import load_data

# reduce TensorFlow verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="ROOT file containing the data set", type=str, required=True)
    parser.add_argument("--model", help="Name of the file containing the model.", type=str, required=True)
    return parser.parse_args()


def __main__():
    arguments = command_line()
    dataframe, _ = load_data(arguments.filename)
    data = [dataframe[i] for i in range(0, len(dataframe))]
    # read model
    model = tf.keras.models.load_model(arguments.model)
    # inference
    model.predict(data)

if __name__ == "__main__":
    __main__()
