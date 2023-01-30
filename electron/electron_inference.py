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

model_columns = ["eop", "best_pt", "best_qop", "chi2", "chi2V", "first_qop", "ndof", "ndofT",
    "ndofV", "p", "qop", "tx", "ty", "x", "y", "z", "n_vertices", "n_tracks", "kalman_ip_chi2", "ecal_energy", 
    "ecal_digit_0", "ecal_digit_1", "ecal_digit_2", "ecal_digit_3", "ecal_digit_4", "ecal_digit_5"]


def command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="ROOT file containing the data set", type=str, required=True)
    parser.add_argument("--model", help="Name of the file containing the model.", type=str, required=True)
    return parser.parse_args()


def __main__():
    arguments = command_line()
    dataframe, columns = load_data(arguments.filename)
    for column in model_columns:
        if column not in columns:
            print("Missing data.")
            return
    data = [dataframe[column] for column in model_columns]
    # read model
    model = tf.keras.models.load_model(arguments.model)
    # inference
    model.predict(data)

if __name__ == "__main__":
    __main__()
