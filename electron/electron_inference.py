import argparse
import numpy as np
import tensorflow as tf
from os import environ

from utilities import load_data, shuffle_data

# reduce TensorFlow verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

label = "mcp_electron"
model_columns = ["eop", "best_pt", "best_qop", "chi2", "chi2V", "first_qop", "ndof", "ndofT",
    "ndofV", "p", "qop", "tx", "ty", "x", "y", "z", "n_vertices", "n_tracks", "kalman_ip_chi2", "ecal_energy", 
    "ecal_digit_0", "ecal_digit_1", "ecal_digit_2", "ecal_digit_3", "ecal_digit_4", "ecal_digit_5"]


def command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="ROOT file containing the data set", type=str, required=True)
    parser.add_argument("--model", help="Name of the file containing the model.", type=str, required=True)
    parser.add_argument("--threshold", help="Threshold for electron probability", type=float, default=0)
    return parser.parse_args()


def __main__():
    arguments = command_line()
    dataframe, columns = load_data(arguments.filename)
    if label not in columns:
        print("Missing labels.")
        return
    labels = dataframe[label].astype(int)
    for column in model_columns:
        if column not in columns:
            print("Missing data.")
            return
    # create dataset
    data = [dataframe[column] for column in model_columns]
    # split into electrons and other particles
    data = np.hstack([data[i].reshape(len(data[0]), 1) for i in range(len(data))])
    data_electron = data[labels == 1]
    data_other = data[labels == 0]
    print(f"Number of electrons ({len(data_electron)}) and other particles ({len(data_other)}) in data set")
    # shuffle and select the same number of other particles as there are electrons
    rng = np.random.default_rng()
    rng.shuffle(data_other)
    data_other = data_other[:len(data_electron)]
    # create training and testing data set
    data = np.vstack((data_electron, data_other))
    labels_electron = np.ones((len(data_electron), 1), dtype=int)
    labels_other = np.zeros((len(data_other), 1), dtype=int)
    labels = np.vstack((labels_electron, labels_other))
    data, labels = shuffle_data(rng, data, labels)
    # read model
    model = tf.keras.models.load_model(arguments.model)
    model.summary()
    # inference
    predictions = model.predict(data)
    # analysis
    predictions = np.transpose(predictions)[0]
    predictions = list(map(lambda x: 0 if x < arguments.threshold else 1, predictions))
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(0, len(labels)):
        if predictions[i] == 1 and labels[i] == 1:
            tp = tp + 1
            continue
        if predictions[i] == 1 and labels[i] == 0:
            fp = fp + 1
            continue
        if predictions[i] == 0 and labels[i] == 0:
            tn = tn + 1
            continue
        if predictions[i] == 0 and labels[i] == 1:
            fn = fn + 1
            continue
    print(f"True positives: {tp}")
    print(f"False positives: {fp}")
    print(f"True negatives: {tn}")
    print(f"False negatives: {fn}")
    print(f"Accuracy: {(((tp + tn) * 100) / len(labels)):.2f}%")

if __name__ == "__main__":
    __main__()
