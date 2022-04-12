import argparse
import numpy as np
import matplotlib.pyplot as plt
from ROOT import TFile, RDataFrame

from utilities import load_data, unpack_digit_indices


def command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="ROOT file containing the training data set", type=str, required=True)
    return parser.parse_args()


def __main__():
    arguments = command_line()
    dataframe, columns = load_data(arguments.filename)
    if "mcp_electron" not in columns:
        print("Missing labels.")
        return
    labels = dataframe["mcp_electron"].astype(int)
    if "digit_indices" not in columns:
        print("Missing data.")
        return
    data = unpack_digit_indices(dataframe["digit_indices"])
    # plot all data
    for index in range(len(data)):
        plt.plot(data[index], "o", label=f"Column {index}")
    plt.legend(loc="upper right")
    plt.show()
    # plot only electrons
    data = dataframe["digit_indices"]
    data_electrons = data[labels == 1]
    data_electrons = unpack_digit_indices(data_electrons)
    for index in range(len(data_electrons)):
        plt.plot(data_electrons[index], "o", label=f"Column {index}")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    __main__()
