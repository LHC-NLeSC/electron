import argparse
import numpy as np
import matplotlib.pyplot as plt
from ROOT import TFile, RDataFrame

from utilities import load_data


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
    if "calo_energy" not in columns:
        print("Missing data.")
        return
    # plot energy for all tracks and only electrons
    data = dataframe["calo_energy"]
    plt.plot(data, "o")
    plt.show()
    # plot energy for just electrons
    data = dataframe["calo_energy"]
    data_electrons = data[labels == 1]
    plt.plot(data_electrons, "o")
    plt.show()
    # plot E/p for all data
    if "ep" not in columns:
        print("Missing data.")
        return
    data = dataframe["ep"]
    bins = [0.01 * i for i in range(0, 210)]
    plt.hist(data, bins=bins, histtype="step")
    plt.xlabel("E/p")
    plt.ylabel("Tracks")
    plt.show()
    # # plot E/p for electrons
    data = dataframe["ep"]
    data_electrons = data[labels == 1]
    bins = [0.01 * i for i in range(0, 210)]
    plt.hist(data_electrons, bins=bins, histtype="step")
    plt.xlabel("E/p")
    plt.ylabel("Tracks")
    plt.show()


if __name__ == "__main__":
    __main__()
