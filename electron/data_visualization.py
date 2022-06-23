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
    if "ecal_energy" not in columns:
        print("Missing data.")
        return
    # plot energy for all tracks
    data = dataframe["ecal_energy"]
    plt.plot(data, "o")
    plt.show()
    # plot energy for electrons
    data_electrons = data[labels == 1]
    plt.plot(data_electrons, "o")
    plt.show()
    # plot histogram of energy for all tracks
    # bins = [100 * i for i in range(0, 210)]
    plt.hist(data, bins=1000, histtype="step")
    plt.xlabel("Energy")
    plt.ylabel("Tracks")
    plt.show()
    # plot histogram of energy for electrons
    plt.hist(data_electrons, bins=1000, histtype="step")
    plt.xlabel("Energy")
    plt.ylabel("Tracks")
    plt.show()
    # plot E/p for all tracks
    if "ep" not in columns:
        print("Missing data.")
        return
    data = dataframe["ep"]
    plt.hist(data, bins=100, histtype="step")
    plt.xlabel("E/p")
    plt.ylabel("Tracks")
    plt.show()
    # # plot E/p for electrons
    data_electrons = data[labels == 1]
    plt.hist(data_electrons, bins=100, histtype="step")
    plt.xlabel("E/p")
    plt.ylabel("Tracks")
    plt.show()


if __name__ == "__main__":
    __main__()
