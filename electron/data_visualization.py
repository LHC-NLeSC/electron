import argparse
import numpy as np
import matplotlib.pyplot as plt
from ROOT import TFile, RDataFrame

from utilities import load_data, unpack_digit_indices


def command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="ROOT file containing the training data set", type=str, required=True)
    parser.add_argument("--filter", help="Filter out columns with a single value.", action="store_true")
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
    # plot all data
    data = unpack_digit_indices(dataframe["digit_indices"], filter=arguments.filter)
    print(f"Shape of unpacked digit_indices: {data.shape}")
    for index in range(len(data)):
        plt.plot(data[index], "o", label=f"Column {index}")
    plt.legend(loc="upper right")
    plt.show()
    # plot only electrons
    data = dataframe["digit_indices"]
    data_electrons = data[labels == 1]
    data_electrons = unpack_digit_indices(data_electrons, filter=arguments.filter)
    print(f"Shape of unpacked electron digit_indices: {data_electrons.shape}")
    for index in range(len(data_electrons)):
        plt.plot(data_electrons[index], "o", label=f"Column {index}")
    plt.legend(loc="upper right")
    plt.show()
    # plot E/p for all data
    if "p" not in columns:
        print("Missing data.")
        return
    e_p = []
    data = unpack_digit_indices(dataframe["digit_indices"])
    for index in range(data.shape[1]):
        energy = 0.0
        for column in range(data.shape[0]):
            if data[column][index] != 9999:
                energy = energy + data[column][index]
        e_p.append(energy / dataframe["p"][index])
    bins = [0.01 * i for i in range(0, 210)]
    plt.hist(e_p, bins=bins, histtype="step")
    plt.xticks([0.1 * i for i in range(0, 21)])
    plt.xlabel("E/p")
    plt.ylabel("Tracks")
    plt.show()
    # plot E/p for electrons
    if "p" not in columns:
        print("Missing data.")
        return
    e_p = []
    data = dataframe["digit_indices"]
    data_electrons = data[labels == 1]
    data_electrons = unpack_digit_indices(data_electrons)
    for index in range(data_electrons.shape[1]):
        energy = 0.0
        for column in range(data_electrons.shape[0]):
            if data_electrons[column][index] != 9999:
                energy = energy + data_electrons[column][index]
        e_p.append(energy / dataframe["p"][labels == 1][index])
    bins = [0.01 * i for i in range(0, 210)]
    plt.hist(e_p, bins=bins, histtype="step")
    plt.xticks([0.1 * i for i in range(0, 21)])
    plt.xlabel("E/p")
    plt.ylabel("Tracks")
    plt.show()


if __name__ == "__main__":
    __main__()
