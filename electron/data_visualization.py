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
    if "mcp_electron" not in columns and "interesting_electron" not in columns:
        print("Missing labels.")
        return
    labels_electron = dataframe["mcp_electron"].astype(int)
    labels_interesting_electron = dataframe["interesting_electron"].astype(int)
    # plot E/p distribution
    if "eop" not in columns:
        print("Missing data.")
        return
    data = dataframe["eop"]
    data_non_electrons = data[labels_electron == 0]
    data_electrons = data[labels_electron == 1]
    data_interesting_electrons = data[labels_interesting_electron == 1]
    plt.hist(data_non_electrons, bins=200, histtype="step", weights=np.ones_like(data_non_electrons) / len(data_non_electrons), label="Not electrons")
    plt.hist(data_electrons, bins=200, histtype="step", weights=np.ones_like(data_electrons) / len(data_electrons), label="True electrons")
    plt.hist(data_interesting_electrons, bins=200, histtype="step", weights=np.ones_like(data_interesting_electrons) / len(data_interesting_electrons), label="True electrons - interesting")
    plt.xlabel("E/p")
    plt.xlim(0, 2)
    plt.legend()
    plt.show()
    # plot X^2 ip distribution
    if "kalman_ip_chi2" not in columns:
        print("Missing data.")
        return
    data = dataframe["kalman_ip_chi2"]
    data_non_electrons = np.log(data[labels_electron == 0])
    data_electrons = np.log(data[labels_electron == 1])
    data_interesting_electrons = np.log(data[labels_interesting_electron == 1])
    plt.hist(data_non_electrons, bins=200, histtype="step", weights=np.ones_like(data_non_electrons) / len(data_non_electrons), label="Not electrons")
    plt.hist(data_electrons, bins=200, histtype="step", weights=np.ones_like(data_electrons) / len(data_electrons), label="True electrons")
    plt.hist(data_interesting_electrons, bins=200, histtype="step", weights=np.ones_like(data_interesting_electrons) / len(data_interesting_electrons), label="True electrons - interesting")
    plt.xlabel("ln(X^2 ip)")
    plt.xlim(-10, 12)
    plt.legend()
    plt.show()
    # plot Pt distribution
    if "best_pt" not in columns:
        print("Missing data.")
        return
    data = dataframe["best_pt"]
    data_non_electrons = np.log(data[labels_electron == 0])
    data_electrons = np.log(data[labels_electron == 1])
    data_interesting_electrons = np.log(data[labels_interesting_electron == 1])
    plt.hist(data_non_electrons, bins=200, histtype="step", weights=np.ones_like(data_non_electrons) / len(data_non_electrons), label="Not electrons")
    plt.hist(data_electrons, bins=200, histtype="step", weights=np.ones_like(data_electrons) / len(data_electrons), label="True electrons")
    plt.hist(data_interesting_electrons, bins=200, histtype="step", weights=np.ones_like(data_interesting_electrons) / len(data_interesting_electrons), label="True electrons - interesting")
    plt.xlabel("ln(Pt)")
    plt.xlim(5, 10)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    __main__()
