import numpy as np
from ROOT import TFile, RDataFrame


def load_data(filename : str):
    global columns
    kalman_file = TFile(filename)
    dataframe = RDataFrame("kalman_validator/kalman_ip_tree", kalman_file)
    dataframe = dataframe.Define("p", "abs(1.f/best_qop)")
    dataframe = dataframe.Define("ep", "abs(calo_energy/p)")
    return dataframe.AsNumpy(), dataframe.GetColumnNames()


def shuffle_data(rng, data, labels):
    assert(len(data) == len(labels))
    permutation = rng.permutation(len(data))
    return data[permutation], labels[permutation]
