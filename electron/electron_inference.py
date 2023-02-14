import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import onnx2torch

from utilities import load_data, shuffle_data, ElectronDataset, testing_loop
from networks import ElectronNetwork, ElectronNetworkNormalized


label = "mcp_electron"
model_columns = ["eop", "best_pt", "best_qop", "chi2", "chi2V", "first_qop", "ndof", "ndofT",
    "ndofV", "p", "qop", "tx", "ty", "x", "y", "z", "n_vertices", "n_tracks", "kalman_ip_chi2", "ecal_energy", 
    "ecal_digit_0", "ecal_digit_1", "ecal_digit_2", "ecal_digit_3", "ecal_digit_4", "ecal_digit_5"]


def command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="File containing the data set", type=str, required=True)
    parser.add_argument("--nocuda", help="Disable CUDA", action="store_true")
    parser.add_argument("--model", help="Name of the file containing the model.", type=str, required=True)
    parser.add_argument("--batch", help="Batch size", type=int, default=512)
    parser.add_argument("--normalize", help="Use a normalization layer", action="store_true")
    parser.add_argument("--int8", help="Quantize the trained model to INT8", action="store_true")
    return parser.parse_args()


def __main__():
    arguments = command_line()
    if not arguments.nocuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if ".root" in arguments.filename:
        dataframe, columns = load_data(arguments.filename)
        print(f"Columns in the table: {len(dataframe)}")
        print(columns)
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
        # select the same number of other particles as there are electrons
        rng = np.random.default_rng()
        rng.shuffle(data_other)
        data_other = data_other[:len(data_electron)]
        # create training and testing data set
        data = np.vstack((data_electron, data_other))
        labels_electron = np.ones((len(data_electron), 1), dtype=int)
        labels_other = np.zeros((len(data_other), 1), dtype=int)
        labels = np.vstack((labels_electron, labels_other))
        data, labels = shuffle_data(rng, data, labels)
    else:
        data = np.load(f"{arguments.filename}_test_data.npy")
        labels = np.load(f"{arguments.filename}_test_labels.npy")
        print(f"Test set size: {len(data)}")
    test_dataset = ElectronDataset(torch.tensor(datadtype=torch.float32, device=device), torch.tensor(labelsdtype=torch.float32, device=device))
    test_dataloader = DataLoader(test_dataset, batch_size=arguments.batch, shuffle=True)
    # read model
    num_features = data.shape[1]
    if arguments.int8:
        model = torch.load(arguments.model)
    else:
        if "onnx" in arguments.model:
            model = onnx2torch.convert(arguments.model)
        else:
            if arguments.normalize:
                model = ElectronNetworkNormalized(num_features=num_features)
            else:
                model = ElectronNetwork(num_features=num_features)
            weights = torch.load(arguments.model)
            model.load_state_dict(weights)
    print(f"Device: {device}")
    model.to(device)
    print()
    print(model)
    print()
    # inference
    loss_function = nn.BCELoss()
    accuracy, loss = testing_loop(model, test_dataloader, loss_function)
    print(f"Accuracy: {accuracy * 100.0:.2f}%")
    print(f"Loss: {loss}")

if __name__ == "__main__":
    __main__()
