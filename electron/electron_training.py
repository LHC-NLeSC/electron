import argparse
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch import nn

from utilities import load_data, ElectronDataset, testing_loop
from networks import ElectronNetwork, ElectronNetworkNormalized


label = "mcp_electron"
basic_training_column = "eop"
additional_training_columns = ["best_pt", "best_qop", "chi2", "chi2V", "first_qop", "ndof", "ndofT",
    "ndofV", "p", "qop", "tx", "ty", "x", "y", "z", "n_vertices", "n_tracks", "kalman_ip_chi2", "ecal_energy", 
    "ecal_digit_0", "ecal_digit_1", "ecal_digit_2", "ecal_digit_3", "ecal_digit_4", "ecal_digit_5"]


def training_loop(model, dataloader, loss_function, optimizer):
    model.train()
    for x, y in dataloader:
        prediction = model(x)
        loss = loss_function(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="ROOT file containing the training data set", type=str, required=True)
    # parameters
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=1024)
    parser.add_argument("--batch", help="Batch size", type=int, default=512)
    parser.add_argument("--learning", help="Learning rate", type=float, default=1e-3)
    # preprocessing
    parser.add_argument("--normalize", help="Use a normalization layer", action="store_true")
    # data
    parser.add_argument("--int8", help="Quantize the trained model to INT8", action="store_true")
    parser.add_argument("--plot", help="Plot accuracy over time", action="store_true")
    parser.add_argument("--save", help="Save the trained model to disk", action="store_true")
    return parser.parse_args()


def __main__():
    device = "cpu"
    arguments = command_line()
    dataframe, columns = load_data(arguments.filename)
    print(f"Columns in the table: {len(dataframe)}")
    print(columns)
    if label not in columns:
        print("Missing labels.")
        return
    labels = dataframe[label].astype(int)
    if basic_training_column not in columns:
        print("Missing training data.")
        return
    for column in additional_training_columns:
        if column not in columns:
            print("Missing additional training data.")
            return
    trainining_columns = additional_training_columns
    trainining_columns.append(basic_training_column)
    print(f"Columns for training: {len(trainining_columns)}")
    print(f"Entries in the table: {len(dataframe[basic_training_column])}")
    data = [dataframe[column] for column in trainining_columns]
    # split into electrons and other particles
    data = np.hstack([data[i].reshape(len(data[0]), 1) for i in range(len(data))])
    data_electron = data[labels == 1]
    data_other = data[labels == 0]
    print(f"Number of electrons ({len(data_electron)}) and other particles ({len(data_other)}) in data set")
    # select the same number of other particles as there are electrons
    rng = np.random.default_rng()
    rng.shuffle(data_other)
    data_other = data_other[:len(data_electron)]
    data = np.vstack((data_electron, data_other))
    labels_electron = np.ones((len(data_electron), 1), dtype=int)
    labels_other = np.zeros((len(data_other), 1), dtype=int)
    labels = np.vstack((labels_electron, labels_other))
    # create training and testing data set
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.3)
    print(f"Training set size: {len(data_train)}")
    print(f"Test set size: {len(data_test)}")
    training_data = ElectronDataset(torch.FloatTensor(data_train), torch.FloatTensor(labels_train))
    test_data = ElectronDataset(torch.FloatTensor(data_test), torch.FloatTensor(labels_test))
    training_dataloader = DataLoader(training_data, batch_size=arguments.batch, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=arguments.batch, shuffle=True)
    # model
    num_features = data_train.shape[1]
    if arguments.normalize:
        model = ElectronNetworkNormalized(num_features=num_features)
    else:
        model = ElectronNetwork(num_features=num_features)
    print(f"Device: {device}")
    model.to(device)
    print()
    print(model)
    print(f"Model parameters: {sum([x.reshape(-1).shape[0] for x in model.parameters()])}")
    print()
    # training and testing
    num_epochs = arguments.epochs
    batch_size = arguments.batch
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning)
    best_accuracy = -np.inf
    accuracy_history = list()
    best_weights = None
    for epoch in range(0, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        training_loop(model, training_dataloader, loss_function, optimizer)
        accuracy = testing_loop(model, test_dataloader)
        accuracy_history.append(accuracy * 100.0)
        print(f"\tAccuracy: {accuracy * 100.0:.2f}%")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_weights)
    print(f"Best Accuracy: {best_accuracy * 100.0:.2f}%")
    # plotting
    if arguments.plot:
        epochs = np.arange(0, num_epochs)
        plt.plot(epochs, accuracy_history, "r", label="Validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")
        plt.show()
    # save model
    if arguments.save:
        print("Saving model to disk")
        torch.save(model.state_dict(), "electron_model.pth")
        print("Saving model to ONNX format")
        dummy_input = torch.randn(1, 26)
        torch.onnx.export(model, dummy_input, "electron_model.onnx", export_params=True)
    # INT8 quantization
    if arguments.int8:
        print("INT8 quantization")
        model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
        if arguments.normalize:
            model_fused = torch.quantization.fuse_modules(model, [["layer1", "relu"]])
        else:
            model_fused = torch.quantization.fuse_modules(model, [["layer0", "relu"]])
        model_prepared = torch.quantization.prepare_qat(model_fused.train())
        for epoch in range(0, num_epochs):
            training_loop(model_prepared, training_dataloader, loss_function, optimizer)
        model_prepared.eval()
        model_int8 = torch.quantization.convert(model_prepared)
        print()
        print(model_int8)
        print()
        # save model
        if arguments.save:
            print("Saving INT8 model to disk")
            torch.save(model_int8.state_dict(), "electron_model_int8.pth")
            print("Saving INT8 model to ONNX format")
            dummy_input = torch.randn(1, 26)
            torch.onnx.export(model_int8, dummy_input, "electron_model_int8.onnx", export_params=True)

if __name__ == "__main__":
    __main__()
