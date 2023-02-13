import argparse
import numpy as np

from utilities import load_data


label = "mcp_electron"
basic_training_column = "eop"
additional_training_columns = ["best_pt", "best_qop", "chi2", "chi2V", "first_qop", "ndof", "ndofT",
    "ndofV", "p", "qop", "tx", "ty", "x", "y", "z", "n_vertices", "n_tracks", "kalman_ip_chi2", "ecal_energy", 
    "ecal_digit_0", "ecal_digit_1", "ecal_digit_2", "ecal_digit_3", "ecal_digit_4", "ecal_digit_5"]


def command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="ROOT file containing the data set", type=str, required=True)
    parser.add_argument("--output", help="Prefix for the output file", type=str, required=True)
    return parser.parse_args()


def __main__():
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
    # save train, validation, test datasets
    max_train = int(0.6 * len(data))
    max_validation = int(0.8 * len(data))
    print(f"Training dataset size: {len(data[:max_train])}")
    print(f"Validation dataset size: {len(data[max_train:max_validation])}")
    print(f"Test dataset size: {len(data[max_validation:])}")
    np.save(f"{arguments.output}_train_data", data[:max_train])
    np.save(f"{arguments.output}_train_labels", labels[:max_train])
    np.save(f"{arguments.output}_valid_data", data[max_train:max_validation])
    np.save(f"{arguments.output}_valid_labels", labels[max_train:max_validation])
    np.save(f"{arguments.output}_test_data", data[max_validation:])
    np.save(f"{arguments.output}_test_labels", labels[max_validation:])

if __name__ == "__main__":
    __main__()