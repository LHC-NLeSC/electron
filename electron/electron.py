import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import onnx
import tf2onnx
from os import environ

from utilities import load_data, shuffle_data, threshold_method

# reduce TensorFlow verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


label = "mcp_electron"
basic_training_column = "ecal_energy"
additional_training_columns = ["best_pt", "best_qop", "chi2", "chi2V", "first_qop", "ndof", "ndofT",
    "ndofV", "p", "qop", "tx", "ty", "x", "y", "z", "ep"]


def command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="ROOT file containing the training data set", type=str, required=True)
    # parameters
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=1024)
    parser.add_argument("--batch", help="Batch size", type=int, default=512)
    # preprocessing
    parser.add_argument("--normalize", help="Use a normalization layer", action="store_true")
    # data
    parser.add_argument("--plot", help="Plot accuracy over time", action="store_true")
    parser.add_argument("--save", help="Save the trained model to disk", action="store_true")
    return parser.parse_args()


def __main__():
    arguments = command_line()
    dataframe, columns = load_data(arguments.filename)
    print(f"Columns in the table: {len(dataframe)}")
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
    test_point = int(len(data) * 0.8)
    print(f"Training set size: {test_point}")
    print(f"Test set size: {len(data) - test_point}")
    # model
    num_features = data.shape[1]
    if arguments.normalize:
        print("Normalization enabled")
        normalization_layer = tf.keras.layers.Normalization()
        normalization_layer.adapt(data[:test_point])
        model = tf.keras.Sequential([
            normalization_layer,
            tf.keras.layers.Dense(units=((num_features + 1) / 2), activation="relu"),
            tf.keras.layers.Dense(units=1)
            ])
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=((num_features + 1) / 2), input_dim=num_features, activation="relu"),
            tf.keras.layers.Dense(units=1)
            ])
    print()
    model.summary()
    print()
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"]
        )
    # training
    num_epochs = arguments.epochs
    batch_size = arguments.batch
    training_history = model.fit(
        data[:test_point],
        labels[:test_point],
        validation_split=0.2,
        epochs=num_epochs,
        batch_size=batch_size,
        verbose=0
        )
    # evaluation
    loss, accuracy = model.evaluate(data[test_point:], labels[test_point:], verbose=0)
    print(f"Loss: {loss}, Accuracy: {accuracy}")
    # threshold method evaluation
    threshold_accuracy = threshold_method(data[test_point:], labels[test_point:])
    print(f"Threshold method accuracy: {threshold_accuracy}")
    # plotting
    if arguments.plot:
        epochs = np.arange(0, num_epochs)
        plt.plot(epochs, training_history.history["loss"], "bo", label="Training loss")
        plt.plot(epochs, training_history.history["val_loss"], "ro", label="Validation loss")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.show()
        plt.plot(epochs, training_history.history["accuracy"], "b", label="Training accuracy")
        plt.plot(epochs, training_history.history["val_accuracy"], "r", label="Validation accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")
        plt.show()
    # save model
    if arguments.save:
        print("Saving model to disk")
        model.save("ghostprob_model.h5")
        print("Saving model to ONNX format")
        input_signature = [tf.TensorSpec(input.shape, input.dtype) for input in model.inputs]
        model_onnx, _ = tf2onnx.convert.from_keras(model, input_signature)
        onnx.save(model_onnx, "ghostprob_model.onnx")

if __name__ == "__main__":
    __main__()
