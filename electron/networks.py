import torch
from torch import nn


class ElectronNetwork(nn.Module):
    def __init__(self, num_features):
        super(ElectronNetwork, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.layer0 = nn.Linear(num_features, int((num_features + 1) / 2))
        self.relu = nn.ReLU()
        self.output = nn.Linear(int((num_features + 1) / 2), 1)
        self.sigmoid = nn.Sigmoid()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.layer0(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        x = self.dequant(x)
        return x

class ElectronNetworkNormalized(nn.Module):
    def __init__(self, num_features):
        super(ElectronNetworkNormalized, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.layer0 = nn.BatchNorm1d(num_features)
        self.layer1 = nn.Linear(num_features, int((num_features + 1) / 2))
        self.relu = nn.ReLU()
        self.output = nn.Linear(int((num_features + 1) / 2), 1)
        self.sigmoid = nn.Sigmoid()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        x = self.dequant(x)
        return x