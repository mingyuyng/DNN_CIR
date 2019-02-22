import numpy as np
from numpy import asarray as ar
from numpy import sqrt
from numpy.random import rand, randn
import torch
import torch.nn as nn
from torch.autograd import Variable


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNN_cn2_fc2(nn.Module):
    def __init__(self, isdrop, size=128):
        super(CNN_cn2_fc2, self).__init__()
        # Cin = 1, Cout = 256, Kernel_size = 11
        self.conv1 = nn.Conv1d(1, 64, 11, stride=1, padding=5)
        # Cin = 256, Cout = 256, Kernel_size = 33
        self.conv2 = nn.Conv1d(64, 128, 5, stride=1, padding=2)
        # Cin = 256, Cout = 256, Kernel_size = 17
        self.conv3 = nn.Conv1d(128, 256, 3, stride=1, padding=1)
        self.size = size
        # Batch Nromalization
        self.batnorm1 = nn.BatchNorm1d(64)
        self.batnorm2 = nn.BatchNorm1d(128)
        self.batnorm3 = nn.BatchNorm1d(256)
        self.relu = nn.LeakyReLU(0.1, True)

        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.isdrop = isdrop
        self.drop = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(int(self.size / 4) * 128, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, x):

        x = self.conv1(x)          # Cin = 1, Cout = 64, Kernel_size = 11
        x = self.batnorm1(x)
        x = self.relu(x)
        if self.isdrop == 1:
            x = self.drop(x)
        x = self.maxpool1(x)

        x = self.conv2(x)          # Cin = 64, Cout = 128, Kernel_size = 5
        x = self.batnorm2(x)
        x = self.relu(x)
        if self.isdrop == 1:
            x = self.drop(x)
        x = self.maxpool2(x)

        x = x.view(-1, int(self.size / 4) * 128)
        x = self.fc1(x)            # Din = 16*256, Dout = 1024
        x = self.relu(x)
        x = self.fc3(x)            # Din = 1024, Dout = 1

        return x


class CNN_cn2_fc3(nn.Module):
    def __init__(self, isdrop, size=128):
        super(CNN_cn2_fc3, self).__init__()
        # Cin = 1, Cout = 256, Kernel_size = 11
        self.conv1 = nn.Conv1d(1, 64, 11, stride=1, padding=5)
        # Cin = 256, Cout = 256, Kernel_size = 33
        self.conv2 = nn.Conv1d(64, 128, 5, stride=1, padding=2)
        # Cin = 256, Cout = 256, Kernel_size = 17
        self.conv3 = nn.Conv1d(128, 256, 3, stride=1, padding=1)
        self.size = size
        # Batch Nromalization
        self.batnorm1 = nn.BatchNorm1d(64)
        self.batnorm2 = nn.BatchNorm1d(128)
        self.batnorm3 = nn.BatchNorm1d(256)
        self.relu = nn.LeakyReLU(0.1, True)

        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.isdrop = isdrop
        self.drop = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(int(self.size / 4) * 128, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, x):

        x = self.conv1(x)          # Cin = 1, Cout = 64, Kernel_size = 11
        x = self.batnorm1(x)
        x = self.relu(x)
        if self.isdrop == 1:
            x = self.drop(x)
        x = self.maxpool1(x)

        x = self.conv2(x)          # Cin = 64, Cout = 128, Kernel_size = 5
        x = self.batnorm2(x)
        x = self.relu(x)
        if self.isdrop == 1:
            x = self.drop(x)
        x = self.maxpool2(x)

        x = x.view(-1, int(self.size / 4) * 128)
        x = self.fc1(x)            # Din = 16*256, Dout = 1024
        x = self.relu(x)
        x = self.fc2(x)            # Din = 1024, Dout = 1024
        x = self.relu(x)
        x = self.fc3(x)            # Din = 1024, Dout = 1

        return x


class CNN_cn3_fc3(nn.Module):
    def __init__(self, isdrop, size=128):
        super(CNN_cn3_fc3, self).__init__()
        # Cin = 1, Cout = 256, Kernel_size = 11
        self.conv1 = nn.Conv1d(1, 64, 11, stride=1, padding=5)
        # Cin = 256, Cout = 256, Kernel_size = 33
        self.conv2 = nn.Conv1d(64, 128, 5, stride=1, padding=2)
        # Cin = 256, Cout = 256, Kernel_size = 17
        self.conv3 = nn.Conv1d(128, 256, 3, stride=1, padding=1)
        self.size = size
        # Batch Nromalization
        self.batnorm1 = nn.BatchNorm1d(64)
        self.batnorm2 = nn.BatchNorm1d(128)
        self.batnorm3 = nn.BatchNorm1d(256)
        self.relu = nn.LeakyReLU(0.1, True)

        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.isdrop = isdrop
        self.drop = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(int(self.size / 8) * 256, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, x):

        x = self.conv1(x)          # Cin = 1, Cout = 64, Kernel_size = 11
        x = self.batnorm1(x)
        x = self.relu(x)
        if self.isdrop == 1:
            x = self.drop(x)
        x = self.maxpool1(x)

        x = self.conv2(x)          # Cin = 64, Cout = 128, Kernel_size = 5
        x = self.batnorm2(x)
        x = self.relu(x)
        if self.isdrop == 1:
            x = self.drop(x)
        x = self.maxpool2(x)

        x = self.conv3(x)          # Cin = 128, Cout = 256, Kernel_size = 3
        x = self.batnorm3(x)
        x = self.relu(x)
        if self.isdrop == 1:
            x = self.drop(x)
        x = self.maxpool3(x)

        x = x.view(-1, int(self.size / 8) * 256)
        x = self.fc1(x)            # Din = 16*256, Dout = 1024
        x = self.relu(x)
        x = self.fc2(x)            # Din = 1024, Dout = 1024
        x = self.relu(x)
        x = self.fc3(x)            # Din = 1024, Dout = 1

        return x
