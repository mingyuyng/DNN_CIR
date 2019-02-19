import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import math
from numpy import asarray as ar
from numpy import sqrt
from numpy.random import rand, randn
from torch.autograd import Variable
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import scipy.io as sio


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, (9, 1), bias=True, padding=(4, 0))
        self.conv2 = nn.Conv2d(64, 32, (1, 1), bias=True)
        self.conv3 = nn.Conv2d(32, 2, (5, 1), bias=True, padding=(2, 0))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        return out


train = 0
net = NeuralNet()

mat_data = sio.loadmat('Train_set_bw80_time_uniform_5_15_SS.mat')

trainingdata = mat_data['saved_features']
train_real = np.real(trainingdata) * 50000
train_imag = np.imag(trainingdata) * 50000
targetdata = mat_data['saved_features_fake']
target_abs = np.abs(targetdata)

train_set = torch.empty(15000, 3, trainingdata.shape[0], 1)
train_set[:, 0, :, 0] = torch.from_numpy(train_real[:, :15000].T)
train_set[:, 1, :, 0] = torch.from_numpy(train_imag[:, :15000].T)
train_set[:, 2, :, 0] = torch.from_numpy(target_abs[:, :15000].T).type(torch.LongTensor)

test_set = torch.empty(5000, 2, trainingdata.shape[0], 1)
test_set[:, 0, :, 0] = torch.from_numpy(train_real[:, 15000:].T)
test_set[:, 1, :, 0] = torch.from_numpy(train_imag[:, 15000:].T)
test_target = torch.from_numpy(target_abs[:, 15000:].T).type(torch.LongTensor)


if train == 1:

    print(net)
    batch_size = 128
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    num_training = np.ceil(trainingdata.shape[1] / batch_size)

    w = torch.empty(2, dtype=torch.float)
    w[0] = 1
    w[1] = 2000
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
    num_epoch = 10
    record = np.zeros(shape=(int(num_training) * num_epoch, 3))

    for epoch in range(num_epoch):
        for i, data in enumerate(trainloader, 0):
            train = data[:, :2, :, :]
            target = data[:, 2, :, :]
            target = target.type(torch.LongTensor)
            outputs = net(train)  # need to be (n, inputsize)
            loss = criterion(outputs, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cnt = epoch * int(num_training) + i
            record[cnt, :] = np.array([epoch, cnt, loss])
            if (i + 1) % 5 == 0:
                print('loss after', ' epoch: ', epoch + 1, 'iter: ', i + 1, ' step optimization: ', loss.item())

    torch.save(net.state_dict(), 'DNN_ToA_model.ckpt')
else:
    net.load_state_dict(torch.load('DNN_ToA_model.ckpt'))
    import pdb
    pdb.set_trace()  # breakpoint 12c696b3 //

    out = net(test_set)

    print(out)
