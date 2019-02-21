import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from numpy import asarray as ar
from numpy import sqrt
from numpy.random import rand, randn
import random
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import dataloader as dl
import model as md


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract(v):
    return v.data.storage().tolist()


def stats(d):
    return [np.mean(d), np.std(d)]


dataset = dl.trainLoader('Train_set_net_log_window_poisson')

train_loader = torch.utils.data.DataLoader(dataset, batch_size=200,
                                           shuffle=True, num_workers=1,
                                           pin_memory=True)
Unet = md.CNN()
L2criterion = nn.MSELoss()
learning_rate = 1e-4
optimizer = optim.Adam(Unet.parameters(), lr=learning_rate)
print_interval = 1

num_epochs = 20
for epoch in range(num_epochs):
    for i, (frame) in enumerate(train_loader):

        x = frame['time_cir'].type(torch.FloatTensor)
        y = frame['dist'].type(torch.FloatTensor)
        uinput = Variable(x)
        utarget = Variable(y)

        pred_data = Unet(uinput.unsqueeze(1))
        error_L2 = L2criterion(pred_data[:, 0], utarget)
        optimizer.zero_grad()
        error_L2.backward()
        optimizer.step()  # Only optimizes G's parameters
        get_error = extract(error_L2)[0]
        if epoch % print_interval == 0:
            print("Epoch %s: Iter: %s err: %s \n" %
                  (epoch, i, get_error))

torch.save(Unet.state_dict(), 'CNN_all_log_window_poisson.w')
