import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from numpy import asarray as ar
from numpy import sqrt
from numpy.random import rand, randn
from scipy.interpolate import interp1d
import dataloader as dl
import model as md


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract(v):
    return v.data.storage().tolist()


def stats(d):
    return [np.mean(d), np.std(d)]


dataset = dl.trainLoader('data/Train_set_coarse_net_log_window_poisson')

train_loader = torch.utils.data.DataLoader(dataset, batch_size=200,
                                           shuffle=True, num_workers=1,
                                           pin_memory=True)
net = md.CNN_cn2_fc2(1, 128)
criterion = nn.MSELoss()
learning_rate = 1e-4
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

print_interval = 1

num_epochs = 20
for epoch in range(num_epochs):
    # Step the scheduler for each epoch
    exp_lr_scheduler.step()
    # Iterate the whole dataset
    for i, (frame) in enumerate(train_loader):
        # zero out the parameter gradients
        optimizer.zero_grad()
        # Forward
        uinput = frame['time_cir'].type(torch.FloatTensor)
        utarget = frame['dist'].type(torch.FloatTensor)
        pred_data = net(uinput.unsqueeze(1))
        error = criterion(pred_data, utarget.unsqueeze(1))
        # Backward
        error.backward()
        optimizer.step()
        get_error = extract(error)[0]
        if epoch % print_interval == 0:
            print("Epoch %s: Iter: %s err: %s \n" %
                  (epoch, i, get_error))

torch.save(net.state_dict(), 'models/CNN_partial_log_window_poisson_cn2_fc2_drop.w')
