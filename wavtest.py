import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from numpy import asarray as ar
from numpy import sqrt
from numpy.random import rand, randn
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import dataloader as dl
import model as md


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract(v):
    return v.data.storage().tolist()


def stats(d):
    return [np.mean(d), np.std(d)]


dataset = dl.testLoader('data/Test_set_coarse_net_log_window_poisson')

test_loader = torch.utils.data.DataLoader(dataset, batch_size=100,
                                          shuffle=True, num_workers=1,
                                          pin_memory=True)
net = md.CNN_cn2_fc2(0)
net.load_state_dict(torch.load('models/CNN_partial_log_window_poisson_cn2_fc2.w'))
error = []
est_error = []
net.eval()

for i, (frame) in enumerate(test_loader):

    x = frame['time_cir'].type(torch.FloatTensor)
    y = frame['dist'].type(torch.FloatTensor)
    y_coarse = frame['coarse'].type(torch.FloatTensor)
    y_true = frame['true'].type(torch.FloatTensor)

    pred_data = net(x.unsqueeze(1))
    e = pred_data.squeeze(1) - y
    error.append(e.detach().numpy())
    est = y_coarse - pred_data.squeeze(1) - y_true
    est_error.append(est.detach().numpy())
    print(i)


err = np.hstack(error)
err_est = np.hstack(est_error)
print(e.shape)
print(err.shape)

import pdb
pdb.set_trace()  # breakpoint b8c87337 //

plt.plot(err)
plt.show()
