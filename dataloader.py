import torch
import torch.utils.data as data
import scipy.io as sio
import numpy as np


class trainLoader(data.Dataset):

    def __init__(self, filename):

        self.mat_data = sio.loadmat(filename)
        self.data = self.mat_data['x_train'].T

    def __getitem__(self, index):

        time_cir = self.data[index].astype('double')
        dist = self.mat_data['y_train'][0, index].astype('double')

        pair = {'time_cir': time_cir, 'dist': dist}
        return pair

    def __len__(self):
        return self.mat_data['x_train'].shape[1]


class testLoader(data.Dataset):

    def __init__(self, filename):

        self.mat_data = sio.loadmat(filename)
        self.data = self.mat_data['x_test'].T

    def __getitem__(self, index):

        time_cir = self.data[index].astype('double')
        dist = self.mat_data['y_test'][0, index].astype('double')
        coarse = self.mat_data['y_coar'][0, index].astype('double')
        true = self.mat_data['y_true'][0, index].astype('double')
        pair = {'time_cir': time_cir, 'dist': dist}
        return pair

    def __len__(self):
        return self.mat_data['x_test'].shape[1]
