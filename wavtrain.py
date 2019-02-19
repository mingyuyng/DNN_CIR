import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import tensorflow as tf
from keras.layers import merge
from keras import backend as K
import math
import cmath
from numpy import asarray as ar
from numpy import sqrt
from numpy.random import rand, randn
import random
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import scipy.io as sio
import scipy.signal
from scipy.interpolate import interp1d
import wave as wave
import librosa
import os.path
import model as model
sess = tf.InteractiveSession()


def extract(v):
    return v.data.storage().tolist()


def stats(d):
    return [np.mean(d), np.std(d)]


num_file = 0
sample_num = 8192
preprocess_rate = 3
X = torch.zeros(1000, 1, sample_num)
Y = torch.zeros(1000, 1, sample_num)
Hsample_rate = torch.zeros(1000, 1)
ds_rate = 2
F_leng = 2048

# Load data
for i in range(999):
    if os.path.isfile('p225/ori_p225_' + str(i + 1).zfill(3) + '.wav') and os.path.isfile('p225/intp_p225_' + str(i + 1).zfill(3) + '.wav'):
        aud, fs = librosa.load('p225/ori_p225_' + str(i + 1).zfill(3) + '.wav', sr=16000)
        aud2, fs2 = librosa.load('p225/intp_p225_' + str(i + 1).zfill(3) + '.wav', sr=16000)
        Y[num_file, :, :] = torch.tensor(aud)
        X[num_file, :, :] = torch.tensor(aud2)
        num_file += 1


Xtrain = X[:num_file - 80, :, :]
Ytrain = Y[:num_file - 80, :, :]
Xtest = X[num_file - 80:, :, :]
Ytest = Y[num_file - 80:, :, :]
# print X[0,:,:], Y[0,:,:], Xtrain.shape, Xtest.shape
# print fs, fs2, num_file

batch_size = 32
train_size = num_file - 80
print_interval = 2

Unet = model.unet()
criterion = nn.BCELoss()
L2criterion = nn.MSELoss()
learning_rate = 1e-4
optimizer = optim.Adam(Unet.parameters(), lr=learning_rate)

#y_p = Unet(Xtrain[0:5,:,:])
# print Xtrain[0:5,:,:].shape, y_p.shape

num_epochs = 1000

f = open("audio_1speaker_unet.log", "w")
f.write("Parameter: epoch=%s, batch size: %s, learning rate: %s \n" %
        (num_epochs, batch_size, learning_rate))


for epoch in range(num_epochs):
    Unet.zero_grad()  # Cancelled the gradient?
    index = random.sample(range(train_size), batch_size)
    uinput = Variable(Xtrain[index, :, :])
    pred_data = Unet(uinput)
    error_L2 = L2criterion(pred_data, Variable(Ytrain[index, :, :]))
    # optimizer.zero_grad()
    error_L2.backward()
    optimizer.step()  # Only optimizes G's parameters
    get_error = extract(error_L2)[0]

    if epoch % print_interval == 0:
        print("Epoch %s: err: %s \n" %
              (epoch, get_error))
        f.write("Epoch %s: err: %s \n" %
                (epoch, get_error))
        f.flush()

torch.save(Unet.state_dict(), 'Unet_2.w')

f.close()
