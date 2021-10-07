# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %% [markdown]
# # Testing Dataset
# 
# Dependencies:

# %%
import os
import shutil
import random; random.seed(42)

import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
from torch import nn
#import tensorflow as tf # tensorflow-gpu==2.0.0
#from tensorflow.python.client import device_lib 
#print(device_lib.list_local_devices())
#import cv2

# %% [markdown]
# Directories:

# %%
# training data
train_dir = 'Data/train_and_test_split/dpc_dataset_traintest_4_200_csv/train'
train_dir_video = 'Data/train_and_test_split/dpc_dataset_traintest_4_200_h264/train'

# test data
test_inputs_dir = 'Data/train_and_test_split/dpc_dataset_traintest_4_200_csv/test_inputs/'
test_targets_dir = 'Data/train_and_test_split/dpc_dataset_traintest_4_200_csv/test_targets/'
test_targets_video = 'Data/train_and_test_split/dpc_dataset_traintest_4_200_h264/test_targets/'

# validation data
validation_inputs_dir = 'Data/train_and_test_split/dpc_dataset_traintest_4_200_csv/validation_inputs/'
validation_targets_dir = 'tData/rain_and_test_split/dpc_dataset_traintest_4_200_csv/validation_targets/'
validation_targets_video = 'Data/train_and_test_split/dpc_dataset_traintest_4_200_h264/validation_targets/'

# %% [markdown]
# ## Coordinate based predictions:
# ### Data Transformation Functions

# %%
# some constants
DEFAULT_X_RED, DEFAULT_Y_RED = (240, 240)

PIXEL_DISTANCE_GREEN_TO_RED = 118 # approx. value | calculated with the Pythagorean theorem and averaged: np.sqrt((y_green-y_red)**2 + (x_green-x_red)**2)
PIXEL_DISTANCE_BLUE_TO_GREEN = 90 # approx. value | calculated with the Pythagorean theorem and averaged: np.sqrt((y_blue-y_green)**2 + (x_blue-x_green)**2)

def raw_to_pixel(l):
    '''Convert the raw coordinates to pixel coordinates.'''
    assert isinstance(l, list)
    return [x/5 for x in l]


def pixel_to_raw(l):
    '''Convert the pixel coordinates to raw coordinates.'''
    assert isinstance(l, list)
    return [x*5 for x in l]


def raw_cartesian_to_polar_angles(l):
    '''Convert the cartesian coordinates to polar coordinates.'''
    assert isinstance(l, list)
    x_red, y_red, x_green, y_green, x_blue, y_blue = raw_to_pixel(l)

    angle_green_red = np.arctan((y_green-y_red)/(x_green-x_red+0.001))
    angle_blue_green = np.arctan((y_blue-y_green)/(x_blue-x_green+0.001))
    
    return [np.sin(angle_green_red), np.cos(angle_green_red), np.sin(angle_blue_green), np.cos(angle_blue_green)]

def polar_angles_to_raw_cartesian(l):
    '''Convert the polar coordinates back to cartesian coordinates.'''
    assert isinstance(l, list)
    sin_angle_green_red, cos_angle_green_red, sin_angle_blue_green, cos_angle_blue_green = l
    
    y_green = PIXEL_DISTANCE_GREEN_TO_RED * sin_angle_green_red + DEFAULT_Y_RED
    x_green = PIXEL_DISTANCE_GREEN_TO_RED * cos_angle_green_red + DEFAULT_X_RED

    y_blue = PIXEL_DISTANCE_BLUE_TO_GREEN * sin_angle_blue_green + y_green
    x_blue = PIXEL_DISTANCE_BLUE_TO_GREEN * cos_angle_blue_green + x_green
    
    return pixel_to_raw([DEFAULT_X_RED, DEFAULT_Y_RED, x_green, y_green, x_blue, y_blue])

# %% [markdown]
# Verify that the raw -> pixel conversion and pixel -> raw works as intended, and that the cartesian -> polar conversion and polar -> cartesian conversion works as intended.

# %%
raw_coordinates = list(np.array([240, 240, 357.4438349670886, 228.55685234634907, 444.41827493559794, 205.41712909467287])*5)
pixel_coordinates = raw_to_pixel(raw_coordinates)
new_raw_coordinates = pixel_to_raw(pixel_coordinates)
assert raw_coordinates == new_raw_coordinates, '`Raw -> Pixel` and `Pixel -> Raw` coordinate conversion methods are malfunctioning.'

raw_cartesian = list(np.array([240, 240, 357.4438349670886, 228.55685234634907, 444.41827493559794, 205.41712909467287])*5)
polar = raw_cartesian_to_polar_angles(raw_cartesian)
new_raw_cartesian = polar_angles_to_raw_cartesian(polar)
assert [round(x) for x in raw_cartesian] == [round(x) for x in new_raw_cartesian], 'Cartesian to Polar and Polar to Cartesian methods are malfunctioning.'

# %% [markdown]
# Data reading functions
# %% [markdown]
# Parsing training data:
# training data x-y matching is like this:
# x: a list of 4 frames
# y: the frame that follows

# %%
def parse_training_annotations(csv_file):
    '''Parse the training annotations from a CSV file.'''
    X_data = []
    y_data = []
    f = pd.read_csv(csv_file, header=None, delim_whitespace=True, engine='python')
    temp = []
    for i, row in f.iterrows():
        if len(temp) < 4:
            # convert the cartesian pixel coordinates to polar coordinates
            temp.append(raw_cartesian_to_polar_angles(row.to_list()))
        else:
            # the output frame
            # convert the cartesian pixel coordinates to polar coordinates
            next_frame = raw_cartesian_to_polar_angles(row.to_list())

            # save
            X_data.append(temp.copy())
            y_data.append(next_frame.copy())

            # add output frame to the inputs and remove the first
            temp.pop(0)
            temp.append(next_frame)
    return X_data, y_data

# %% [markdown]
# Load in data

# %%
BATCH_SIZE = 4000

# load in all separate files
X = []
y = []
for filename in tqdm([x for x in os.listdir(train_dir) if not x.startswith('.')]):
    # load in a file
    X_data, y_data = parse_training_annotations(os.path.join(train_dir, filename))
    
    # extract sequential batches and add them to the training data
    for i in range(len(X_data) // BATCH_SIZE):
        X_batch = X_data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        y_batch = y_data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        X.append(X_batch)
        y.append(y_batch)

num_batches = len(X)
num_records = num_batches * BATCH_SIZE
print(f'{num_records} training records spread over {num_batches} batches of size {BATCH_SIZE}')


# %%
X = np.array(X)
y = np.array(y)


# %%
# torch_X has n batches of size 4000, each sample is 4 frame, each frame is 4 value (sin,cos,sin,cos), e.g. torch.Size([69,4000,4,4])
torch_X = torch.from_numpy(X)
# torch_y has n batches of size 4000, each sample is a sequence of frames, unknown length, of 4 values, e.g. torch.Size([69,4000,4])
torch_y = torch.from_numpy(y)


# %%
torch_X = torch_X.view(-1,4,4).float()
torch_y = torch_y.view(-1,4).float()


# %%
class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        #Defining the layers
        # LSTM layer
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = n_layers, batch_first = True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).requires_grad_()

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out


# %%
class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, x, y):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()


# %%
# Instantiate the model with hyperparameters
model = Model(input_size=4, output_size=4, hidden_size=4, n_layers=2)

# Define hyperparameters
n_epochs = 500
lr=0.01

# Define Loss, Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# %%
model(torch_X).shape


# %%

for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    output = model(torch_X)
    loss = criterion(output.view(-1), torch_y.view(-1))
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly
    
    if epoch%10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))


# %%



