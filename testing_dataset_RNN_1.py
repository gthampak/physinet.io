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
            X_data.append(temp)
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

    X = X + X_data
    y = y + y_data


# %%
class DoublePendulumDataset(torch.utils.data.Dataset):
    def __init__(self,X_list,y_list):
        self.sample_list = list(zip(X_list, y_list))
    
    def __getitem__(self,index):
        X_sample,y_sample = self.sample_list[index]
        return torch.from_numpy(np.array(X_sample)).float(),torch.from_numpy(np.array(y_sample)).float()
    
    def __len__(self):
        return len(self.sample_list)


# %%
myDataSet = DoublePendulumDataset(X,y)
myDataLoader = torch.utils.data.DataLoader(myDataSet,batch_size=4000)


# %%
class Model(nn.Module):
    def __init__(self):
        # We want a model of 4 layer LSTM with 32 features output, and a dense layer to form the 4 feature output.
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_size = 32
        self.n_layers = 4

        #Defining the layers
        # LSTM layer
        self.lstm1 = nn.LSTM(input_size = 4, hidden_size = 32, num_layers = 1, batch_first = True)
        self.lstm2 = nn.LSTM(input_size = 32, hidden_size = 32, num_layers = 1, batch_first = True)
        self.lstm3 = nn.LSTM(input_size = 32, hidden_size = 32, num_layers = 1, batch_first = True)
        # Fully connected layer
        self.fc = nn.Linear(32, 4)
    
    def forward(self, x):
        out1, _= self.lstm1(x) # (h0.detach(), c0.detach())
        out2, _= self.lstm2(out1)
        out3, _= self.lstm3(out2)
        out3 = out3[:, -1, :]
        out = self.fc(out3)
        return out


# %%
# Instantiate the model with hyperparameters
model = Model()

# Define hyperparameters
n_epochs = 10
lr=0.001

# Define Loss, Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# %%
print('Training Start')
for epoch in range(n_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(myDataLoader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


# %%
testing_raw_list = [[1199,1160,1414,1711,1818,1509],[1199,1160,1412,1712,1814,1505],[1200,1160,1411,1712,1811,1501],[1200,1160,1411,1712,1808,1497]]
testing_X = []
for entry in testing_raw_list:
    testing_X.append(raw_cartesian_to_polar_angles(entry))
test_out = model(torch.from_numpy(np.array(testing_X).reshape(1,4,4)).float())
test_out_list = test_out.tolist()
test_out_polar = polar_angles_to_raw_cartesian(test_out_list[0])
print(test_out_polar)


# %%



