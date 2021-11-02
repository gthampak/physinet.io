# +
import os
import shutil
import random; random.seed(42)

import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from functools import partial
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
#import tensorflow as tf # tensorflow-gpu==2.0.0
#from tensorflow.python.client import device_lib 
#print(device_lib.list_local_devices())
#import cv2
# -

# !pip install 'ray[tune]'

# +
# training data
train_dir = '/raid/cs152/zxaa2018/penndulum/train_and_test_split/dpc_dataset_traintest_4_200_csv/train'
train_dir_video = '/raid/cs152/zxaa2018/penndulum//train_and_test_split/dpc_dataset_traintest_4_200_h264/train'

# test data
test_inputs_dir = '/raid/cs152/zxaa2018/penndulum/train_and_test_split/dpc_dataset_traintest_4_200_csv/test_inputs/'
test_targets_dir = '/raid/cs152/zxaa2018/penndulum/train_and_test_split/dpc_dataset_traintest_4_200_csv/test_targets/'
test_targets_video = '/raid/cs152/zxaa2018/penndulum/train_and_test_split/dpc_dataset_traintest_4_200_h264/test_targets/'

# validation data
validation_inputs_dir = '/raid/cs152/zxaa2018/penndulum/train_and_test_split/dpc_dataset_traintest_4_200_csv/validation_inputs/'
validation_targets_dir = '/raid/cs152/zxaa2018/penndulum/rain_and_test_split/dpc_dataset_traintest_4_200_csv/validation_targets/'
validation_targets_video = '/raid/cs152/zxaa2018/penndulum/train_and_test_split/dpc_dataset_traintest_4_200_h264/validation_targets/'

# +
# some constants
DEFAULT_X_RED, DEFAULT_Y_RED = (240, 232)

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


# -

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


def parse_testing_annotations(csv_file_X,csv_file_Y):
    '''
    Parse the testing annotations from a CSV file.
    Return a x-y pair
    '''
    print(csv_file_X[-7:],csv_file_Y[-7:])
    X_data = [] #a list containing 4 frames
    y_data = [] #a list containing 200 frames
    # load X file
    f = pd.read_csv(csv_file_X, header=None, delim_whitespace=True, engine='python')
    for i, row in f.iterrows():
        X_data.append(raw_cartesian_to_polar_angles(row.to_list()))
    f = pd.read_csv(csv_file_Y, header=None, delim_whitespace=True, engine='python')
    for i, row in f.iterrows():
        y_data.append(raw_cartesian_to_polar_angles(row.to_list()))
        #print(row)

    return X_data, y_data


class DoublePendulumDataset(torch.utils.data.Dataset):
    def __init__(self,X_list,y_list):
        self.sample_list = list(zip(X_list, y_list))
    
    def __getitem__(self,index):
        X_sample,y_sample = self.sample_list[index]
        return torch.from_numpy(np.array(X_sample)).float(),torch.from_numpy(np.array(y_sample)).float()
    
    def __len__(self):
        return len(self.sample_list)


class LSTMModel(nn.Module):
    def __init__(self):
        # We want a model of 4 layer LSTM with 32 features output, and a dense layer to form the 4 feature output.
        super(LSTMModel, self).__init__()

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


print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))


def load_pend_data():
    # load in all separate files
    X_train = []
    y_train = []
    for filename in tqdm([x for x in os.listdir(train_dir) if not x.startswith('.')]):
        # load in a file
        X_data, y_data = parse_training_annotations(os.path.join(train_dir, filename))

        X_train = X_train + X_data
        y_train = y_train + y_data
    trainset = DoublePendulumDataset(X_train,y_train)
    
    X_test = []
    y_test = []
    isPrint = True
    #for filename in tqdm([x for x in os.listdir(test_inputs_dir) if not x.startswith('.')]):
    for i in range(60):
        # load in a file
        X_data, y_data = parse_testing_annotations(test_inputs_dir + str(i)+'.csv',test_targets_dir + str(i)+'.csv')
        X_test.append(X_data)
        y_test.append(y_data)
    testset = DoublePendulumDataset(X_test,y_test)
    
    return trainset, testset


def train_lstm(config, checkpoint_dir=None, data_dir=None):
    net = LSTMModel()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config["lr"], momentum=0.9)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, testset = load_pend_data()

    test_abs = int(len(trainset) * 0.8)
    
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)
    
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")


def test_accuracy(net, device="cuda:0"):
    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def main(num_samples=10, max_num_epochs=10):
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16])
    }
