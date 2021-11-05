# +
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

# +
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
            X_data.append(temp.copy())
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


def plot_trajectory_from_tensor(coords: torch.Tensor):
    blue_x = coords.index_select(1,torch.tensor([4])).numpy()
    blue_y = coords.index_select(1,torch.tensor([5])).numpy()
    plt.scatter(blue_x,blue_y)


