import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat
import os

# University of Florida, Herbert Wertheim College of Engineering
# J. Crayton Pruitt Department of Biomedical Engineering
# BME 3508 - Biomedical Signals and Systems - Spring 2025
# Python conversion of Final Project Helper Code

# Load training data
data = loadmat('EyesOpenClosed_Training.mat')
dataset = data['EyesOpenClosed_Training'][0]
num_windows = 32  # Training has 32 windows, Testing has 62 windows
window_duration = 23.6  # seconds

# TODO 1.a: Calculate the sampling frequency of the dataset
fs = len(dataset[0][0]) / window_duration
print('sampling_frequency:' fs)
