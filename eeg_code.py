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
dataset = data['EyesOpenClosed_Training']
num_windows = 32  # Training has 32 windows, Testing has 62 windows
window_duration = 23.6  # seconds


# TODO 1.a: Calculate the sampling frequency of the dataset
fs = len(dataset[0][0]) / window_duration
print('sampling_frequency:', fs)

# Fixing the label conversion (only convert if it's a string)
for i in range(num_windows):
    label = dataset[1, i]
    if isinstance(label, np.ndarray):
        label = label[0]  # Extract string from array
    if label == 'C':
        dataset[1, i] = 1
    elif label == 'O':
        dataset[1, i] = 0

# Concatenate windows
concat = np.array([])
labels = np.array([])
for i in range(num_windows):
    concat = np.concatenate([concat, dataset[0, i].flatten()])
    labels = np.concatenate([labels, np.full(len(dataset[0, i]), dataset[1, i])])


# TODO 1.b: Plot the concatenated signal and overlay a square wave
# Ensure the working directory is set correctly
script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(script_dir, 'plots')

# Create the 'plots' directory if it doesn't exist
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

plt.figure(figsize=(12, 6))
xaxis = np.arange(len(concat)) / fs
plt.plot(xaxis, concat, label='Data')
plt.plot(xaxis, labels * 300, label='300 = eyes closed; 0 = eyes open')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [μV]')
plt.title('Raw Training Data')
plt.savefig(os.path.join(plots_dir, 'raw_training_data.png'))
plt.grid(True)

# Save the plot to the 'plots' folder
plt.savefig(os.path.join(plots_dir, 'raw_training_data.png'))
plt.close()

# 1.c Separate data into Eyes Open and Eyes Closed
EyesOpen = []
EyesClosed = []
for i in range(num_windows):
    signal_data = dataset[0][i].flatten()  # Ensure it's 1D
    if dataset[1][i] == 1:  # closed
        EyesClosed.append(signal_data)
    elif dataset[1][i] == 0:  # open
        EyesOpen.append(signal_data)

EyesOpen = np.array(EyesOpen)
EyesClosed = np.array(EyesClosed)

# TODO 1.d: Calculate and plot the concatenated means and stds for each condition

# TODO 1.d: Calculate and plot the concatenated means and stds for each condition

# Calculate means and standard deviations
meanEO = np.mean(EyesOpen, axis=0)
meanEC = np.mean(EyesClosed, axis=0)
stdEO = np.std(EyesOpen, axis=0)
stdEC = np.std(EyesClosed, axis=0)

# Create a 'plots' directory if it doesn't exist
plots_dir = 'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Plot Eyes Open mean
plt.figure(figsize=(12, 6))
xaxis = np.arange(len(meanEO)) / fs
plt.plot(xaxis, meanEO, label='Eyes Open Mean', color='blue')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [μV]')
plt.title('Mean of Eyes Open')
plt.legend()
plt.grid(True)
plt.tight_layout()  # Ensures no clipping of the plot
plt.savefig(os.path.join(plots_dir, 'mean_eyes_open.png'))
plt.close()

# Plot Eyes Open standard deviation
plt.figure(figsize=(12, 6))
xaxis = np.arange(len(stdEO)) / fs
plt.plot(xaxis, stdEO, label='Eyes Open STD', color='blue')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [μV]')
plt.title('Standard Deviation of Eyes Open')
plt.legend()
plt.grid(True)
plt.tight_layout()  # Ensures no clipping of the plot
plt.savefig(os.path.join(plots_dir, 'std_eyes_open.png'))
plt.close()

# Plot Eyes Closed mean
plt.figure(figsize=(12, 6))
xaxis = np.arange(len(meanEC)) / fs
plt.plot(xaxis, meanEC, label='Eyes Closed Mean', color='red')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [μV]')
plt.title('Mean of Eyes Closed')
plt.legend()
plt.grid(True)
plt.tight_layout()  # Ensures no clipping of the plot
plt.savefig(os.path.join(plots_dir, 'mean_eyes_closed.png'))
plt.close()

# Plot Eyes Closed standard deviation
plt.figure(figsize=(12, 6))
xaxis = np.arange(len(stdEC)) / fs
plt.plot(xaxis, stdEC, label='Eyes Closed STD', color='red')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [μV]')
plt.title('Standard Deviation of Eyes Closed')
plt.legend()
plt.grid(True)
plt.tight_layout()  # Ensures no clipping of the plot
plt.savefig(os.path.join(plots_dir, 'std_eyes_closed.png'))
plt.close()

# TODO 1.e: Plot the smoothed concatenated means and stds
# Low and High Order for Moving Average smoothing
low_order = 10  # Low-order filter size
high_order = 100  # High-order filter size

# Smooth the mean and std signals using the low and high-order MA filters
meanEO_low = np.convolve(meanEO, np.ones(low_order)/low_order, mode='valid')
meanEO_high = np.convolve(meanEO, np.ones(high_order)/high_order, mode='valid')
meanEC_low = np.convolve(meanEC, np.ones(low_order)/low_order, mode='valid')
meanEC_high = np.convolve(meanEC, np.ones(high_order)/high_order, mode='valid')

stdEO_low = np.convolve(stdEO, np.ones(low_order)/low_order, mode='valid')
stdEO_high = np.convolve(stdEO, np.ones(high_order)/high_order, mode='valid')
stdEC_low = np.convolve(stdEC, np.ones(low_order)/low_order, mode='valid')
stdEC_high = np.convolve(stdEC, np.ones(high_order)/high_order, mode='valid')

# Adjust x-axis to match the length after smoothing
xaxis_low = np.arange(len(meanEO_low)) / fs
xaxis_high = np.arange(len(meanEO_high)) / fs

# Plot for Eyes Open (O) and Eyes Closed (C) - Mean and Std with Low and High Order smoothing
plt.figure(figsize=(12, 6))

# Eyes Closed Mean
plt.subplot(2, 2, 1)
plt.plot(xaxis_low, meanEC_low, label='Eyes Closed Mean (Low Order)', color='red')
plt.plot(xaxis_high, meanEC_high, label='Eyes Closed Mean (High Order)', color='orange')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [μV]')
plt.title('Eyes Closed Mean')
plt.legend()
plt.grid(True)

# Eyes Open Mean
plt.subplot(2, 2, 2)
plt.plot(xaxis_low, meanEO_low, label='Eyes Open Mean (Low Order)', color='blue')
plt.plot(xaxis_high, meanEO_high, label='Eyes Open Mean (High Order)', color='green')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [μV]')
plt.title('Eyes Open Mean')
plt.legend()
plt.grid(True)

# Eyes Closed Std
plt.subplot(2, 2, 3)
plt.plot(xaxis_low, stdEC_low, label='Eyes Closed Std (Low Order)', color='red')
plt.plot(xaxis_high, stdEC_high, label='Eyes Closed Std (High Order)', color='orange')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [μV]')
plt.title('Eyes Closed Std')
plt.legend()
plt.grid(True)

# Eyes Open Std
plt.subplot(2, 2, 4)
plt.plot(xaxis_low, stdEO_low, label='Eyes Open Std (Low Order)', color='blue')
plt.plot(xaxis_high, stdEO_high, label='Eyes Open Std (High Order)', color='green')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [μV]')
plt.title('Eyes Open Std')
plt.legend()
plt.grid(True)

# Adjust layout
plt.tight_layout()

# Save the plots to the 'plots' folder
plt.savefig(os.path.join(plots_dir, 'smoothed_means_and_stds.png'))
plt.close()



# For Eyes Open:
set_data = EyesOpen.flatten()  # Define dataset (EyesOpen or EyesClosed)
L = len(set_data)
Y = np.fft.fft(set_data)
f = fs * np.arange(L / 2) / L  # Adjust the frequency array to match P1's length

P2 = np.abs(Y / L)
P1 = P2[:int(L / 2)]  # Exclude the DC and Nyquist components as required
P1[1:-1] = 2 * P1[1:-1]  # Double the values, except for DC and Nyquist

# Fix: Trim f to match the length of P1
f = f[:len(P1)]  # Trim f to the length of P1

# Calculate Power Spectrum (PSD)
P = 10 * np.log10(P1**2)  # Convert to dB scale

# Plot Magnitude Spectrum and Power Spectrum (PSD) for Eyes Open
plt.figure(figsize=(12, 6))

# Magnitude Spectrum
plt.subplot(1, 2, 1)
plt.plot(f, P1)
plt.title('Magnitude Spectrum - Eyes Open')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [μV]')
plt.grid(True)

# Power Spectrum (PSD)
plt.subplot(1, 2, 2)
plt.plot(f, P)
plt.title('Power Spectrum (PSD) - Eyes Open')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power [dB/Hz]')
plt.grid(True)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'eyes_open_spectrum.png'))
plt.close()

# For Eyes Closed (same adjustment):
set_data = EyesClosed.flatten()  # Define dataset (EyesOpen or EyesClosed)
L = len(set_data)
Y = np.fft.fft(set_data)
f = fs * np.arange(L / 2) / L  # Adjust the frequency array to match P1's length

P2 = np.abs(Y / L)
P1 = P2[:int(L / 2)]  # Exclude the DC and Nyquist components as required
P1[1:-1] = 2 * P1[1:-1]  # Double the values, except for DC and Nyquist

# Fix: Trim f to match the length of P1
f = f[:len(P1)]  # Trim f to the length of P1

# Calculate Power Spectrum (PSD)
P = 10 * np.log10(P1**2)  # Convert to dB scale

# Plot Magnitude Spectrum and Power Spectrum (PSD) for Eyes Closed
plt.figure(figsize=(12, 6))

# Magnitude Spectrum
plt.subplot(1, 2, 1)
plt.plot(f, P1)
plt.title('Magnitude Spectrum - Eyes Closed')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [μV]')
plt.grid(True)

# Power Spectrum (PSD)
plt.subplot(1, 2, 2)
plt.plot(f, P)
plt.title('Power Spectrum (PSD) - Eyes Closed')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power [dB/Hz]')
plt.grid(True)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'eyes_closed_spectrum.png'))
plt.close()



