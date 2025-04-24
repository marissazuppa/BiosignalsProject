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

# Calculate the sampling frequency of the dataset
fs = int(len(dataset[0][0]) / window_duration)

# Concatenate data windows together
# Change labels of C and O into numerical values (C=1, O=0)
for i in range(num_windows):
    if dataset[1][i] == 'C':  # if closed
        dataset[1][i] = 1
    if dataset[1][i] == 'O':  # if open
        dataset[1][i] = 0

# Concatenate windows
concat = np.array([])
labels = np.array([])
for i in range(num_windows):
    concat = np.concatenate([concat, dataset[0][i].flatten()])
    labels = np.concatenate([labels, np.full(len(dataset[0][i]), dataset[1][i])])

# Plot the concatenated signal and overlay a square wave
plt.figure(figsize=(12, 6))
xaxis = np.arange(len(concat)) / fs
plt.plot(xaxis, concat, label='Data')
plt.plot(xaxis, labels * 300, label='300 = eyes closed; 0 = eyes open')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [μV]')
plt.title('Raw Training Data')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# 1.c Separate data into Eyes Open and Eyes Closed
EyesOpen = []
EyesClosed = []
for i in range(num_windows):
    if dataset[1][i] == 1:  # closed
        EyesClosed.append(dataset[0][i])
    if dataset[1][i] == 0:  # open
        EyesOpen.append(dataset[0][i])

EyesOpen = np.array(EyesOpen)
EyesClosed = np.array(EyesClosed)

# 1.d: Calculate and plot the concatenated means and stds for each condition
meanEO = np.mean(EyesOpen, axis=0)
meanEC = np.mean(EyesClosed, axis=0)
stdEO = np.std(EyesOpen, axis=0)
stdEC = np.std(EyesClosed, axis=0)

# Plot the concatenated means and standard deviations
plt.figure(figsize=(12, 6))
xaxis = np.arange(len(meanEO)) / fs
plt.plot(xaxis, meanEO, label='Eyes Open Mean')
plt.plot(xaxis, meanEC, label='Eyes Closed Mean')
plt.fill_between(xaxis, meanEO - stdEO, meanEO + stdEO, alpha=0.2, label='Eyes Open Std Dev')
plt.fill_between(xaxis, meanEC - stdEC, meanEC + stdEC, alpha=0.2, label='Eyes Closed Std Dev')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [μV]')
plt.title('Mean and Standard Deviation of Eyes Open/Closed')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
xaxis = np.arange(len(meanEO)) / fs
plt.plot(xaxis, meanEO, label='Eyes Open Mean')
plt.plot(xaxis, meanEC, label='Eyes Closed Mean')
plt.fill_between(xaxis, meanEO - stdEO, meanEO + stdEO, alpha=0.2)
plt.fill_between(xaxis, meanEC - stdEC, meanEC + stdEC, alpha=0.2)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [μV]')
plt.title('Mean and Standard Deviation of Eyes Open/Closed')
plt.legend()
plt.grid(True)
plt.show()

# Plot the smoothed concatenated means and stds
# Using a simple moving average for smoothing
window_size = 100
meanEO_smooth = np.convolve(meanEO, np.ones(window_size)/window_size, mode='valid')
meanEC_smooth = np.convolve(meanEC, np.ones(window_size)/window_size, mode='valid')
stdEO_smooth = np.convolve(stdEO, np.ones(window_size)/window_size, mode='valid')
stdEC_smooth = np.convolve(stdEC, np.ones(window_size)/window_size, mode='valid')

# Plot the smoothed data
plt.figure(figsize=(12, 6))
xaxis = np.arange(len(meanEO_smooth)) / fs
plt.plot(xaxis, meanEO_smooth, label='Eyes Open Mean (Smoothed)')
plt.plot(xaxis, meanEC_smooth, label='Eyes Closed Mean (Smoothed)')
plt.fill_between(xaxis, meanEO_smooth - stdEO_smooth, meanEO_smooth + stdEO_smooth, alpha=0.2, label='Eyes Open Std Dev (Smoothed)')
plt.fill_between(xaxis, meanEC_smooth - stdEC_smooth, meanEC_smooth + stdEC_smooth, alpha=0.2, label='Eyes Closed Std Dev (Smoothed)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [μV]')
plt.title('Smoothed Mean and Standard Deviation of Eyes Open/Closed')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
xaxis = np.arange(len(meanEO_smooth)) / fs
plt.plot(xaxis, meanEO_smooth, label='Eyes Open Mean (Smoothed)')
plt.plot(xaxis, meanEC_smooth, label='Eyes Closed Mean (Smoothed)')
plt.fill_between(xaxis, meanEO_smooth - stdEO_smooth, meanEO_smooth + stdEO_smooth, alpha=0.2)
plt.fill_between(xaxis, meanEC_smooth - stdEC_smooth, meanEC_smooth + stdEC_smooth, alpha=0.2)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [μV]')
plt.title('Smoothed Mean and Standard Deviation')
plt.legend()
plt.grid(True)
plt.show()

# 1.f Calculate the mean and standard deviation
print(f"Eyes Closed - Mean: {np.mean(meanEC):.2f}, Std: {np.mean(stdEC):.2f}")
print(f"Eyes Open - Mean: {np.mean(meanEO):.2f}, Std: {np.mean(stdEO):.2f}")

# Calculate the magnitude spectrum and PSD using FFT
set_data = EyesClosed.flatten()  # define dataset (EyesClosed or EyesOpen)

# FFT
L = len(set_data)
T = 1/fs
Y = np.fft.fft(set_data)
f = fs * np.arange(L/2 + 1) / L

P2 = np.abs(Y/L)
P1 = P2[:int(L/2) + 1]
P1[1:-1] = 2 * P1[1:-1]

# Calculate the single-sided Power Spectrum
P = 10 * np.log10(P1**2)  # single-sided Power spectrum (PSD)

# Plot the single-sided magnitude spectrum and PSD
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(f, P1)
plt.title('Single Sided Magnitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude [μV]')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(f, P)
plt.title('Single Sided Power Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power [μV²/Hz]')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot the spectrogram
plt.figure(figsize=(12, 6))
plt.specgram(concat, NFFT=1024, Fs=fs, noverlap=int(fs))
plt.colorbar(label='Power/Frequency [dB/Hz]')
plt.title('Spectrogram of Concatenated Data')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.show()

# 2. Data Conditioning (Filtering)
filt_order = 4  # define filter order
filt_cutoff = 40  # define filter cutoff frequency
set_data = EyesClosed.flatten()

# Butterworth lowpass filter
nyquist = fs / 2
normalized_cutoff = filt_cutoff / nyquist
b, a = signal.butter(filt_order, normalized_cutoff, btype='low')
filtered_data = signal.filtfilt(b, a, set_data)

# Plot the magnitude spectrum or PSD of the filtered data
f, P1, P = compute_psd(filtered_data, fs)
plot_psd(f, P1, P, 'Filtered Data')

# 3. Implement the classifier
dataset = data['EyesOpenClosed_Training'][0]

# Identify alpha-band frequencies and power threshold
alpha_range = (8, 13)  # Hz, typical alpha band range
threshold = 2.5  # adjust this value based on analysis of training data

# Loop through windowed data and classify
classified_dataset = []
for i in range(num_windows):
    window_data = dataset[0][i]
    
    # Compute FFT
    L = len(window_data)
    Y = np.fft.fft(window_data)
    f = fs * np.arange(L/2 + 1) / L
    
    P2 = np.abs(Y/L)
    P1 = P2[:int(L/2) + 1]
    P1[1:-1] = 2 * P1[1:-1]
    P = 10 * np.log10(P1**2)
    
    # Find alpha band indices
    fst = np.argmin(np.abs(f - alpha_range[0]))
    fe = np.argmin(np.abs(f - alpha_range[1]))
    
    # Calculate average power in alpha band
    avg_mag = np.mean(P1[fst:fe])
    avg_pwr = np.mean(P[fst:fe])
    
    # Classify based on threshold
    flag = 1 if avg_mag >= threshold else 0
    
    classified_dataset.append({
        'data': window_data,
        'classification': flag,
        'avg_mag': avg_mag,
        'avg_pwr': avg_pwr
    })

# Compare classification results
true_labels = np.array([1 if label == 'C' else 0 for label in dataset[1]])
predicted_labels = np.array([item['classification'] for item in classified_dataset])
accuracy = 100 * np.mean(predicted_labels == true_labels)
print(f"Classification Accuracy: {accuracy:.2f}%")
# TODO 3.d: Apply classification to testing data
test_data = loadmat('EyesOpenClosed_Testing.mat')
test_dataset = test_data['EyesOpenClosed_Testing'][0]
num_test_windows = 62  # Testing has 62 windows

# Loop through testing data and classify
test_classified_dataset = []
for i in range(num_test_windows):
    window_data = test_dataset[0][i]
    
    # Compute FFT
    L = len(window_data)
    Y = np.fft.fft(window_data)
    f = fs * np.arange(L/2 + 1) / L
    
    P2 = np.abs(Y/L)
    P1 = P2[:int(L/2) + 1]
    P1[1:-1] = 2 * P1[1:-1]
    P = 10 * np.log10(P1**2)
    
    # Find alpha band indices
    fst = np.argmin(np.abs(f - alpha_range[0]))
    fe = np.argmin(np.abs(f - alpha_range[1]))
    
    # Calculate average power in alpha band
    avg_mag = np.mean(P1[fst:fe])
    avg_pwr = np.mean(P[fst:fe])
    
    # Classify based on threshold
    flag = 1 if avg_mag >= threshold else 0
    
    test_classified_dataset.append({
        'data': window_data,
        'classification': flag,
        'avg_mag': avg_mag,
        'avg_pwr': avg_pwr
    })

# Compare classification results for testing data
true_test_labels = np.array([1 if label == 'C' else 0 for label in test_dataset[1]])
predicted_test_labels = np.array([item['classification'] for item in test_classified_dataset])
test_accuracy = 100 * np.mean(predicted_test_labels == true_test_labels)
print(f"Testing Classification Accuracy: {test_accuracy:.2f}%")

# Plot testing data results
# Concatenate test data windows together
test_concat = np.array([])
test_labels = np.array([])
for i in range(num_test_windows):
    test_concat = np.concatenate([test_concat, test_dataset[0][i].flatten()])
    test_labels = np.concatenate([test_labels, np.full(len(test_dataset[0][i]), 1 if test_dataset[1][i] == 'C' else 0)])

# Plot the concatenated test signal and overlay a square wave
plt.figure(figsize=(12, 6))
xaxis = np.arange(len(test_concat)) / fs
plt.plot(xaxis, test_concat, label='Test Data')
plt.plot(xaxis, test_labels * 300, label='300 = eyes closed; 0 = eyes open')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [μV]')
plt.title('Raw Testing Data')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Calculate and plot the means and stds for test data
test_EyesOpen = []
test_EyesClosed = []
for i in range(num_test_windows):
    if test_dataset[1][i] == 'C':  # closed
        test_EyesClosed.append(test_dataset[0][i])
    if test_dataset[1][i] == 'O':  # open
        test_EyesOpen.append(test_dataset[0][i])

test_EyesOpen = np.array(test_EyesOpen)
test_EyesClosed = np.array(test_EyesClosed)

test_meanEO = np.mean(test_EyesOpen, axis=0)
test_meanEC = np.mean(test_EyesClosed, axis=0)
test_stdEO = np.std(test_EyesOpen, axis=0)
test_stdEC = np.std(test_EyesClosed, axis=0)

# Plot the concatenated means and standard deviations for test data
plt.figure(figsize=(12, 6))
xaxis = np.arange(len(test_meanEO)) / fs
plt.plot(xaxis, test_meanEO, label='Eyes Open Mean (Test)')
plt.plot(xaxis, test_meanEC, label='Eyes Closed Mean (Test)')
plt.fill_between(xaxis, test_meanEO - test_stdEO, test_meanEO + test_stdEO, alpha=0.2, label='Eyes Open Std Dev (Test)')
plt.fill_between(xaxis, test_meanEC - test_stdEC, test_meanEC + test_stdEC, alpha=0.2, label='Eyes Closed Std Dev (Test)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [μV]')
plt.title('Mean and Standard Deviation of Eyes Open/Closed (Test Data)')
plt.legend()
plt.grid(True)
plt.show()

def compute_psd(data, fs):
    """Helper function to compute PSD"""
    L = len(data)
    Y = np.fft.fft(data)
    f = fs * np.arange(L/2 + 1) / L
    
    P2 = np.abs(Y/L)
    P1 = P2[:int(L/2) + 1]
    P1[1:-1] = 2 * P1[1:-1]
    P = 10 * np.log10(P1**2)
    
    return f, P1, P

def plot_psd(f, P1, P, title_prefix):
    """Helper function to plot PSD"""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(f, P1)
    plt.title(f'{title_prefix} Single Sided Magnitude Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude [μV]')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(f, P)
    plt.title(f'{title_prefix} Single Sided Power Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power [μV²/Hz]')
    plt.grid(True)
    plt.tight_layout()
    plt.show() 
