import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Load training data
data = loadmat('EyesOpenClosed_Training.mat')
dataset = data['EyesOpenClosed_Training'][0]
num_windows = 32
window_duration = 23.6  # seconds

# Correct number of samples per window
num_samples_per_window = len(dataset[0][0])  # Should be 4097
fs = num_samples_per_window / window_duration
print('Sampling frequency:', fs)

# Convert labels C/O to 1/0
for i in range(num_windows):
    if dataset[1][i] == 'C':
        dataset[1][i] = 1
    elif dataset[1][i] == 'O':
        dataset[1][i] = 0

# Concatenate data and labels
concat = np.concatenate([dataset[0][i].flatten() for i in range(num_windows)])
labels = np.concatenate([np.full(len(dataset[0][i]), dataset[1][i]) for i in range(num_windows)])

# Generate time axis
total_samples = len(concat)
xaxis = np.arange(total_samples) / fs

# Plot
plt.figure(figsize=(12, 6))
plt.plot(xaxis, concat, label='Data')
plt.plot(xaxis, labels * 300, label='300 = eyes closed; 0 = eyes open', linestyle='--')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [μV]')
plt.title('Raw Training Data')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print(f"Time axis range: {xaxis[0]} to {xaxis[-1]}")





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

# TODO 1.d: Calculate and plot the concatenated means and stds for each condition
meanEO = np.mean(EyesOpen, axis=0)
meanEC = np.mean(EyesClosed, axis=0)
stdEO = np.std(EyesOpen, axis=0)
stdEC = np.std(EyesClosed, axis=0)

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

# TODO 1.e: Plot the smoothed concatenated means and stds
# Using a simple moving average for smoothing
window_size = 100
meanEO_smooth = np.convolve(meanEO, np.ones(window_size)/window_size, mode='valid')
meanEC_smooth = np.convolve(meanEC, np.ones(window_size)/window_size, mode='valid')
stdEO_smooth = np.convolve(stdEO, np.ones(window_size)/window_size, mode='valid')
stdEC_smooth = np.convolve(stdEC, np.ones(window_size)/window_size, mode='valid')

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

# TODO 1.g: Calculate the single-sided Power Spectrum
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
