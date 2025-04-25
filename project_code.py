import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.fft import fft
from scipy.signal import welch

# Load data
mat = loadmat('eeg_data.mat')
dataset = mat['eeg']
labels = mat['labels'].flatten()

# Step 1.a: Calculate sampling frequency
window_duration = 2  # seconds
fs = len(dataset[0][0]) / window_duration
print(f"Sampling Frequency: {fs:.2f} Hz")

# Step 1.b: Plot concatenated EEG signal with labels
concatenated_signal = np.hstack(dataset[0])
time = np.arange(len(concatenated_signal)) / fs
label_overlay = np.repeat(labels * 300, dataset[0][0].shape[0])

plt.figure(figsize=(12, 4))
plt.plot(time, concatenated_signal, label='EEG Signal')
plt.plot(time, label_overlay, label='Label Overlay (Scaled)')
plt.title('Concatenated EEG Signal with Eye State Labels')
plt.xlabel('Time (s)')
plt.ylabel('EEG Amplitude')
plt.legend()
plt.tight_layout()
plt.savefig('concatenated_signal.png', dpi=300)
plt.show()

# Step 1.c: Separate Eyes Open (EO) and Eyes Closed (EC) data
EyesOpen = np.array([trial for trial, label in zip(dataset[0], labels) if label == 0])
EyesClosed = np.array([trial for trial, label in zip(dataset[0], labels) if label == 1])
print("EO shape:", EyesOpen.shape)
print("EC shape:", EyesClosed.shape)

# Step 1.d: Compute and plot mean and std for EC and EO
meanEO = np.mean(EyesOpen, axis=0)
stdEO = np.std(EyesOpen, axis=0)
meanEC = np.mean(EyesClosed, axis=0)
stdEC = np.std(EyesClosed, axis=0)

plt.figure(figsize=(12, 4))
plt.plot(meanEO, label='Mean Eyes Open')
plt.fill_between(np.arange(len(meanEO)), meanEO-stdEO, meanEO+stdEO, alpha=0.3)
plt.plot(meanEC, label='Mean Eyes Closed')
plt.fill_between(np.arange(len(meanEC)), meanEC-stdEC, meanEC+stdEC, alpha=0.3)
plt.legend()
plt.title('Mean EEG Signal with Standard Deviation')
plt.tight_layout()
plt.savefig('mean_std_plot.png', dpi=300)
plt.show()

# Step 1.e: Apply moving average filter with different orders
orders = [2, 4, 6, 8]
plt.figure(figsize=(12, 4))
for w in orders:
    smooth = np.convolve(meanEC, np.ones(w)/w, mode='valid')
    plt.plot(smooth, label=f'Order {w}')
plt.title('Moving Average Smoothing of Eyes Closed Mean Signal')
plt.legend()
plt.tight_layout()
plt.savefig('moving_average_smoothing.png', dpi=300)
plt.show()

# Step 1.f: Interpretation summary statistics
print("Mean EC:", np.mean(meanEC))
print("Mean EO:", np.mean(meanEO))
print("STD EC:", np.std(meanEC))
print("STD EO:", np.std(meanEO))
print("Difference in mean (EC - EO):", np.mean(meanEC) - np.mean(meanEO))

# Step 1.g: FFT
EC_fft = np.abs(fft(meanEC))
EO_fft = np.abs(fft(meanEO))
freqs = np.fft.fftfreq(len(meanEC), d=1/fs)

plt.figure(figsize=(12, 4))
plt.plot(freqs[:len(freqs)//2], EC_fft[:len(freqs)//2], label='Eyes Closed')
plt.plot(freqs[:len(freqs)//2], EO_fft[:len(freqs)//2], label='Eyes Open')
plt.title('FFT Magnitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.tight_layout()
plt.savefig('fft_plot.png', dpi=300)
plt.show()

# Step 1.h: Power Spectral Density
fEC, psdEC = welch(meanEC, fs=fs)
fEO, psdEO = welch(meanEO, fs=fs)

plt.figure(figsize=(12, 4))
plt.semilogy(fEC, psdEC, label='Eyes Closed')
plt.semilogy(fEO, psdEO, label='Eyes Open')
plt.title('Power Spectral Density')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.legend()
plt.tight_layout()
plt.savefig('psd_plot.png', dpi=300)
plt.show()

# Step 1.i: PSD Interpretation (Alpha band 8–13 Hz)
alpha_idx = (fEC >= 8) & (fEC <= 13)
print("Mean Alpha Power EC:", np.mean(psdEC[alpha_idx]))
print("Mean Alpha Power EO:", np.mean(psdEO[alpha_idx]))

# Step 1.j: Spectrogram
plt.figure(figsize=(12, 4))
plt.specgram(concatenated_signal, NFFT=256, Fs=fs, noverlap=128, cmap='viridis')
plt.colorbar(label='Power (dB)')
plt.title('Spectrogram of EEG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()
plt.savefig('spectrogram.png', dpi=300)
plt.show()
