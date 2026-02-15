import numpy as np

def compute_snr(clean, noisy):
    # Calculate the power of the clean signal (mean of squared values)
    signal_power = np.mean(clean ** 2)
    # Calculate the power of the noise (mean of squared differences between clean and noisy)
    noise_power = np.mean((clean - noisy) ** 2)
    # Return the Signal-to-Noise Ratio in decibels (dB)
    return 10 * np.log10(signal_power / noise_power)

def compute_rmse(clean, denoised):
    # Calculate the Root Mean Square Error between clean and denoised signals
    return np.sqrt(np.mean((clean - denoised) ** 2))

def compute_corr(clean, denoised):
    # Calculate the Pearson correlation coefficient between clean and denoised signals
    # Flatten the arrays to 1D and extract the correlation value (off-diagonal element)
    return np.corrcoef(clean.flatten(), denoised.flatten())[0, 1]