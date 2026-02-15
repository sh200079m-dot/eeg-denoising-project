import numpy as np

def add_baseline_wander(signal, sfreq, freq=0.3, amplitude_ratio=0.05):
    """
    Add low-frequency baseline drift
    """
    # Get the number of channels and time points from the signal shape
    n_channels, n_samples = signal.shape
    # Create a time vector based on sampling frequency
    t = np.arange(n_samples) / sfreq

    # Generate a low-frequency sinusoidal drift
    drift = np.sin(2 * np.pi * freq * t)
    # Repeat the drift pattern for all channels
    drift = np.tile(drift, (n_channels, 1))

    # Add the scaled drift to the original signal
    return signal + amplitude_ratio * drift

def add_line_noise(signal, sfreq, freq=50, snr_db=5):
    """
    Add sinusoidal line noise to EEG signal
    """
    # Get the number of channels and time points from the signal shape
    n_channels, n_samples = signal.shape
    # Create a time vector based on sampling frequency
    t = np.arange(n_samples) / sfreq

    # Generate sinusoidal noise at the specified frequency (e.g., 50 Hz power line noise)
    noise = np.sin(2 * np.pi * freq * t)
    # Repeat the noise pattern for all channels
    noise = np.tile(noise, (n_channels, 1))

    # Calculate the power of the original signal
    signal_power = np.mean(signal ** 2)
    # Calculate the power of the generated noise
    noise_power = np.mean(noise ** 2)

    # Determine the desired noise power based on the input SNR (in dB)
    desired_noise_power = signal_power / (10 ** (snr_db / 10))
    # Scale the noise to achieve the desired SNR
    noise *= np.sqrt(desired_noise_power / noise_power)

    # Add the scaled noise to the original signal
    return signal + noise