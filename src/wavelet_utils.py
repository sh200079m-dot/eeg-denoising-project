import pywt
import numpy as np


def wavelet_decompose(signal, wavelet="db4", level=5):
    """
    signal: shape (n_channels, n_samples)
    return: list of coeffs per channel
    Perform wavelet decomposition on each channel of the input signal
    """
    # Initialize an empty list to store coefficients for all channels
    coeffs_all = []
    # Iterate over each channel in the signal
    for ch in signal:
        # Perform wavelet decomposition on the current channel
        coeffs = pywt.wavedec(ch, wavelet, level=level)
        # Append the coefficients to the list
        coeffs_all.append(coeffs)
    return coeffs_all


def wavelet_reconstruct(coeffs_all, wavelet="db4"):
    """
    Reconstruct the signal from wavelet coefficients for all channels
    """
    # Initialize an empty list to store reconstructed channels
    reconstructed = []
    # Iterate over the coefficients for each channel
    for coeffs in coeffs_all:
        # Perform wavelet reconstruction for the current channel
        rec = pywt.waverec(coeffs, wavelet)
        # Append the reconstructed channel to the list
        reconstructed.append(rec)
    # Convert the list of reconstructed channels to a numpy array
    return np.array(reconstructed)