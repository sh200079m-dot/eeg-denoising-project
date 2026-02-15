from wavelet_utils import wavelet_decompose, wavelet_reconstruct
from adaptive_threshold import adaptive_threshold
from threshold_rules import hard_threshold
import numpy as np


def adaptive_wavelet_denoise(signal, wavelet="db4", level=5):
    # Perform wavelet decomposition on the input signal up to the specified level
    coeffs_all = wavelet_decompose(signal, wavelet, level)
    # Initialize a list to store denoised coefficients for each channel
    denoised_coeffs = []

    # Iterate over each channel's coefficients
    for ch_idx, coeffs in enumerate(coeffs_all):
        # Calculate the energy of the current channel (mean of squared signal values)
        channel_energy = np.mean(signal[ch_idx] ** 2)

        # Iterate over each decomposition level (excluding the approximation coefficients at level 0)
        for lvl in range(1, len(coeffs)):
            # Compute an adaptive threshold based on the coefficients, channel energy, and current level
            T = adaptive_threshold(coeffs[lvl], channel_energy, lvl)
            # Apply hard thresholding to the detail coefficients
            coeffs[lvl] = hard_threshold(coeffs[lvl], T)

        # Append the denoised coefficients for this channel to the list
        denoised_coeffs.append(coeffs)

    # Reconstruct the denoised signal from the thresholded coefficients
    return wavelet_reconstruct(denoised_coeffs, wavelet)