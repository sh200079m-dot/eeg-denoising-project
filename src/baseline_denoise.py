from wavelet_utils import wavelet_decompose, wavelet_reconstruct
from threshold_rules import rigrsure, hard_threshold
import numpy as np


def baseline_wavelet_denoise(signal, wavelet="db4", level=5):
    # Perform wavelet decomposition on the input signal up to the specified level
    coeffs_all = wavelet_decompose(signal, wavelet, level)

    # Iterate over each channel's coefficients
    for ch_idx, coeffs in enumerate(coeffs_all):
        # Skip the approximation coefficients (coeffs[0]) - we don't modify them
        for i in range(1, len(coeffs)):
            # Calculate threshold using the rigrsure (Stein's Unbiased Risk Estimate) method
            T = rigrsure(coeffs[i])
            # Apply hard thresholding to the detail coefficients
            coeffs[i] = hard_threshold(coeffs[i], T)

        # Store the denoised coefficients for this channel
        coeffs_all[ch_idx] = coeffs

    # Reconstruct the denoised signal from the thresholded coefficients
    denoised = wavelet_reconstruct(coeffs_all, wavelet)
    return denoised