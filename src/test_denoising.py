"""
Comprehensive EEG Wavelet Denoising Test Script
------------------------------------------------
This script evaluates baseline (Rigrsure + Hard) and
proposed adaptive wavelet denoising methods on:

- Two subjects (Control & Parkinson)
- Two wavelets (db4 and dmey)

Quantitative metrics:
- SNR
- RMSE
- Correlation Coefficient

Author: Shaghayegh Masoudian
Course: Advanced Digital Signal Processing (ADSP)
"""

import numpy as np
from load_data import load_eeg
from preprocessing import basic_preprocessing
from segment import extract_segment
from add_noise import add_line_noise, add_baseline_wander
from baseline_denoise import baseline_wavelet_denoise
from adaptive_denoise import adaptive_wavelet_denoise
from evaluate_methods import evaluate


# ============================
# Configuration
# ============================

subjects = ["sub-001", "sub-101"]      # Control & Parkinson
wavelets = ["db4", "dmey"]             # Reference & Literature-best wavelet
snr_db = 10                            # Controlled SNR for line noise
baseline_drift_ratio = 0.05            # Baseline wander amplitude ratio


# ============================
# Main Evaluation Loop
# ============================

for subject in subjects:

    print("\n====================================================")
    print(f"Processing Subject: {subject}")
    print("====================================================")

    # ----------------------------
    # Load & Preprocess EEG
    # ----------------------------
    raw = load_eeg(subject)
    raw = basic_preprocessing(raw)

    # Extract a 10-second middle segment
    segment, sfreq = extract_segment(raw)

    # Convert to microvolts (ground truth reference)
    clean = segment * 1e6

    # ----------------------------
    # Add Controlled Artificial Noise
    # ----------------------------
    noisy = add_line_noise(clean, sfreq, snr_db=snr_db)
    noisy = add_baseline_wander(noisy, sfreq,
                                amplitude_ratio=baseline_drift_ratio)

    # ----------------------------
    # Wavelet Comparison
    # ----------------------------
    for w in wavelets:

        print(f"\n---- Wavelet: {w} ----")

        # Baseline method (Rigrsure + Hard threshold)
        baseline = baseline_wavelet_denoise(noisy, wavelet=w)

        # Proposed adaptive method
        adaptive = adaptive_wavelet_denoise(noisy, wavelet=w)

        # ----------------------------
        # Quantitative Evaluation
        # ----------------------------
        results = evaluate(clean, noisy, baseline, adaptive)

        print("Evaluation Metrics:")
        for k, v in results.items():
            print(f"{k}: {v:.3f}")

print("\n\nEvaluation Completed Successfully.")