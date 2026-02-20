"""
Comprehensive EEG Wavelet Denoising Test Script
------------------------------------------------
This script evaluates baseline (Rigrsure + Hard) and
proposed adaptive wavelet denoising methods on healthy subjects (sub-101 to sub-149)
"""

import numpy as np
from load_data import load_eeg
from preprocessing import basic_preprocessing
from segment import extract_segment
from add_noise import add_line_noise, add_baseline_wander
from baseline_denoise import baseline_wavelet_denoise
from adaptive_denoise import adaptive_wavelet_denoise
from evaluate_methods import evaluate
import pandas as pd
import os


# ============================
# Configuration
# ============================

# ✅ فقط افراد سالم (sub-101 تا sub-149)
healthy_subjects = [f"sub-{i:03d}" for i in range(101, 150)]
print(f"Number of healthy subjects: {len(healthy_subjects)}")  # باید ۴۹ باشه

subjects = healthy_subjects  # برای اجرا روی همه ۴۹ نفر
# subjects = healthy_subjects[:5]  # برای تست فقط ۵ نفر اول (sub-101 تا sub-105)

wavelets = ["db4", "dmey"]
snr_db = 10
baseline_drift_ratio = 0.05


# ============================
# Main Evaluation Loop
# ============================

all_results = []
successful = 0
failed = []

print("=" * 70)
print(f"Starting processing of {len(subjects)} healthy subjects")
print("=" * 70)

for idx, subject in enumerate(subjects):
    print(f"\n[{idx+1}/{len(subjects)}] Processing: {subject}")
    print("-" * 50)

    try:
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
            print(f"  Wavelet: {w}")

            # Baseline method (Rigrsure + Hard threshold)
            baseline = baseline_wavelet_denoise(noisy, wavelet=w)

            # Proposed adaptive method
            adaptive = adaptive_wavelet_denoise(noisy, wavelet=w)

            # ----------------------------
            # Quantitative Evaluation
            # ----------------------------
            results = evaluate(clean, noisy, baseline, adaptive)

            # Add metadata
            results['subject'] = subject
            results['wavelet'] = w
            results['group'] = 'healthy'
            all_results.append(results)

            print(f"    SNR_baseline={results['Baseline SNR']:.3f}, "
                  f"SNR_adaptive={results['Adaptive SNR']:.3f}")

        successful += 1

    except Exception as e:
        print(f"  ❌ Error with {subject}: {e}")
        failed.append(subject)

print("\n" + "=" * 70)
print(f"✅ Successfully processed: {successful}/{len(subjects)}")
print(f"❌ Failed: {len(failed)}")
if failed:
    print(f"Failed subjects: {failed}")
print("=" * 70)


# ============================
# Save Results
# ============================
if all_results:
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Save to CSV
    output_file = "healthy_49_subjects_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n📁 Results saved to: {output_file}")
    
    # Calculate statistics
    print("\n📊 Summary Statistics (Healthy Subjects - 49 people):")
    print("=" * 60)
    
    for wavelet in wavelets:
        print(f"\n{wavelet.upper()} Wavelet:")
        subset = df[df.wavelet == wavelet]
        
        baseline_snr = subset['Baseline SNR'].mean()
        adaptive_snr = subset['Adaptive SNR'].mean()
        baseline_rmse = subset['Baseline RMSE'].mean()
        adaptive_rmse = subset['Adaptive RMSE'].mean()
        baseline_corr = subset['Baseline Corr'].mean()
        adaptive_corr = subset['Adaptive Corr'].mean()
        
        print(f"  SNR - Baseline: {baseline_snr:.3f} ± {subset['Baseline SNR'].std():.3f}")
        print(f"  SNR - Adaptive: {adaptive_snr:.3f} ± {subset['Adaptive SNR'].std():.3f}")
        print(f"  SNR Improvement: {adaptive_snr - baseline_snr:+.3f}")
        print()
        print(f"  RMSE - Baseline: {baseline_rmse:.3f} ± {subset['Baseline RMSE'].std():.3f}")
        print(f"  RMSE - Adaptive: {adaptive_rmse:.3f} ± {subset['Adaptive RMSE'].std():.3f}")
        print()
        print(f"  CORR - Baseline: {baseline_corr:.3f} ± {subset['Baseline Corr'].std():.3f}")
        print(f"  CORR - Adaptive: {adaptive_corr:.3f} ± {subset['Adaptive Corr'].std():.3f}")
    
    print("\n" + "=" * 60)
    print("✅ Processing complete!")
    print("=" * 60)