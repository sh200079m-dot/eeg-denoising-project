"""
Plot EEG with correct scale (microvolts)
Shows clean vs noisy signals with statistics
"""

import matplotlib.pyplot as plt
import numpy as np
from load_data import load_eeg
from preprocessing import basic_preprocessing
from segment import extract_segment
from add_noise import add_line_noise, add_baseline_wander

# ============================
# Configuration
# ============================
subjects = ["sub-001", "sub-101"]
snr_db = 10
baseline_drift_ratio = 0.05
channel_to_plot = 0  # First channel

# ============================
# Main plotting function
# ============================
def plot_correct_scale(subject):
    print(f"\nPlotting correct scale for {subject}...")
    
    # Load and preprocess
    raw = load_eeg(subject)
    raw_clean = basic_preprocessing(raw)
    segment, sfreq = extract_segment(raw_clean)
    
    # Convert to microvolts
    clean = segment * 1e6
    
    # Add noise
    noisy = add_line_noise(clean, sfreq, snr_db=snr_db)
    noisy = add_baseline_wander(noisy, sfreq, amplitude_ratio=baseline_drift_ratio)
    
    # Select 1 second for plotting
    samples = slice(0, int(1 * sfreq))
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(clean[channel_to_plot, samples], 
             label="Clean EEG", linewidth=2, color='blue')
    plt.plot(noisy[channel_to_plot, samples], 
             label="Noisy EEG", linewidth=1.5, color='red', alpha=0.7)
    
    plt.xlabel("Samples")
    plt.ylabel("Amplitude (μV)")
    plt.title(f"EEG Channel 1 - 1 Second - {subject}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Statistics box
    stats_text = f"""Mean: {clean[channel_to_plot, samples].mean():.1f} μV
Std: {clean[channel_to_plot, samples].std():.1f} μV
Min: {clean[channel_to_plot, samples].min():.1f} μV
Max: {clean[channel_to_plot, samples].max():.1f} μV"""
    
    plt.text(0.02, 0.98, stats_text, 
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    filename = f"eeg_correct_scale_{subject}.png"
    plt.savefig(filename, dpi=150)
    print(f"Saved: {filename}")
    plt.close()

# ============================
# Run for both subjects
# ============================
if __name__ == "__main__":
    for subject in subjects:
        plot_correct_scale(subject)
    print("\nDone! Created:")
    print("  - eeg_correct_scale_sub-001.png")
    print("  - eeg_correct_scale_sub-101.png")