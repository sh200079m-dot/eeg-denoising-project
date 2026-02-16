"""
EEG Denoising Comparison Plotter
--------------------------------
Generates comprehensive comparison plots for:
- Both subjects (sub-001, sub-101)
- Both wavelets (db4, dmey)
- All methods (Noisy, Baseline, Adaptive, Clean)

Saves individual PNG files for each combination
"""

import matplotlib.pyplot as plt
import numpy as np
from load_data import load_eeg
from preprocessing import basic_preprocessing
from segment import extract_segment
from add_noise import add_line_noise, add_baseline_wander
from baseline_denoise import baseline_wavelet_denoise
from adaptive_denoise import adaptive_wavelet_denoise


# ============================
# Configuration
# ============================
subjects = ["sub-001", "sub-101"]
wavelets = ["db4", "dmey"]
snr_db = 10
baseline_drift_ratio = 0.05
duration_plot = 2  # 2 seconds for visualization (still plot 2 seconds)
channel_to_plot = 0  # First channel


# ============================
# Main Plotting Function
# ============================
def plot_comparison(subject, wavelet):
    """Generate and save comparison plot for given subject and wavelet"""
    
    print(f"\nPlotting: {subject} - Wavelet: {wavelet}")
    
    # ----------------------------
    # Load & Preprocess
    # ----------------------------
    raw = load_eeg(subject)
    raw = basic_preprocessing(raw)
    segment, sfreq = extract_segment(raw)
    clean = segment * 1e6  # Convert to microvolts
    
    # ----------------------------
    # Add Noise
    # ----------------------------
    noisy = add_line_noise(clean, sfreq, snr_db=snr_db)
    noisy = add_baseline_wander(noisy, sfreq, amplitude_ratio=baseline_drift_ratio)
    
    # ----------------------------
    # Apply Denoising Methods
    # ----------------------------
    baseline = baseline_wavelet_denoise(noisy, wavelet=wavelet)
    adaptive = adaptive_wavelet_denoise(noisy, wavelet=wavelet)
    
    # ----------------------------
    # Calculate SNR on FULL signal (all channels, all 10 seconds)
    # ----------------------------
    def calc_snr_full(clean_sig, denoised_sig):
        noise = clean_sig - denoised_sig
        snr = 10 * np.log10(np.mean(clean_sig**2) / np.mean(noise**2))
        return snr
    
    snr_noisy_full = calc_snr_full(clean, noisy)
    snr_baseline_full = calc_snr_full(clean, baseline)
    snr_adaptive_full = calc_snr_full(clean, adaptive)
    
    print(f"  Full SNR - Noisy: {snr_noisy_full:.3f} dB")
    print(f"  Full SNR - Baseline: {snr_baseline_full:.3f} dB")
    print(f"  Full SNR - Adaptive: {snr_adaptive_full:.3f} dB")
    
    # ----------------------------
    # Prepare Plot Data (only first 2 seconds for visualization)
    # ----------------------------
    samples = slice(0, int(duration_plot * sfreq))
    time = np.arange(samples.stop) / sfreq
    
    # ----------------------------
    # Create Plot
    # ----------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top plot: All signals together
    ax1 = axes[0]
    ax1.plot(time, noisy[channel_to_plot, samples], 
             label=f'Noisy (Full SNR: {snr_noisy_full:.2f} dB)', alpha=0.5, color='red', linewidth=1)
    ax1.plot(time, baseline[channel_to_plot, samples], 
             label=f'Baseline Rigrsure (Full SNR: {snr_baseline_full:.2f} dB)', color='blue', linewidth=1.5)
    ax1.plot(time, adaptive[channel_to_plot, samples], 
             label=f'Proposed Adaptive (Full SNR: {snr_adaptive_full:.2f} dB)', color='green', linewidth=2)
    ax1.plot(time, clean[channel_to_plot, samples], 
             label='Clean Reference', linestyle='--', color='black', linewidth=1)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude (μV)')
    ax1.set_title(f'EEG Denoising Comparison - {subject} (Wavelet: {wavelet}) - Channel 1 (First {duration_plot}s)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Separate vertical offset for clarity
    ax2 = axes[1]
    offset = 0
    signals = [
        (noisy[channel_to_plot, samples], 'Noisy', 'red', 1, offset),
        (baseline[channel_to_plot, samples], 'Baseline', 'blue', 1.5, offset + 40),
        (adaptive[channel_to_plot, samples], 'Adaptive', 'green', 2, offset + 80),
        (clean[channel_to_plot, samples], 'Clean', 'black', 1, offset + 120)
    ]
    
    for sig, label, color, lw, off in signals:
        ax2.plot(time, sig + off, label=label, color=color, linewidth=lw)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude (μV) + offset')
    ax2.set_title('Signals with vertical offset for better comparison')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_yticks([])
    
    plt.tight_layout()
    
    # ----------------------------
    # Save Plot
    # ----------------------------
    filename = f"denoising_comparison_{subject}_{wavelet}_fullsnr.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close(fig)


# ============================
# Create Summary Bar Chart
# ============================
def create_summary_chart():
    """Create bar chart comparing SNR improvements"""
    
    # Data from your test_denoising.py output
    data = {
        ('sub-001', 'db4', 'Baseline'): 9.999,
        ('sub-001', 'db4', 'Adaptive'): 10.082,
        ('sub-001', 'dmey', 'Baseline'): 9.904,
        ('sub-001', 'dmey', 'Adaptive'): 9.899,
        ('sub-101', 'db4', 'Baseline'): 10.000,
        ('sub-101', 'db4', 'Adaptive'): 10.283,
        ('sub-101', 'dmey', 'Baseline'): 9.904,
        ('sub-101', 'dmey', 'Adaptive'): 9.964,
    }
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Define positions
    x_pos = np.arange(4)  # 4 combinations (2 subjects × 2 wavelets)
    width = 0.35
    
    # Prepare data arrays
    baseline_values = [
        data[('sub-001', 'db4', 'Baseline')],
        data[('sub-001', 'dmey', 'Baseline')],
        data[('sub-101', 'db4', 'Baseline')],
        data[('sub-101', 'dmey', 'Baseline')]
    ]
    
    adaptive_values = [
        data[('sub-001', 'db4', 'Adaptive')],
        data[('sub-001', 'dmey', 'Adaptive')],
        data[('sub-101', 'db4', 'Adaptive')],
        data[('sub-101', 'dmey', 'Adaptive')]
    ]
    
    # Plot bars
    bars_baseline = ax.bar(x_pos - width/2, baseline_values, width, 
                           label='Baseline (Rigrsure)', color='steelblue', edgecolor='navy', alpha=0.8)
    bars_adaptive = ax.bar(x_pos + width/2, adaptive_values, width,
                          label='Proposed Adaptive', color='coral', edgecolor='darkred', alpha=0.8)
    
    # Customize chart
    ax.set_xlabel('Subject - Wavelet Combination', fontsize=12)
    ax.set_ylabel('SNR (dB)', fontsize=12)
    ax.set_title('SNR Comparison: Baseline vs Adaptive Method', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['sub-001\ndb4', 'sub-001\ndmey', 'sub-101\ndb4', 'sub-101\ndmey'])
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(9.5, 10.5)  # Set appropriate y-axis limits
    
    # Add value labels on bars
    for bars in [bars_baseline, bars_adaptive]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add improvement text
    for i in range(4):
        improvement = adaptive_values[i] - baseline_values[i]
        color = 'green' if improvement > 0 else 'red'
        ax.annotate(f'{improvement:+.3f}',
                   xy=(x_pos[i], max(baseline_values[i], adaptive_values[i]) + 0.05),
                   ha='center', va='bottom', fontsize=10, 
                   color=color, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.2))
    
    plt.tight_layout()
    plt.savefig("snr_comparison_summary.png", dpi=150, bbox_inches='tight')
    print("Saved: snr_comparison_summary.png")
    plt.close(fig)


# ============================
# Main Execution
# ============================
if __name__ == "__main__":
    print("=" * 60)
    print("Generating EEG Denoising Comparison Plots")
    print("=" * 60)
    
    # Generate individual plots for each combination
    for subject in subjects:
        for wavelet in wavelets:
            plot_comparison(subject, wavelet)
    
    # Generate summary bar chart
    create_summary_chart()
    
    print("\n" + "=" * 60)
    print("All plots generated successfully!")
    print("Files created:")
    for subject in subjects:
        for wavelet in wavelets:
            print(f"  - denoising_comparison_{subject}_{wavelet}_fullsnr.png")
    print("  - snr_comparison_summary.png")
    print("=" * 60)