"""
Analyze results for 49 healthy subjects
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
df = pd.read_csv("healthy_49_subjects_results.csv")

print("=" * 60)
print("📊 Statistical Analysis for 49 Healthy Subjects")
print("=" * 60)

# 1. Overall statistics
print("\n📈 Overall Statistics:")
for wavelet in ['db4', 'dmey']:
    print(f"\n{wavelet.upper()}:")
    subset = df[df.wavelet == wavelet]
    print(f"  SNR Improvement: {subset['Adaptive SNR'].mean() - subset['Baseline SNR'].mean():+.3f}")
    print(f"  % Improved: {(subset['Adaptive SNR'] > subset['Baseline SNR']).mean()*100:.1f}%")

# 2. Paired t-test
from scipy import stats
print("\n📊 Paired t-test (Adaptive vs Baseline):")
for wavelet in ['db4', 'dmey']:
    subset = df[df.wavelet == wavelet]
    t_stat, p_value = stats.ttest_rel(subset['Adaptive SNR'], subset['Baseline SNR'])
    print(f"\n{wavelet}: t={t_stat:.3f}, p={p_value:.4f} {'✅' if p_value<0.05 else '❌'}")

# 3. Plot distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, wavelet in enumerate(['db4', 'dmey']):
    subset = df[df.wavelet == wavelet]
    improvement = subset['Adaptive SNR'] - subset['Baseline SNR']
    
    axes[idx].hist(improvement, bins=15, alpha=0.7, color='green' if idx==0 else 'orange')
    axes[idx].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[idx].set_xlabel('SNR Improvement (dB)')
    axes[idx].set_ylabel('Number of Subjects')
    axes[idx].set_title(f'{wavelet}: Improvement Distribution (n=49)')
    axes[idx].grid(True, alpha=0.3)
    
    # Add statistics
    mean_imp = improvement.mean()
    std_imp = improvement.std()
    axes[idx].text(0.05, 0.95, f'Mean: {mean_imp:.3f}±{std_imp:.3f}',
                   transform=axes[idx].transAxes, fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig("healthy_49_improvement.png", dpi=150)
plt.show()

print("\n✅ Analysis complete!")