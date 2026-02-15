from metrics import compute_snr, compute_rmse, compute_corr

def evaluate(clean, noisy, baseline, adaptive):
    # Initialize an empty dictionary to store all evaluation results
    results = {}

    # Calculate Signal-to-Noise Ratio (SNR) for each signal type
    results["Noisy SNR"] = compute_snr(clean, noisy)
    results["Baseline SNR"] = compute_snr(clean, baseline)
    results["Adaptive SNR"] = compute_snr(clean, adaptive)

    # Calculate Root Mean Square Error (RMSE) for denoised signals
    results["Baseline RMSE"] = compute_rmse(clean, baseline)
    results["Adaptive RMSE"] = compute_rmse(clean, adaptive)

    # Calculate Pearson correlation coefficient for denoised signals
    results["Baseline Corr"] = compute_corr(clean, baseline)
    results["Adaptive Corr"] = compute_corr(clean, adaptive)

    # Return the dictionary containing all computed metrics
    return results