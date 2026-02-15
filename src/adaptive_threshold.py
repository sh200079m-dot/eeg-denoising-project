import numpy as np

def adaptive_threshold(coeffs, channel_energy, level):
    """
    Mild adaptive threshold for EEG denoising
    """

    # Robust noise estimate using median absolute deviation
    sigma = np.median(np.abs(coeffs)) / 0.6745

    #Mild effect of decomposition level (previously was 0.2)
    alpha = 1 + 0.05 * level

    #Normalize channel energy
    beta = channel_energy / np.median(channel_energy)

    #Overall scaling factor to prevent over-smoothing (very important)
    gamma = 0.4

    # Calculate the adaptive threshold
    T = gamma * alpha * beta * sigma
    return T