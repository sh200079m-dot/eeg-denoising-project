import numpy as np


def rigrsure(coeffs):
    """
    coeffs: 1D numpy array (detail coefficients)
    Compute threshold using Stein's Unbiased Risk Estimate (SURE) method
    """
    # Convert input to numpy array if it isn't already
    coeffs = np.asarray(coeffs)
    # Get the number of coefficients
    n = coeffs.size
    # Return zero threshold if the array is empty
    if n == 0:
        return 0

    # Sort the squared coefficients in ascending order
    sorted_coeffs = np.sort(coeffs ** 2)
    # Calculate the risk for each potential threshold
    # Formula: (n - 2 * (1..n) + cumulative sum of sorted squared coefficients) / n
    risks = (n - 2 * np.arange(1, n+1) + np.cumsum(sorted_coeffs)) / n
    # Find the index of the minimum risk
    idx = np.argmin(risks)
    # The threshold is the square root of the coefficient at that index
    threshold = np.sqrt(sorted_coeffs[idx])
    return threshold


def hard_threshold(c, T):
    """
    Apply hard thresholding: keep coefficients with absolute value >= T, set others to zero
    """
    return c * (np.abs(c) >= T)


def soft_threshold(c, T):
    """
    Apply soft thresholding: shrink coefficients towards zero by T
    """
    return np.sign(c) * np.maximum(np.abs(c) - T, 0)