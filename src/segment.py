print("segment.py is running")

import numpy as np
from load_data import load_eeg
from preprocessing import basic_preprocessing


def extract_segment(raw, duration_sec=10):
    """
    Extract a middle segment of EEG
    """
    # Get the sampling frequency from the raw data info
    sfreq = raw.info["sfreq"]
    # Get the total number of time samples in the raw data
    total_samples = raw.n_times

    # Calculate the number of samples needed for the desired duration
    segment_samples = int(duration_sec * sfreq)

    # Calculate start and stop indices to extract a segment from the middle of the recording
    start = total_samples // 2 - segment_samples // 2
    stop = start + segment_samples

    # Extract the data segment (all channels, specified time range)
    data = raw.get_data()[:, start:stop]

    return data, sfreq


if __name__ == "__main__":
    # Load EEG data for subject "sub-001"
    raw = load_eeg("sub-001")
    # Apply basic preprocessing
    raw_clean = basic_preprocessing(raw)

    # Extract a 10-second middle segment from the preprocessed data
    segment, sfreq = extract_segment(raw_clean)

    # Print information about the extracted segment
    print("Segment shape:", segment.shape)
    print("Sampling freq:", sfreq)