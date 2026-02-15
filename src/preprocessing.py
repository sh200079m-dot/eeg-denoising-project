import matplotlib
# Set the matplotlib backend to TkAgg for interactive plotting (must be done before importing pyplot)
matplotlib.use('TkAgg')

import mne
import numpy as np
from load_data import load_eeg
import matplotlib.pyplot as plt


def basic_preprocessing(raw):
    # Select only EEG channels (exclude stimulus, EOG, etc.)
    raw.pick(picks="eeg")
    # Create a standard 10-20 electrode montage for channel positions
    montage = mne.channels.make_standard_montage("standard_1020")
    # Apply the montage to the raw data (case-insensitive matching)
    raw.set_montage(montage, match_case=False)
    # Optional bandpass filter (currently commented out)
    #raw.filter(l_freq=0.5, h_freq=45)
    return raw


if __name__ == "__main__":
    # Load EEG data for subject "sub-001"
    raw = load_eeg("sub-001")
    # Apply basic preprocessing steps
    raw_clean = basic_preprocessing(raw)
    
    # Print information about the preprocessed data
    print(raw_clean)
    
    # Plot the EEG signal (first 10 channels, 5 seconds duration) and save the figure
    fig1 = raw_clean.plot(n_channels=10, duration=5, scalings="auto")
    fig1.savefig("eeg_signal_preprocessed.png", dpi=150)
    plt.close(fig1)
    print("Saved: eeg_signal_preprocessed.png")
    
    # Compute and plot the Power Spectral Density (PSD) up to 60 Hz
    psd = raw_clean.compute_psd(fmax=60)
    fig2 = psd.plot()
    fig2.savefig("eeg_psd_preprocessed.png", dpi=150)
    plt.close(fig2)
    print("Saved: eeg_psd_preprocessed.png")
    
    # Commented out interactive display
    # plt.show(block=True)
    
    print("Done! Plots saved as PNG files.")