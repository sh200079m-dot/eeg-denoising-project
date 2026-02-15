import os
import mne

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")


def load_eeg(subject_id):
    """
    Load EEG data for a given subject in EEGLAB (.set) format
    """

    # Construct the subject directory path using the provided subject ID
    subject_dir = os.path.join(DATA_DIR, subject_id)

    # Define possible subdirectories where the EEG file might be located
    possible_dirs = [
        os.path.join(subject_dir, "eeg"),
        subject_dir
    ]

    eeg_file = None

    # Search through possible directories for a .set file
    for d in possible_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith(".set"):
                    eeg_file = os.path.join(d, f)
                    break

    # Raise an error if no .set file is found
    if eeg_file is None:
        raise FileNotFoundError(f"No .set EEG file found for {subject_id}")

    # Print the path of the file being loaded
    print(f"Loading EEG file: {eeg_file}")

    # Load the EEG file using MNE's EEGLAB reader with preloading enabled
    raw = mne.io.read_raw_eeglab(eeg_file, preload=True)
    return raw


if __name__ == "__main__":
    # Test the function with subject "sub-001"
    raw_test = load_eeg("sub-001")
    # Print the loaded raw object information
    print(raw_test)
    # Plot the first 10 channels for 5 seconds with automatic scaling
    raw_test.plot(n_channels=10, duration=5, scalings="auto")