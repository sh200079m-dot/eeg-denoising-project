# EEG Denoising with Adaptive Wavelet Thresholding

## 📌 Project Overview
This project implements an **adaptive channel-wise thresholding method** for EEG signal denoising using wavelet transform. The proposed method is compared with the classical **Rigrsure (SURE) thresholding** approach on two subjects from the "Rest eyes open – Parkinson's Disease" dataset.


## 📊 Dataset
**Source:** "Rest eyes open – Parkinson's Disease 64-Channel" from OpenNeuro  
🔗 [Dataset Link]((https://openneuro.org/datasets/ds004584/versions/1.0.0)  )
**Subjects used:** sub-001 (Parkinson patient), sub-101 (Healthy control)  
**Specifications:** 63 channels, 500 Hz sampling rate

## Repository Structure
- `src/`: Python source codes
- `notebooks/`: Jupyter exploration notebook
- `*.png`: Result figures(For reference, the original comparison plots (without _fullsnr in their names) show results for Channel 1 only and were used during initial development, while the _fullsnr versions represent the final results with global SNR calculation.)
- 

## 👩‍💻 Author
Shaghayegh Masoudian 
course: Advanced Digital Signal Processing (ADSP)
## 📁Project Structure
EEG_Project/
├── data/                    # EEG dataset (not included in repo)
│   ├── sub-001/
│   │   └── eeg/
│   └── sub-101/
│       └── eeg/
├── notebooks/               # Jupyter notebooks
│   └── explore_data.ipynb
├── src/                     # Python source codes
│   ├── load_data.py         # EEG data loading
│   ├── preprocessing.py     # Basic preprocessing
│   ├── segment.py           # Segment extraction
│   ├── add_noise.py         # Artificial noise addition
│   ├── wavelet_utils.py     # Wavelet decomposition/reconstruction
│   ├── threshold_rules.py   # Rigrsure and thresholding functions
│   ├── baseline_denoise.py  # Baseline method
│   ├── adaptive_denoise.py  # Proposed adaptive method
│   ├── adaptive_threshold.py # Core adaptive threshold formula
│   ├── metrics.py           # Evaluation metrics (SNR, RMSE, correlation)
│   ├── evaluate_methods.py  # Comprehensive evaluation
│   ├── test_denoising.py    # Main testing script
│   ├── test_noise.py        # Noise simulation test
│   └── plot_denoising_comparison.py # Visualization
├── *.png                     # Result figures
├── .gitignore
└── README.md

# EEG Denoising with Adaptive Wavelet Thresholding

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![MNE](https://img.shields.io/badge/MNE-1.0%2B-orange)](https://mne.tools/)
[![PyWavelets](https://img.shields.io/badge/PyWavelets-1.1%2B-green)](https://pywavelets.readthedocs.io/)
**Key features:**
- Adaptive threshold calculation based on each channel's energy
- Comparison of two wavelets: `db4` and `dmey`
- Evaluation metrics: SNR, RMSE, and Pearson correlation coefficient
- Visualization tools for result comparison
