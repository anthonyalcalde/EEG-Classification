"""
Frequency-domain feature extraction via Welch's method.
"""
import numpy as np
import pandas as pd
from scipy.signal import welch

import config


def band_power(epoch, fs, bands=config.FREQ_BANDS, nperseg=None):
    """
    Compute average power spectral density within each frequency band,
    per channel, for one epoch of shape (n_channels, n_samples).

    Returns
    -------
    dict[str, ndarray] mapping band name -> (n_channels,) power array
    """
    nperseg = nperseg or min(256, epoch.shape[-1])
    freqs, psd = welch(epoch, fs=fs, nperseg=nperseg, axis=-1)  # psd: (n_channels, n_freqs)
    powers = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs <= hi)
        powers[name] = psd[:, mask].mean(axis=-1) if mask.any() else np.zeros(psd.shape[0])
    return powers


def extract_features(X, fs, ch_names, bands=config.FREQ_BANDS):
    """
    Turn a batch of epochs into a tabular feature matrix: one row per epoch,
    one column per (channel, band) pair, e.g. "C3_alpha".

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_samples)

    Returns
    -------
    pandas.DataFrame, shape (n_epochs, n_channels * n_bands)
    """
    rows = []
    for epoch in X:
        powers = band_power(epoch, fs, bands)
        row = {}
        for band_name, values in powers.items():
            for ch, v in zip(ch_names, values):
                row[f"{ch}_{band_name}"] = v
        rows.append(row)
    return pd.DataFrame(rows)
