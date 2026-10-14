"""
Signal preprocessing: bandpass/notch filtering and simple artifact rejection,
applied per-epoch with SciPy.
"""
from scipy.signal import butter, filtfilt, iirnotch

import config


def _butter_bandpass(low, high, fs, order):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return b, a


def bandpass_filter(data, low=config.BANDPASS_LOW, high=config.BANDPASS_HIGH,
                     fs=config.SFREQ, order=config.FILTER_ORDER):
    """Zero-phase Butterworth bandpass filter. data shape: (..., n_samples)."""
    b, a = _butter_bandpass(low, high, fs, order)
    return filtfilt(b, a, data, axis=-1)


def notch_filter(data, freq=config.NOTCH_FREQ, fs=config.SFREQ, quality=30):
    """Remove power-line interference at `freq` Hz."""
    b, a = iirnotch(freq / (fs / 2), quality)
    return filtfilt(b, a, data, axis=-1)


def preprocess(X, fs=config.SFREQ, apply_notch=True):
    """Run the filtering chain on a batch of epochs (n_epochs, n_channels, n_samples)."""
    X = bandpass_filter(X, fs=fs)
    if apply_notch:
        X = notch_filter(X, fs=fs)
    return X


def reject_bad_epochs(X, y, threshold=config.ARTIFACT_PEAK_TO_PEAK):
    """
    Drop epochs whose peak-to-peak amplitude on any channel exceeds `threshold`.
    A simple, explainable stand-in for a full artifact-rejection pipeline (e.g. ICA).
    """
    ptp = X.max(axis=-1) - X.min(axis=-1)  # (n_epochs, n_channels)
    keep = (ptp <= threshold).all(axis=-1)
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(f"  artifact rejection: dropped {n_dropped}/{len(y)} epochs")
    return X[keep], y[keep]
