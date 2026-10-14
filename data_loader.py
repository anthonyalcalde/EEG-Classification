"""
Data loading for the PhysioNet EEG Motor Movement/Imagery Dataset.

Real data is downloaded automatically via MNE-Python's built-in PhysioNet
fetcher (mne.datasets.eegbci). MNE is used here purely for file I/O --
reading the .edf recordings and their annotation events. Everything
downstream (filtering, feature extraction) is done by hand with SciPy/NumPy
so that part of the pipeline stays fully transparent and controllable.
"""
import numpy as np

import config


def fetch_and_load_raw(subject, runs):
    """
    Download (if not already cached) and load PhysioNet EEGMMI data for one subject.

    Parameters
    ----------
    subject : int
        Subject number, 1-109.
    runs : list[int]
        Run numbers to concatenate (see config.py for the run groupings).

    Returns
    -------
    mne.io.Raw
        Concatenated, channel-standardized raw recording.
    """
    import mne
    from mne.datasets import eegbci

    paths = eegbci.load_data(subject, runs, verbose=False)
    raws = [mne.io.read_raw_edf(p, preload=True, verbose=False) for p in paths]
    raw = mne.concatenate_raws(raws)

    # Standardize channel names/positions to the 10-20 system
    eegbci.standardize(raw)
    raw.set_montage("standard_1005", on_missing="ignore", verbose=False)
    return raw


def raw_to_epochs(raw, event_map, tmin=config.EPOCH_TMIN, tmax=config.EPOCH_TMAX):
    """
    Slice a continuous Raw recording into labeled epochs around annotation events.

    Parameters
    ----------
    raw : mne.io.Raw
    event_map : dict
        Maps annotation description (e.g. "T1", "T2") -> integer class label.
    tmin, tmax : float
        Epoch window relative to event onset, in seconds.

    Returns
    -------
    X : ndarray, shape (n_epochs, n_channels, n_samples)
    y : ndarray, shape (n_epochs,)
    ch_names : list[str]
    sfreq : float
    """
    import mne

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    keep_id = {k: v for k, v in event_id.items() if k in event_map}
    if not keep_id:
        raise ValueError(f"None of {list(event_map)} found in annotations {list(event_id)}")

    epochs = mne.Epochs(
        raw, events, event_id=keep_id, tmin=tmin, tmax=tmax,
        baseline=None, preload=True, verbose=False,
    )
    X = epochs.get_data()  # (n_epochs, n_channels, n_samples), in Volts
    raw_codes = epochs.events[:, 2]
    code_to_label = {v: event_map[k] for k, v in keep_id.items()}
    y = np.array([code_to_label[c] for c in raw_codes])
    return X, y, epochs.ch_names, raw.info["sfreq"]


def load_dataset(subjects, runs, event_map):
    """Fetch, load and concatenate epochs across multiple subjects."""
    X_all, y_all = [], []
    ch_names, sfreq = None, None
    for subj in subjects:
        if subj in config.BAD_SUBJECTS:
            print(f"  skipping subject {subj}: known bad annotations")
            continue
        try:
            raw = fetch_and_load_raw(subj, runs)
            X, y, ch_names, sfreq = raw_to_epochs(raw, event_map)
        except Exception as e:
            print(f"  skipping subject {subj}: {e}")
            continue
        X_all.append(X)
        y_all.append(y)
        print(f"  subject {subj}: {len(y)} epochs loaded")

    if not X_all:
        raise RuntimeError("No usable data was loaded for any requested subject.")

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    return X, y, ch_names, sfreq


def generate_synthetic_dataset(n_epochs_per_class=40, n_channels=8, sfreq=160,
                                duration=4.0, random_state=config.RANDOM_STATE):
    """
    Build a small synthetic two-class dataset shaped like real PhysioNet
    motor-imagery epochs, for offline pipeline testing / demos with no
    network access. Class 0 carries extra mu/beta-band (9-12 Hz) power on
    the first half of channels, class 1 on the second half -- a rough stand
    -in for the contralateral mu-rhythm patterns seen in left vs right hand
    imagery, just enough signal for the classifiers to learn something real.
    """
    rng = np.random.default_rng(random_state)
    n_samples = int(duration * sfreq)
    t = np.arange(n_samples) / sfreq
    X, y = [], []

    def make_epoch(strong_channels):
        epoch = rng.normal(0, 1e-5, size=(n_channels, n_samples))
        for ch in range(n_channels):
            if ch in strong_channels:
                freq, amp = rng.uniform(9, 12), 3e-5
            else:
                freq, amp = rng.uniform(1, 4), 0.5e-5
            phase = rng.uniform(0, 2 * np.pi)
            epoch[ch] += amp * np.sin(2 * np.pi * freq * t + phase)
        return epoch

    half = n_channels // 2
    for _ in range(n_epochs_per_class):
        X.append(make_epoch(strong_channels=set(range(0, half))))
        y.append(0)
        X.append(make_epoch(strong_channels=set(range(half, n_channels))))
        y.append(1)

    X = np.array(X)
    y = np.array(y)
    ch_names = [f"CH{i + 1}" for i in range(n_channels)]
    return X, y, ch_names, float(sfreq)
