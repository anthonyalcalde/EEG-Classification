"""
Configuration for the EEG Motor Imagery Classification pipeline.

Dataset: PhysioNet EEG Motor Movement/Imagery Dataset (eegmmidb)
https://physionet.org/content/eegmmidb/1.0.0/
"""

# --- Dataset structure ----------------------------------------------------
SFREQ = 160  # Hz, native sampling rate of the PhysioNet EEGMMI dataset

# Run numbers, per the PhysioNet documentation for this dataset
RUNS_BASELINE_EYES_OPEN = [1]
RUNS_BASELINE_EYES_CLOSED = [2]
RUNS_EXEC_LR_HAND = [3, 7, 11]           # real movement: left vs right fist
RUNS_IMAGINED_LR_HAND = [4, 8, 12]       # imagined movement: left vs right fist
RUNS_EXEC_HANDS_FEET = [5, 9, 13]        # real movement: both fists vs both feet
RUNS_IMAGINED_HANDS_FEET = [6, 10, 14]   # imagined: both fists vs both feet

# Subjects with documented annotation/timing issues in the raw dataset
BAD_SUBJECTS = {88, 89, 92, 100, 104}

# --- Preprocessing ----------------------------------------------------
BANDPASS_LOW = 1.0      # Hz
BANDPASS_HIGH = 40.0    # Hz
NOTCH_FREQ = 60.0       # Hz, US power-line frequency (use 50.0 outside US/NA)
FILTER_ORDER = 4

# Epoch window relative to event onset, in seconds
EPOCH_TMIN = 0.0
EPOCH_TMAX = 4.0

# Peak-to-peak amplitude rejection threshold, in Volts (MNE reads EDF in volts)
ARTIFACT_PEAK_TO_PEAK = 150e-6

# --- Feature extraction ----------------------------------------------------
FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40),
}

# --- Misc ----------------------------------------------------
OUTPUT_DIR = "output"
RANDOM_STATE = 42
