# EEG Motor Imagery Classification

Classifies EEG signal segments from the [PhysioNet EEG Motor Movement/Imagery
Dataset](https://physionet.org/content/eegmmidb/1.0.0/) — 109 subjects, 64-channel
EEG at 160 Hz, performing and imagining left/right hand and both-hands/feet
movements. The pipeline filters the raw signal, extracts frequency-band power
features, and trains/compares several scikit-learn classifiers.

## How it works

1. **`data_loader.py`** — downloads (once, then caches) and reads the raw `.edf`
   recordings via MNE, and slices them into labeled epochs around each
   movement/imagery event.
2. **`preprocessing.py`** — 1–40 Hz Butterworth bandpass filter, 60 Hz notch
   filter for power-line noise, and peak-to-peak amplitude artifact rejection.
   All hand-written with `scipy.signal`.
3. **`feature_extraction.py`** — Welch's method PSD per channel, averaged into
   delta/theta/alpha/beta/gamma band power. One row per epoch, one column per
   channel × band.
4. **`models.py`** — trains LDA, Logistic Regression, SVM (RBF), Random Forest,
   and k-NN on the standardized features.
5. **`evaluate.py`** — accuracy, precision, recall, F1 per model, a comparison
   bar chart, and a confusion matrix for the best model.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Try it instantly with synthetic data** (no download, runs in seconds — useful
for confirming your environment works before waiting on real downloads):

```bash
python main.py --demo
```

**Run on real PhysioNet data:**

```bash
python main.py --subjects 1 2 3 4 5 --task lr_imagined
```

The first run downloads the needed `.edf` files into `~/mne_data` (a few MB
per subject/run) and reuses that cache afterward. `--subjects` accepts any
numbers 1–109 (a few subjects with known annotation issues are auto-skipped —
see `BAD_SUBJECTS` in `config.py`).

`--task` options:

| Task           | Classes                    | Real vs imagined |
|----------------|-----------------------------|-------------------|
| `lr_imagined`  | left hand vs right hand     | imagined (default)|
| `lr_executed`  | left hand vs right hand     | actually performed|
| `hf_imagined`  | both fists vs both feet     | imagined           |
| `hf_executed`  | both fists vs both feet     | actually performed|

Results (`model_comparison.csv`, `model_comparison.png`, `confusion_matrix.png`)
are written to `output/` (change with `--out`).

## Notes

- **Demo mode is a smoke test, not a benchmark.** The synthetic signals are
  built to be easily separable, so you'll likely see near-perfect scores —
  that just confirms the pipeline runs correctly end to end. Real EEG is
  much noisier; expect meaningfully lower (and more interesting) numbers,
  especially on `lr_imagined`, which is a genuinely hard classification task
  even in published literature (~60–80% accuracy per subject is typical).
- Per-subject variability is large in this dataset — a model trained across
  many subjects at once ("subject-independent") will generally score lower
  than one trained and tested within a single subject. Start with a handful
  of subjects and a single task to sanity-check things, then scale up.
- To try more subjects, just extend `--subjects` — the loader streams and
  concatenates them, skipping any that fail to load or download rather than
  aborting the whole run.

## Extending

- Add more features, e.g. band power ratios, or Hjorth parameters
- Try common spatial patterns (CSP) via `mne.decoding.CSP`, the standard
  spatial-filtering approach for motor imagery
- Add k-fold cross-validation instead of a single train/test split
- Add subject-independent vs subject-dependent evaluation modes
