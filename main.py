"""
End-to-end EEG motor-imagery classification pipeline.

Usage
-----
Quick offline test with synthetic data (no download, runs in seconds):
    python main.py --demo

Real PhysioNet data, e.g. left vs right hand imagery for 5 subjects:
    python main.py --subjects 1 2 3 4 5 --task lr_imagined

The first time you run against real data, MNE will download the requested
.edf files from PhysioNet into ~/mne_data (a few MB per subject/run); later
runs reuse the cache.
"""
import argparse
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config
import data_loader
import evaluate
import feature_extraction
import models
import preprocessing

TASKS = {
    "lr_imagined": (config.RUNS_IMAGINED_LR_HAND, {"T1": 0, "T2": 1}, ["Left hand", "Right hand"]),
    "lr_executed": (config.RUNS_EXEC_LR_HAND, {"T1": 0, "T2": 1}, ["Left hand", "Right hand"]),
    "hf_imagined": (config.RUNS_IMAGINED_HANDS_FEET, {"T1": 0, "T2": 1}, ["Both fists", "Both feet"]),
    "hf_executed": (config.RUNS_EXEC_HANDS_FEET, {"T1": 0, "T2": 1}, ["Both fists", "Both feet"]),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", action="store_true", help="use synthetic data, skip PhysioNet download")
    p.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 6)),
                    help="PhysioNet subject numbers, 1-109")
    p.add_argument("--task", choices=TASKS, default="lr_imagined")
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--out", default=config.OUTPUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.demo:
        print("Demo mode: generating synthetic motor-imagery-like epochs...")
        X, y, ch_names, sfreq = data_loader.generate_synthetic_dataset()
        class_names = ["Class 0", "Class 1"]
    else:
        runs, event_map, class_names = TASKS[args.task]
        print(f"Loading PhysioNet data for subjects {args.subjects}, task={args.task} ...")
        X, y, ch_names, sfreq = data_loader.load_dataset(args.subjects, runs, event_map)

    print(f"Loaded {X.shape[0]} epochs, {X.shape[1]} channels, {X.shape[2]} samples/epoch @ {sfreq} Hz")

    print("Filtering (bandpass + notch)...")
    X = preprocessing.preprocess(X, fs=sfreq)
    X, y = preprocessing.reject_bad_epochs(X, y)

    print("Extracting band-power features...")
    features_df = feature_extraction.extract_features(X, sfreq, ch_names)
    print(f"Feature matrix: {features_df.shape[0]} epochs x {features_df.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        features_df.values, y, test_size=args.test_size,
        random_state=config.RANDOM_STATE, stratify=y,
    )
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    print("Training models:", ", ".join(models.MODELS))
    fitted = models.train_all(X_train, y_train)

    results_df, predictions = evaluate.evaluate_all(fitted, X_test, y_test)
    print("\nResults:\n", results_df.round(3))

    results_df.to_csv(os.path.join(args.out, "model_comparison.csv"))
    evaluate.plot_model_comparison(results_df, os.path.join(args.out, "model_comparison.png"))

    best_name = results_df["f1"].idxmax()
    evaluate.plot_confusion(
        y_test, predictions[best_name], class_names,
        f"Confusion matrix - {best_name}",
        os.path.join(args.out, "confusion_matrix.png"),
    )
    print(f"\nBest model by F1: {best_name}")
    print(f"Saved results to {args.out}/")


if __name__ == "__main__":
    main()
