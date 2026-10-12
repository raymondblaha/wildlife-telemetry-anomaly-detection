"""Run the public-safe synthetic manuscript artifact pipeline end to end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=["wtd_primary", "wtd_four_pings", "mule_60_40"],
        default="wtd_primary",
    )
    parser.add_argument(
        "--data-dir",
        default="data/sample/manuscript_synthetic",
        help="Directory for generated synthetic train/test CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Artifact directory. Defaults to outputs/<profile>.",
    )
    parser.add_argument("--train-animals", type=int, default=16)
    parser.add_argument("--test-animals", type=int, default=8)
    parser.add_argument("--rows-per-animal", type=int, default=180)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--random-state", type=int, default=13)
    return parser.parse_args()


def run_step(label: str, args: list[str]) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(args))
    subprocess.run(args, check=True)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir or f"outputs/{args.profile}")
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    python = sys.executable
    profile = args.profile
    train_csv = data_dir / f"{profile}_train.csv"
    test_csv = data_dir / f"{profile}_test.csv"

    run_step(
        "1. Generate manuscript-shaped synthetic telemetry",
        [
            python,
            str(root / "scripts" / "generate_manuscript_synthetic_inputs.py"),
            "--profile",
            profile,
            "--output-dir",
            str(data_dir),
            "--train-animals",
            str(args.train_animals),
            "--test-animals",
            str(args.test_animals),
            "--rows-per-animal",
            str(args.rows_per_animal),
            "--random-state",
            str(args.random_state),
        ],
    )
    run_step(
        "2. Preprocess features with train-only scaler/encoder",
        [
            python,
            str(root / "scripts" / "prepare_manuscript_features.py"),
            "--profile",
            profile,
            "--train-csv",
            str(train_csv),
            "--test-csv",
            str(test_csv),
            "--output-dir",
            str(output_dir),
            "--random-state",
            str(args.random_state),
        ],
    )
    run_step(
        "3. Train dense autoencoder",
        [
            python,
            str(root / "scripts" / "train_manuscript_autoencoder.py"),
            "--profile",
            profile,
            "--processed-dir",
            str(output_dir),
            "--epochs",
            str(args.epochs),
            "--random-state",
            str(args.random_state),
        ],
    )
    run_step(
        "4. Apply frozen row and animal-level rule",
        [
            python,
            str(root / "scripts" / "apply_manuscript_frozen_rule.py"),
            "--profile",
            profile,
            "--processed-dir",
            str(output_dir),
        ],
    )
    run_step(
        "5. Summarize feature residual contributions",
        [
            python,
            str(root / "scripts" / "manuscript_feature_contributions.py"),
            "--profile",
            profile,
            "--processed-dir",
            str(output_dir),
        ],
    )
    print("\nSynthetic manuscript artifacts are ready.")
    print(f"Artifact directory: {output_dir}")
    print("Open dashboard with:")
    print("streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
