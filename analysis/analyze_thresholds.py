# analysis/analyze_thresholds.py
"""
Analyze how different min-images-per-class thresholds affect
the number of classes and images in the dataset, using only
the main train CSV (e.g., 50k_train_set.csv or full_train_set.csv).
"""

import argparse
from pathlib import Path
import pandas as pd


def analyze_thresholds(counts, thresholds):
    """Given a Series of counts per class, print summary for each threshold."""
    total_classes = len(counts)
    total_images = int(counts.sum())

    print("\nTotal classes:", total_classes)
    print("Total images :", total_images)

    print("\nMin-images-per-class threshold analysis:")
    print(
        "Thresh\tClasses_kept\tClasses_removed\tImages_kept\tImages_removed\t%classes_kept\t%images_kept"
    )
    print("-" * 90)

    rows = []

    for t in thresholds:
        mask = counts >= t
        classes_kept = int(mask.sum())
        images_kept = int(counts[mask].sum())
        classes_removed = total_classes - classes_kept
        images_removed = total_images - images_kept

        pct_classes_kept = classes_kept / total_classes * 100
        pct_images_kept = images_kept / total_images * 100

        row = "{:<5} | {:<12} | {:<14} | {:<12} | {:<14} | {:<12.2f} | {:<12.2f}".format(
        t, classes_kept, classes_removed, images_kept, images_removed,
        pct_classes_kept, pct_images_kept)
        print(row)


        rows.append(
            {
                "threshold": t,
                "classes_kept": classes_kept,
                "classes_removed": classes_removed,
                "images_kept": images_kept,
                "images_removed": images_removed,
                "pct_classes_kept": pct_classes_kept,
                "pct_images_kept": pct_images_kept,
            }
        )

    return rows


def main(args):
    train_path = Path(args.train_csv)
    print(f"Reading train CSV from: {train_path}")
    df = pd.read_csv(train_path)

    # Decide which column defines a "class"
    if args.label_field == "hotel":
        label_col = "hotel_id"
    else:
        label_col = "chain_id"

    if label_col not in df.columns:
        print(
            f"\nERROR: Column '{label_col}' not found in {train_path}.\n"
            f"Available columns are: {list(df.columns)}"
        )
        return

    print(f"Using '{label_col}' as the class label.\n")

    # Count images per class
    counts = df[label_col].value_counts()

    print("Class distribution BEFORE filtering (images per class):")
    print(counts.describe())

    # Thresholds to test
    thresholds = [1, 2, 3, 4, 5, 6, 10, 20, 100]

    rows = analyze_thresholds(counts, thresholds)

    # Optional: save to CSV
    out_path = Path("threshold_results.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved detailed results to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-csv",
        default="../full_train_set.csv",
        help="Path to train CSV (e.g., ../50k_train_set.csv or ../full_train_set.csv)",
    )
    ap.add_argument(
        "--label-field",
        choices=["hotel", "chain"],
        default="hotel",
        help="Whether to treat each hotel_id or chain_id as a class.",
    )
    args = ap.parse_args()
    main(args)
