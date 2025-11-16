# analysis/skewness_analysis.py

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew

# -------------------------------------------------------------------
# CONFIG – change these if needed
# -------------------------------------------------------------------
# Path is relative to THIS file. Adjust if your CSV is elsewhere.
DATA_PATH = Path("../full_train_set.csv")   # or "../50k_train_set.csv"
LABEL_COL = "hotel_id"                      # what column defines a "class"
FIG_PATH  = Path("images_per_hotel.png")    # where to save the plot
# -------------------------------------------------------------------


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the Hotels-50K metadata CSV."""
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows\n")
    return df


def compute_image_counts(df: pd.DataFrame, label_col: str) -> pd.Series:
    """
    Return a Series: index = label_id (hotel), value = #images for that label.
    Also print summary stats of this distribution.
    """
    counts = df[label_col].value_counts()
    print("Summary of images per hotel:")
    print(counts.describe(), "\n")
    return counts


def compute_skewness(counts: pd.Series) -> float:
    """
    Compute and print skewness of the class-size distribution.
    Positive skew → heavy right tail (many small classes, few very large).
    """
    sk = skew(counts)
    print(f"Skewness of images-per-hotel distribution: {sk:.4f}")
    if sk > 0:
        print("→ Positively skewed: many hotels with few images, "
              "and a small number of hotels with many images.\n")
    elif sk < 0:
        print("→ Negatively skewed (unusual for this dataset).\n")
    else:
        print("→ Approximately symmetric.\n")
    return sk

def plot_image_count_distribution(
    counts: pd.Series,
    output_path: Path | None = None,
    max_images_to_show: int = 20,
    label_threshold: int = 200,
):
    """
    Plot a bar chart of images-per-hotel distribution.

    - Shows exact bars for 1..max_images_to_show images.
    - Aggregates all hotels with > max_images_to_show images into a single 'N+'
      bar (e.g., '21+').
    - Adds value labels only for bars with height >= label_threshold.
    """
    # How many hotels have k images?
    value_counts = counts.value_counts().sort_index()

    # Split into head (1..max_images_to_show) and tail (> max_images_to_show)
    head = value_counts[value_counts.index <= max_images_to_show]
    tail = value_counts[value_counts.index > max_images_to_show]

    x_labels = list(head.index.astype(str))
    heights = list(head.values)

    # Add aggregated tail as one bin (e.g., "21+")
    if not tail.empty:
        x_labels.append(f"{max_images_to_show}+")
        heights.append(int(tail.sum()))

    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(x_labels)), heights)

    # Add labels only for reasonably large bars
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        if height >= label_threshold:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + (height * 0.02),
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    plt.title("Number of Images per Hotel (Truncated Class Distribution)")
    plt.xlabel("Number of Images per Hotel")
    plt.ylabel("Number of Hotels")
    plt.xticks(range(len(x_labels)), x_labels)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300)
        print(f"Saved figure to {output_path}")

    plt.show()



def main():
    """Orchestrate the full skewness + class distribution analysis."""
    df = load_dataset(DATA_PATH)
    counts = compute_image_counts(df, LABEL_COL)
    _ = compute_skewness(counts)
    plot_image_count_distribution(counts, output_path=FIG_PATH)


# Only run main() when this file is executed directly (not imported)
if __name__ == "__main__":
    main()
