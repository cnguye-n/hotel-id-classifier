# analysis/skewness_analysis.py

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew

#this outputs the skewness of the class distribution (images per hotel) and also creates the
#images_per_hotel.png bar chart and images_per_hotel_boxplot.png boxplot figures
#the functions for those are on the bottom of the file right above main

# -------------------------------------------------------------------
# CONFIG – change these if needed
# -------------------------------------------------------------------
# Path is relative to THIS file. Adjust if your CSV is elsewhere.
DATA_PATH = Path("../full_train_set.csv")   # or "../50k_train_set.csv"
LABEL_COL = "hotel_id"                      # what column defines a "class"
BARCHAT_PATH  = Path("images_per_hotel.png")    # where to save the plot
BOXPLOT_PATH = Path("images_per_hotel_boxplot.png")  # where to save boxplot figure
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

def compute_boxplot_stats(counts: pd.Series) -> dict:
    """
    Compute standard boxplot / IQR statistics and print them.

    Returns a dict with:
      q1, median, q3, iqr, lower_bound, upper_bound,
      lower_whisker, upper_whisker, n_outliers, min, max
    """
    q1 = counts.quantile(0.25)
    median = counts.quantile(0.50)
    q3 = counts.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    lower_whisker = counts[counts >= lower_bound].min()
    upper_whisker = counts[counts <= upper_bound].max()

    outliers = counts[(counts < lower_whisker) | (counts > upper_whisker)]

    stats = {
        "q1": q1,
        "median": median,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "lower_whisker": lower_whisker,
        "upper_whisker": upper_whisker,
        "n_outliers": int(len(outliers)),
        "min": float(counts.min()),
        "max": float(counts.max()),
    }

    print("Boxplot / IQR stats for images-per-hotel")
    print("----------------------------------------")
    print(f"Min:              {stats['min']:.2f}")
    print(f"Q1  (25%):        {stats['q1']:.2f}")
    print(f"Median (50%):     {stats['median']:.2f}")
    print(f"Q3  (75%):        {stats['q3']:.2f}")
    print(f"Max:              {stats['max']:.2f}")
    print(f"IQR (Q3 - Q1):    {stats['iqr']:.2f}")
    print(f"Lower whisker:    {stats['lower_whisker']:.2f}")
    print(f"Upper whisker:    {stats['upper_whisker']:.2f}")
    print(f"1.5*IQR lower bd: {stats['lower_bound']:.2f}")
    print(f"1.5*IQR upper bd: {stats['upper_bound']:.2f}")
    print(f"# of outliers:    {stats['n_outliers']}")
    print()
    return stats

def plot_boxplot_image_counts(
    counts: pd.Series,
    output_path: Path | None = None,
    n_outliers_to_show: int = 5,
):
    """
    Plot:
      - Left: boxplot of images per hotel + top N outliers
      - Right: 'tail' view showing ALL outliers beyond the upper whisker

    Also prints full boxplot stats in the terminal via compute_boxplot_stats().
    """
    # Compute stats + print them in terminal
    stats = compute_boxplot_stats(counts)
    data = counts.values

    # All outliers (beyond upper whisker) and the top-N we want to highlight
    all_outliers = counts[counts > stats["upper_whisker"]].sort_values(ascending=False)
    sample_outliers = all_outliers.head(n_outliers_to_show)

    # Make a 1x2 figure: left = boxplot, right = tail plot
    fig, (ax_box, ax_tail) = plt.subplots(
        1,
        2,
        figsize=(14, 4),
        gridspec_kw={"width_ratios": [2.0, 1.2]},
    )

    # ------------------------------------------------------------------
    # LEFT: main boxplot with top-N outliers
    # ------------------------------------------------------------------
    ax_box.boxplot(
        data,
        vert=False,
        showfliers=False,  # we will plot selected outliers manually
        patch_artist=True,
        boxprops=dict(facecolor="#cce5ff", alpha=0.8),
        medianprops=dict(color="darkred", linewidth=2),
    )

    ax_box.set_title(
        f"Images per Hotel (Boxplot with Top {n_outliers_to_show} Outliers)",
        fontsize=12,
    )
    ax_box.set_xlabel("Number of Images per Hotel")
    ax_box.set_yticks([])

    # Plot top-N outliers as black dots
    for val in sample_outliers:
        ax_box.plot(val, 1, "ko", markersize=4)

    # Zoom around whiskers + sample outliers
    max_to_show = max(
        stats["upper_whisker"] + stats["iqr"],
        sample_outliers.max() if not sample_outliers.empty else stats["upper_whisker"],
    )
    ax_box.set_xlim(0, max_to_show + 10)

    # Q1, Median, Q3 guideline lines
    for x, label, color in [
        (stats["q1"], "Q1", "orange"),
        (stats["median"], "Median", "red"),
        (stats["q3"], "Q3", "orange"),
    ]:
        ax_box.axvline(x, color=color, linestyle="--", linewidth=1)
        ax_box.text(
            x,
            0.85,
            label,
            rotation=90,
            va="bottom",
            ha="center",
            fontsize=8,
        )

    # Single summary box (Q1, Median, Q3, IQR, outlier info)
    summary_text = (
        f"Q1 = {stats['q1']:.0f}\n"
        f"Median = {stats['median']:.0f}\n"
        f"Q3 = {stats['q3']:.0f}\n"
        f"IQR = {stats['iqr']:.0f}\n"
        f"Outliers shown = {len(sample_outliers)} of {stats['n_outliers']}"
    )

    ax_box.text(
        max_to_show - 1,
        0.9,
        summary_text,
        fontsize=8,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    # ------------------------------------------------------------------
    # RIGHT: tail plot – all outliers beyond upper whisker
    # ------------------------------------------------------------------
    ax_tail.set_title(f"Tail: All Outliers (> upper whisker)\n(n = {len(all_outliers)})")
    ax_tail.set_xlabel("Number of Images per Hotel")
    ax_tail.set_yticks([])

    if len(all_outliers) > 0:
        # Plot all outliers as a horizontal "rug" of points at y=1
        ax_tail.scatter(all_outliers.values, [1] * len(all_outliers), s=6, alpha=0.4)

        # Show where the upper whisker is (boundary between "regular" and "outlier")
        ax_tail.axvline(stats["upper_whisker"], color="gray", linestyle="--", linewidth=1)
        ax_tail.text(
            stats["upper_whisker"],
            1.05,
            "Upper whisker",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
            color="gray",
        )

        # Nicely padded x-limits around the outliers
        min_tail = all_outliers.min()
        max_tail = all_outliers.max()
        ax_tail.set_xlim(min_tail - 10, max_tail + 10)
        ax_tail.set_ylim(0.8, 1.2)
    else:
        ax_tail.text(
            0.5,
            0.5,
            "No outliers beyond upper whisker",
            ha="center",
            va="center",
            transform=ax_tail.transAxes,
        )

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300)
        print(f"Saved boxplot + tail figure to {output_path}")

    plt.show()

def main():
    """Output the full skewness + class distribution analysis."""
    df = load_dataset(DATA_PATH)
    counts = compute_image_counts(df, LABEL_COL)
    _ = compute_skewness(counts)
    #plot_image_count_distribution(counts, output_path=BARCHAT_PATH) #uncomment if you haven't created bar chart image already
    #plot_boxplot_image_counts(counts, output_path=BOXPLOT_PATH, n_outliers_to_show=10) #uncomment if you haven't created boxplot image already


# Only run main() when this file is executed directly (not imported)
if __name__ == "__main__":
    main()
