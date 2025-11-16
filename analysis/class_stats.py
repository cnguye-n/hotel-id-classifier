import pandas as pd


def load_data(
    train_csv: str = "50k_train_set.csv",
    hotel_csv: str = "hotel_info.csv",
    chain_csv: str = "chain_info.csv",
):
    """Load training image metadata, hotel info, and chain info."""
    train_df = pd.read_csv(train_csv)
    hotel_df = pd.read_csv(hotel_csv)
    chain_df = pd.read_csv(chain_csv)
    return train_df, hotel_df, chain_df


def compute_hotel_image_counts(train_df, hotel_df, chain_df):
    """
    Count how many images each hotel_id has, then attach hotel and chain info.
    Returns a DataFrame sorted by num_images (descending).
    """
    # Count images per hotel_id
    counts = (
        train_df.groupby("hotel_id")
        .size()
        .reset_index(name="num_images")
    )

    # Attach hotel_name + chain_id
    counts = counts.merge(
        hotel_df[["hotel_id", "hotel_name", "chain_id"]],
        on="hotel_id",
        how="left",
    )

    # Attach chain_name
    counts = counts.merge(chain_df, on="chain_id", how="left")

    # In case of missing chain_name, fill with "unknown"
    counts["chain_name"] = counts["chain_name"].fillna("unknown")

    # Sort by number of images (most first)
    counts_sorted = counts.sort_values("num_images", ascending=False)
    return counts_sorted


def print_top_and_bottom(counts_sorted, n: int = 10):
    """Print top n and bottom n hotels by number of images."""
    print(f"\nTop {n} hotels with the most images:")
    print(
        counts_sorted.head(n)[["hotel_id", "hotel_name", "num_images"]]
        .to_string(index=False)
    )

    print(f"\nBottom {n} hotels with the fewest images:")
    bottom = counts_sorted.sort_values("num_images", ascending=True).head(n)
    print(
        bottom[["hotel_id", "hotel_name", "num_images"]]
        .to_string(index=False)
    )


def save_counts_csv(counts_sorted, out_csv: str = "hotel_image_counts.csv"):
    """
    Save a CSV with hotel_id, hotel_name, chain_id, chain_name, num_images,
    sorted by num_images (descending).
    """
    cols = ["hotel_id", "hotel_name", "chain_id", "chain_name", "num_images"]
    counts_sorted[cols].to_csv(out_csv, index=False)
    print(f"\nSaved full hotel image stats to: {out_csv}")


if __name__ == "__main__":
    # If you want to use full_train_set.csv instead, change the first argument.
    train_df, hotel_df, chain_df = load_data("../full_train_set.csv",
                                             "../hotel_info.csv",
                                             "../chain_info.csv")

    counts_sorted = compute_hotel_image_counts(train_df, hotel_df, chain_df)

    print_top_and_bottom(counts_sorted, n=10)

    #save_counts_csv(counts_sorted, out_csv="hotel_image_counts.csv") #already have it created
