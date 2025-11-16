import argparse, os, json, math, random, time
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm

random.seed(42)

def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)

def download_image(url: str, out_path: Path, timeout=15):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception:
        if out_path.exists():
            out_path.unlink(missing_ok=True)
        return False

def main(args):
    root = Path(".")
    data_dir = root / "data"
    raw_dir = data_dir / "raw"
    splits_dir = data_dir / "splits"
    cache_dir = data_dir / "cache"
    artifacts = root / "artifacts"
    for d in [raw_dir, splits_dir, cache_dir, artifacts]:
        d.mkdir(parents=True, exist_ok=True)

    print("Reading CSVs...")
    chain_df = pd.read_csv(args.chain_csv)
    hotel_df = pd.read_csv(args.hotel_csv)
    test_df  = pd.read_csv(args.test_csv)

    # Merge to get names
    merged = test_df.merge(hotel_df, on="hotel_id", how="left") \
                    .merge(chain_df, on="chain_id", how="left", suffixes=("", "_chain"))
    # Decide label field
    if args.label_field == "hotel":
        merged["label_id"] = merged["hotel_id"].astype(str)
        merged["label_name"] = merged["hotel_name"].fillna("UnknownHotel")
    else:
        merged["label_id"] = merged["chain_id"].astype(str)
        merged["label_name"] = merged["chain_name"].fillna("UnknownChain")

    # --- Diagnostic: check class distribution BEFORE filtering ---
    counts = merged["label_id"].value_counts()
    print("\nClass distribution BEFORE filtering:")
    print(counts.describe())

    print("\nClasses with >= 5 images:", (counts >= 5).sum())
    print("Classes with >= 3 images:", (counts >= 3).sum())
    print("Classes with >= 2 images:", (counts >= 2).sum())
    print("Classes with >= 1 image:", (counts >= 1).sum())
    print("-----------------------------------------------------\n")
    # ---------------------------------------------------------------

    # Optional: filter only classes with >= min_images
    gb = merged.groupby("label_id")
    keep_ids = gb.size()[gb.size() >= args.min_images_per_class].index
    merged = merged[merged["label_id"].isin(keep_ids)].reset_index(drop=True)
    print(f"Kept {merged['label_id'].nunique()} classes after min_images_per_class={args.min_images_per_class}")

    # Download images
    records = []
    print("Downloading images (this may take a while)...")
    for row in tqdm(merged.itertuples(index=False), total=len(merged)):
        img_id = getattr(row, "image_id")
        url    = getattr(row, "image_url")
        label  = getattr(row, "label_id")
        # path: data/raw/<label>/<image_id>.jpg
        out_path = raw_dir / safe_filename(label) / f"{safe_filename(str(img_id))}.jpg"
        if out_path.exists():
            ok = True
        else:
            ok = download_image(url, out_path)
        if ok:
            records.append({
                "image_path": str(out_path.as_posix()),
                "label_id": str(label),
                "label_name": getattr(row, "label_name"),
            })

    df = pd.DataFrame(records)
    if df.empty:
        raise SystemExit("No images downloaded successfully. Please check your URLs/connection.")

    # Build label index
    label_ids = sorted(df["label_id"].unique())
    id2idx = {lid: i for i, lid in enumerate(label_ids)}
    df["label_idx"] = df["label_id"].map(id2idx)

    # Train/val/test split (stratified by class)
    def stratified_split(group, train=0.7, val=0.15):
        n = len(group)
        idx = list(range(n))
        random.shuffle(idx)
        n_train = math.floor(n * train)
        n_val   = math.floor(n * val)
        s_train = group.iloc[idx[:n_train]]
        s_val   = group.iloc[idx[n_train:n_train+n_val]]
        s_test  = group.iloc[idx[n_train+n_val:]]
        return s_train, s_val, s_test

    train_rows, val_rows, test_rows = [], [], []
    for lid, g in df.groupby("label_id"):
        t, v, te = stratified_split(g, train=args.train_ratio, val=args.val_ratio)
        train_rows.append(t); val_rows.append(v); test_rows.append(te)

    train_df = pd.concat(train_rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_df   = pd.concat(val_rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
    test_df2 = pd.concat(test_rows).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Save manifests
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df2.to_csv(splits_dir / "test.csv", index=False)

    # Label mapping artifacts
    label_mapping = {}
    for lid, sub in df.groupby("label_id"):
        label_mapping[str(lid)] = {
            "label_idx": int(id2idx[lid]),
            "label_name": sub["label_name"].iloc[0]
        }
    with open(artifacts / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)

    class_names = [label_mapping[lid]["label_name"] for lid in label_ids]
    with open(artifacts / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    print("Done. Manifests saved in data/splits and mappings in artifacts/")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain-csv", default="chain_info.csv")
    ap.add_argument("--hotel-csv", default="hotel_info.csv")
    ap.add_argument("--test-csv",  default="50k_train_set.csv") #change to full_train_set.csv for the full 1 million files
    ap.add_argument("--label-field", choices=["hotel","chain"], default="hotel",
                    help="")
    ap.add_argument("--min-images-per-class", type=int, default=5,
                    help="")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    args = ap.parse_args()
    main(args)
