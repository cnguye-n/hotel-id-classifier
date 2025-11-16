import argparse, json, os
from pathlib import Path
import pandas as pd
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class ManifestDataset(Dataset):
    def __init__(self, csv_path, train=True, img_size=224):
        self.df = pd.read_csv(csv_path)
        self.train = train
        if train:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1,0.1,0.1,0.05),
                transforms.ToTensor(),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["image_path"]).convert("RGB")
        x = self.tf(img)
        y = int(row["label_idx"])
        return x, y

def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x,y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            ys.extend(y.cpu().tolist())
            ps.extend(pred.cpu().tolist())
    return accuracy_score(ys, ps)

def main(args):
    root = Path(".")
    artifacts = root / "artifacts"
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(artifacts / "class_names.json","r",encoding="utf-8") as f:
        class_names = json.load(f)
    num_classes = len(class_names)

    train_ds = ManifestDataset("data/splits/train.csv", train=True, img_size=args.img_size)
    val_ds   = ManifestDataset("data/splits/val.csv",   train=False, img_size=args.img_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_val = 0.0
    patience, bad = args.patience, 0

    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x,y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss))

        val_acc = evaluate(model, val_loader, device)
        print(f"Val Acc: {val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), models_dir / "model.pt")
            bad = 0
            print("✓ Saved best model.")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    print(f"Best Val Acc: {best_val:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=3)
    args = ap.parse_args()
    main(args)
