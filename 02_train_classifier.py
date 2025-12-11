import argparse, json
from pathlib import Path
import pandas as pd
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm


# -------------------------------
# ImageNet normalization
# -------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# -------------------------------
# Dataset
# -------------------------------
class ManifestDataset(Dataset):
    def __init__(self, csv_path, train=True, img_size=224):
        self.df = pd.read_csv(csv_path)
        self.train = train

        if train:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image_path"]
        try:
            img = Image.open(path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError):
            # fallback to a previous row like V1
            j = (idx - 1) % len(self.df)
            row = self.df.iloc[j]
            img = Image.open(row["image_path"]).convert("RGB")

        x = self.tf(img)
        y = int(row["label_idx"])
        return x, y


# -------------------------------
# Build model from architecture name
# -------------------------------
def build_model(arch, num_classes):
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif arch == "vgg16":
        m = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        # replace last classifier layer
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)

    elif arch == "vit_b_16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)

    elif arch == "swin_t":
        m = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        m.head = nn.Linear(m.head.in_features, num_classes)

    else:
        raise ValueError(f"Unknown architecture {arch}")

    return m


# -------------------------------
# Helper: which params are "head"/classifier vs backbone
# -------------------------------
HEAD_PREFIXES = {
    "resnet18": ["fc."],
    "resnet50": ["fc."],
    "vgg16": ["classifier."],   # we treat full classifier as "head"
    "vit_b_16": ["heads."],
    "swin_t": ["head."],
}


def is_head_param(name, arch):
    prefixes = HEAD_PREFIXES.get(arch, [])
    return any(name.startswith(p) for p in prefixes)


def set_backbone_requires_grad(model, arch, requires_grad: bool):
    """
    Freeze/unfreeze everything except the classifier/head.
    """
    for name, p in model.named_parameters():
        if is_head_param(name, arch):
            # leave head alone here; it will be handled by optimizer selection
            continue
        p.requires_grad = requires_grad


def get_param_groups(model, arch, lr_backbone, lr_head):
    """
    Returns param groups with different LRs for backbone vs head.
    """
    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if is_head_param(name, arch):
            head_params.append(p)
        else:
            backbone_params.append(p)
    return [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ]


def get_head_params(model, arch):
    """
    Returns only the head/classifier parameters for Phase 1 optimizer.
    """
    head_params = []
    for name, p in model.named_parameters():
        if is_head_param(name, arch) and p.requires_grad:
            head_params.append(p)
    return head_params


# -------------------------------
# Class weights (for imbalance)
# -------------------------------
def make_class_weights(train_csv, num_classes, device):
    df = pd.read_csv(train_csv)
    cnt = df["label_idx"].value_counts().to_dict()
    weights = torch.ones(num_classes, dtype=torch.float32)
    for i in range(num_classes):
        n = cnt.get(i, 0)
        weights[i] = 1.0 / max(1, n)
    # normalize (optional)
    weights = weights * (num_classes / weights.sum())
    return weights.to(device)


# -------------------------------
# Evaluation (Top-1, Top-5)
# -------------------------------
def topk_accuracy(logits, y, k=1):
    with torch.no_grad():
        _, pred = logits.topk(k, dim=1)
        correct = pred.eq(y.view(-1, 1)).sum().item()
        return correct / y.size(0)


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total += y.size(0)
            correct1 += int(topk_accuracy(logits, y, k=1) * y.size(0))
            k5 = min(5, logits.size(1))
            correct5 += int(topk_accuracy(logits, y, k=k5) * y.size(0))

    if total == 0:
        return 0.0, 0.0
    return correct1 / total, correct5 / total


# -------------------------------
# One training phase (used twice)
# -------------------------------
def train_one_phase(model, loader, val_loader, device, epochs, opt, loss_fn, save_path, arch, best_val):
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.detach()))

        top1, top5 = evaluate(model, val_loader, device)
        print(f"Val Top1: {top1:.4f} | Top5: {top5:.4f}")
        if top1 > best_val:
            best_val = top1
            torch.save(model.state_dict(), save_path)
            print("✓ Saved BEST model")

    return best_val


# -------------------------------
# Main training (2-phase)
# -------------------------------
def main(args):
    root = Path(".")
    artifacts = root / "artifacts"
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load class names
    with open(artifacts / "class_names.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)
    num_classes = len(class_names)

    # Dataset & loaders
    train_ds = ManifestDataset("data/splits/train.csv", train=True,  img_size=args.img_size)
    val_ds   = ManifestDataset("data/splits/val.csv",   train=False, img_size=args.img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.arch, num_classes).to(device)

    print(f"\nUsing architecture: {args.arch}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Loss (optionally class-weighted)
    if args.class_weighted:
        class_weights = make_class_weights("data/splits/train.csv", num_classes, device)
        print("Using class-weighted CrossEntropy.")
    else:
        class_weights = None
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    best_val = -1.0
    save_path = models_dir / f"{args.arch}_model.pt"

    # -------- Phase 1: freeze backbone, train classifier/head only --------
    if args.freeze_backbone_epochs > 0:
        print(f"\n[Phase 1] Freeze backbone, train head for {args.freeze_backbone_epochs} epochs")
        set_backbone_requires_grad(model, args.arch, requires_grad=False)

        # ensure head params require_grad = True
        for name, p in model.named_parameters():
            if is_head_param(name, args.arch):
                p.requires_grad = True

        head_params = get_head_params(model, args.arch)
        opt_head = torch.optim.AdamW(head_params, lr=args.lr_head, weight_decay=1e-4)

        best_val = train_one_phase(
            model, train_loader, val_loader, device,
            epochs=args.freeze_backbone_epochs,
            opt=opt_head,
            loss_fn=loss_fn,
            save_path=save_path,
            arch=args.arch,
            best_val=best_val
        )

    # -------- Phase 2: unfreeze, fine-tune all --------
    if args.unfreeze_epochs > 0:
        print(f"\n[Phase 2] Unfreeze all, fine-tune for {args.unfreeze_epochs} epochs")
        for p in model.parameters():
            p.requires_grad = True

        param_groups = get_param_groups(
            model, args.arch,
            lr_backbone=args.lr_backbone,
            lr_head=args.lr_head
        )
        opt_all = torch.optim.AdamW(param_groups, weight_decay=1e-4)

        best_val = train_one_phase(
            model, train_loader, val_loader, device,
            epochs=args.unfreeze_epochs,
            opt=opt_all,
            loss_fn=loss_fn,
            save_path=save_path,
            arch=args.arch,
            best_val=best_val
        )

    print(f"\nDone. Best model saved to {save_path} with best Val Top1 = {best_val:.4f}")


# -------------------------------
# Arg parser
# -------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--arch", type=str, default="resnet50",
                    help="resnet50 | resnet18 | vgg16 | vit_b_16 | swin_t")

    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)

    # Two-stage fine-tune
    ap.add_argument("--freeze-backbone-epochs", type=int, default=3,
                    help="Phase 1: train only classifier/head")
    ap.add_argument("--unfreeze-epochs", type=int, default=10,
                    help="Phase 2: fine-tune full model")
    ap.add_argument("--lr-head", type=float, default=1e-3,
                    help="LR for classifier/head")
    ap.add_argument("--lr-backbone", type=float, default=1e-4,
                    help="LR for backbone when unfrozen")

    # Class-weighted loss
    ap.add_argument("--class-weighted", action="store_true",
                    help="Use class-weighted CrossEntropy")

    args = ap.parse_args()
    main(args)
