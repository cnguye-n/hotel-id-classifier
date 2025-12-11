import argparse, json, os
from pathlib import Path
from PIL import Image
import torch, torch.nn as nn
from torchvision import transforms, models


# ImageNet normalization (must match training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_model(arch, num_classes):
    """Match architecture used during training."""
    if arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif arch == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif arch == "vgg16":
        m = models.vgg16(weights=None)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)

    elif arch == "vit_b_16":
        m = models.vit_b_16(weights=None)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)

    elif arch == "swin_t":
        m = models.swin_t(weights=None)
        m.head = nn.Linear(m.head.in_features, num_classes)

    else:
        raise ValueError(f"Unknown architecture: {arch}")

    return m


def main(args):
    ROOT = Path(".")

    # Load class names
    with open(ROOT / "artifacts/class_names.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model matching architecture
    model = build_model(args.arch, num_classes)

    # Load weights
    weight_path = ROOT / "models" / args.model
    print(f"Loading model weights: {weight_path}")
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Same normalization as training!!!
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Load image
    img = Image.open(args.image).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
        topk = min(args.topk, num_classes)
        scores, idxs = probs.topk(topk)

    # Print results
    print("\nTop predictions:\n")
    for s, i in zip(scores.tolist(), idxs.tolist()):
        print(f"{class_names[i]:30s} | {s:.4f}")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--image", required=True)
    ap.add_argument("--topk", type=int, default=3)

    # NEW:
    ap.add_argument("--arch", type=str, required=True,
                    help="resnet18 | resnet50 | vgg16 | vit_b_16 | swin_t")

    ap.add_argument("--model", type=str, required=True,
                    help="filename inside ./models/ (e.g., resnet50_model.pt)")

    args = ap.parse_args()
    main(args)
