import argparse, json
from pathlib import Path
from PIL import Image
import torch, torch.nn as nn
from torchvision import transforms, models

def main(args):
    ROOT = Path(".")
    with open(ROOT/"artifacts/class_names.json","r",encoding="utf-8") as f:
        class_names = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    state = torch.load(ROOT/"models/model.pt", map_location=device)
    model.load_state_dict(state); model.to(device); model.eval()

    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    img = Image.open(args.image).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1)[0]
        topk = min(args.topk, len(class_names))
        scores, idxs = p.topk(topk)
    for s,i in zip(scores.tolist(), idxs.tolist()):
        print(f"{class_names[i]}  |  {s:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()
    main(args)
