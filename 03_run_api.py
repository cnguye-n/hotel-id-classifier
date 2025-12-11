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
... (250 lines left)
Collapse
02_train_classifier.py
12 KB
import io, json, os
from pathlib import Path
from typing import List
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

ROOT = Path(".")
ART = ROOT / "artifacts"
MODELS = ROOT / "models"

with open(ART / "class_names.json","r",encoding="utf-8") as f:
    CLASS_NAMES: List[str] = json.load(f)

# -------------------------------------------------
# Config: which model to load
# -------------------------------------------------
# On Windows CMD you can set these before running uvicorn:
#   set MODEL_ARCH=resnet50
#   set MODEL_FILE=resnet50_model.pt
#
# Defaults if env vars are not set:
MODEL_ARCH = os.getenv("MODEL_ARCH", "resnet50")
MODEL_FILE = os.getenv("MODEL_FILE", f"{MODEL_ARCH}_model.pt")

print(f"Loading model arch={MODEL_ARCH}, file={MODEL_FILE}")

# -------------------------------------------------
# ImageNet normalization (must match training)
# -------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# -------------------------------------------------
# Build model from arch name (must match training)
# -------------------------------------------------
def build_model(arch: str, num_classes: int):
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
        raise ValueError(f"Unknown architecture {arch}")

    return m

# -------------------------------------------------
# Load model
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model(MODEL_ARCH, len(CLASS_NAMES))
state = torch.load(MODELS / MODEL_FILE, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

app = FastAPI(title="Hotel ID Classifier")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h2>Hotel ID Classifier (Python-only)</h2>
    <form method="post" action="/predict" enctype="multipart/form-data">
      <input type="file" name="file" accept="image/*" required/>
      <button type="submit">Predict</button>
    </form>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...), topk: int = 3):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = TF(img).unsqueeze(0).to(device)
    with torch.no_grad():
... (14 lines left)
Collapse
03_run_api.py
4 KB
import argparse, json, os
from pathlib import Path
from PIL import Image
import torch, torch.nn as nn
from torchvision import transforms, models
Expand
04_cli_predict.py
3 KB
these are my files after implementing what Christine added in ⁠general . They're updated to allow switching between models. I'm gonna train the models and send the model.pt files into ⁠model-files
Ms. Lee | Christine Nguyen — 12/8/2025 12:10 AM
:chihuahuaLove:
omg thats so smart!!! thank you so much
﻿
import io, json, os
from pathlib import Path
from typing import List
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

ROOT = Path(".")
ART = ROOT / "artifacts"
MODELS = ROOT / "models"

with open(ART / "class_names.json","r",encoding="utf-8") as f:
    CLASS_NAMES: List[str] = json.load(f)

# -------------------------------------------------
# Config: which model to load
# -------------------------------------------------
# On Windows CMD you can set these before running uvicorn:
#   set MODEL_ARCH=resnet50
#   set MODEL_FILE=resnet50_model.pt
#
# Defaults if env vars are not set:
MODEL_ARCH = os.getenv("MODEL_ARCH", "resnet50")
MODEL_FILE = os.getenv("MODEL_FILE", f"{MODEL_ARCH}_model.pt")

print(f"Loading model arch={MODEL_ARCH}, file={MODEL_FILE}")

# -------------------------------------------------
# ImageNet normalization (must match training)
# -------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# -------------------------------------------------
# Build model from arch name (must match training)
# -------------------------------------------------
def build_model(arch: str, num_classes: int):
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
        raise ValueError(f"Unknown architecture {arch}")

    return m

# -------------------------------------------------
# Load model
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model(MODEL_ARCH, len(CLASS_NAMES))
state = torch.load(MODELS / MODEL_FILE, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

app = FastAPI(title="Hotel ID Classifier")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h2>Hotel ID Classifier (Python-only)</h2>
    <form method="post" action="/predict" enctype="multipart/form-data">
      <input type="file" name="file" accept="image/*" required/>
      <button type="submit">Predict</button>
    </form>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...), topk: int = 3):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = TF(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        topk = max(1, min(len(CLASS_NAMES), topk))
        scores, idxs = probs.topk(topk)
    results = []
    for s,i in zip(scores.tolist(), idxs.tolist()):
        results.append({
            "label_index": i,
            "label_name": CLASS_NAMES[i],
            "score": round(float(s) * 100, 2),
            "confidence": f"{round(float(s) * 100, 2)}%"
        })
    return JSONResponse({"topk": results})
