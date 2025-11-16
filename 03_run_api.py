import io, json
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
state = torch.load(MODELS / "model.pt", map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

TF = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

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
