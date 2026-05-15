import React, { useState } from "react";

// ============================================================================
// MlPipeline.jsx
// A self-contained, beginner-friendly visual documentation of the complete
// machine learning fine-tuning pipeline for the handwritten OCR model.
//
// It renders:
//   - Pipeline overview cards
//   - Recommended models & datasets
//   - Tabbed code blocks (folder structure, dataset, preprocessing,
//     training, evaluation, inference, FastAPI)
//   - Copy-to-clipboard for each snippet
//
// To use it inside App.jsx, simply:
//     import MlPipeline from "./MlPipeline";
//     <MlPipeline dark={dark} />
// ============================================================================

const TABS = [
  { id: "structure", label: "Folder Structure", icon: "solar:folder-with-files-bold" },
  { id: "dataset", label: "Dataset Loading", icon: "solar:database-bold" },
  { id: "preprocess", label: "Preprocessing", icon: "solar:tuning-2-bold" },
  { id: "augment", label: "Augmentation", icon: "solar:magic-stick-3-bold" },
  { id: "train", label: "Training Loop", icon: "solar:cpu-bolt-bold" },
  { id: "evaluate", label: "Evaluation", icon: "solar:chart-2-bold" },
  { id: "inference", label: "Inference", icon: "solar:scanner-2-bold" },
  { id: "api", label: "FastAPI Server", icon: "solar:server-2-bold" },
];

// ---------- Code snippets ----------
const CODE = {
  structure: `handwriting-ocr/
├── data/
│   ├── raw/                # original downloaded datasets (IAM, CVL, EMNIST...)
│   ├── processed/          # cleaned image-text pairs
│   ├── train.csv           # columns: image_path,text
│   ├── val.csv
│   └── test.csv
├── notebooks/
│   └── explore.ipynb
├── src/
│   ├── config.py           # all hyperparameters & paths
│   ├── dataset.py          # PyTorch Dataset class
│   ├── preprocess.py       # grayscale, deskew, denoise, threshold
│   ├── augment.py          # rotation, blur, slant, elastic distortion
│   ├── model.py            # TrOCR loader + processor
│   ├── train.py            # fine-tuning loop
│   ├── evaluate.py         # CER / WER metrics
│   ├── inference.py        # single-image prediction
│   └── utils.py
├── api/
│   ├── main.py             # FastAPI server
│   └── requirements.txt
├── checkpoints/            # saved fine-tuned models
│   └── best_model/
├── requirements.txt
└── README.md`,

  dataset: `# src/dataset.py
# Beginner-friendly PyTorch Dataset for handwritten image-text pairs.

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class HandwritingDataset(Dataset):
    """
    Loads handwriting images and their ground-truth text labels.
    Expects a CSV with columns: image_path, text
    """
    def __init__(self, csv_file, processor, root_dir="data/processed",
                 max_target_length=128, transform=None):
        self.df = pd.read_csv(csv_file)
        self.processor = processor          # TrOCR processor (image + tokenizer)
        self.root_dir = root_dir
        self.max_target_length = max_target_length
        self.transform = transform          # optional augmentation pipeline

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.root_dir, row["image_path"])
        text = str(row["text"])

        # Always load as RGB (TrOCR expects 3 channels)
        image = Image.open(image_path).convert("RGB")

        # Apply augmentation if provided (training only)
        if self.transform is not None:
            image = self.transform(image)

        # Processor handles resize + normalize + tokenization
        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
        ).input_ids

        # Replace pad tokens with -100 so the loss function ignores them
        pad_id = self.processor.tokenizer.pad_token_id
        labels = [tok if tok != pad_id else -100 for tok in labels]

        return {"pixel_values": pixel_values, "labels": labels}`,

  preprocess: `# src/preprocess.py
# Classical CV preprocessing: grayscale, denoise, threshold, deskew.

import cv2
import numpy as np

def to_grayscale(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def denoise(gray):
    # Non-local means works well on handwriting scans
    return cv2.fastNlMeansDenoising(gray, h=15)

def threshold(gray):
    # Adaptive threshold handles uneven paper lighting
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15
    )

def deskew(binary):
    """Rotate the image so text lines are horizontal."""
    coords = np.column_stack(np.where(binary < 255))
    if coords.size == 0:
        return binary
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = binary.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        binary, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

def preprocess_image(path, out_size=(384, 384)):
    img = cv2.imread(path)
    gray = to_grayscale(img)
    gray = denoise(gray)
    binary = threshold(gray)
    deskewed = deskew(binary)
    resized = cv2.resize(deskewed, out_size, interpolation=cv2.INTER_AREA)
    # Back to 3-channel RGB for TrOCR
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    return rgb`,

  augment: `# src/augment.py
# Albumentations-based augmentation tuned for handwriting.

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

def build_train_transform():
    return A.Compose([
        A.Rotate(limit=4, p=0.5, border_mode=0, value=255),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(var_limit=(10.0, 40.0), p=0.3),
        A.Affine(shear={"x": (-10, 10)}, p=0.4),     # simulate slant
        A.ElasticTransform(alpha=30, sigma=4, p=0.3), # cursive distortion
    ])

class PILAlbumentations:
    """Wrap albumentations so it works with PIL images."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, pil_image):
        arr = np.array(pil_image)
        augmented = self.transform(image=arr)["image"]
        return Image.fromarray(augmented)`,

  train: `# src/train.py
# Fine-tune TrOCR on handwritten English text.

import torch
from torch.utils.data import DataLoader
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

from dataset import HandwritingDataset
from augment import build_train_transform, PILAlbumentations
from evaluate import compute_cer, compute_wer

# ---------------- Config ----------------
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME    = "microsoft/trocr-base-handwritten"
BATCH_SIZE    = 8
EPOCHS        = 10
LR            = 5e-5
SAVE_DIR      = "checkpoints/best_model"

# ---------------- Load model + processor ----------------
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model     = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(DEVICE)

# Required decoder configuration for generation
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id           = processor.tokenizer.pad_token_id
model.config.vocab_size             = model.config.decoder.vocab_size
model.config.eos_token_id           = processor.tokenizer.sep_token_id
model.config.max_length             = 128
model.config.num_beams              = 4

# ---------------- Datasets ----------------
train_tf = PILAlbumentations(build_train_transform())

train_ds = HandwritingDataset("data/train.csv", processor, transform=train_tf)
val_ds   = HandwritingDataset("data/val.csv",   processor)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ---------------- Optimizer + scheduler ----------------
optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
)

# ---------------- Training loop ----------------
best_cer = float("inf")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        pixel_values = batch["pixel_values"].to(DEVICE)
        labels = torch.tensor(batch["labels"]).to(DEVICE)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Train loss: {avg_loss:.4f}")

    # ---------- Validation ----------
    model.eval()
    preds, refs = [], []
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(DEVICE)
            generated = model.generate(pixel_values)
            decoded = processor.batch_decode(generated, skip_special_tokens=True)
            label_ids = [[tok for tok in lbl if tok != -100] for lbl in batch["labels"]]
            truth = processor.batch_decode(label_ids, skip_special_tokens=True)
            preds.extend(decoded)
            refs.extend(truth)

    cer = compute_cer(preds, refs)
    wer = compute_wer(preds, refs)
    print(f"Validation  CER: {cer:.4f}  WER: {wer:.4f}")

    # Save best checkpoint
    if cer < best_cer:
        best_cer = cer
        model.save_pretrained(SAVE_DIR)
        processor.save_pretrained(SAVE_DIR)
        print(f"✅ Saved new best model to {SAVE_DIR}")`,

  evaluate: `# src/evaluate.py
# Character Error Rate (CER) and Word Error Rate (WER).
# Uses the 'jiwer' library — pip install jiwer

from jiwer import cer, wer

def compute_cer(predictions, references):
    """Lower is better. 0.0 means perfect character-level match."""
    return cer(references, predictions)

def compute_wer(predictions, references):
    """Lower is better. Measures word-level errors."""
    return wer(references, predictions)

if __name__ == "__main__":
    # Quick standalone test on the held-out test set
    import torch
    from torch.utils.data import DataLoader
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from dataset import HandwritingDataset

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_DIR = "checkpoints/best_model"

    processor = TrOCRProcessor.from_pretrained(SAVE_DIR)
    model = VisionEncoderDecoderModel.from_pretrained(SAVE_DIR).to(DEVICE).eval()

    test_ds = HandwritingDataset("data/test.csv", processor)
    loader = DataLoader(test_ds, batch_size=8)

    preds, refs = [], []
    for batch in loader:
        pv = batch["pixel_values"].to(DEVICE)
        out = model.generate(pv)
        preds.extend(processor.batch_decode(out, skip_special_tokens=True))
        label_ids = [[t for t in l if t != -100] for l in batch["labels"]]
        refs.extend(processor.batch_decode(label_ids, skip_special_tokens=True))

    print("Test CER:", compute_cer(preds, refs))
    print("Test WER:", compute_wer(preds, refs))`,

  inference: `# src/inference.py
# Run the fine-tuned model on a single uploaded image.

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from preprocess import preprocess_image
import numpy as np

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "checkpoints/best_model"

# Load once at module import (fast subsequent calls)
processor = TrOCRProcessor.from_pretrained(SAVE_DIR)
model     = VisionEncoderDecoderModel.from_pretrained(SAVE_DIR).to(DEVICE).eval()

def predict_text(image_path: str) -> str:
    """
    Takes a path to a handwritten image and returns the recognized text.
    """
    # Step 1: classical CV cleanup
    cleaned = preprocess_image(image_path)            # numpy RGB
    pil_image = Image.fromarray(cleaned)

    # Step 2: model inference
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=128, num_beams=4)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    print("Recognized text:", predict_text(path))`,

  api: `# api/main.py
# FastAPI server that exposes the fine-tuned OCR model.
# Run with:  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

import io
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.inference import predict_text

app = FastAPI(title="Handwriting OCR API", version="1.0.0")

# Allow the React web app to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png"}

@app.get("/")
def root():
    return {"status": "ok", "message": "Handwriting OCR API is running."}

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Only JPG/PNG images are allowed.")

    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10MB).")

    # Validate image bytes
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Save to a temp file so preprocess_image (cv2.imread) can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        text = predict_text(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return {"text": text, "length": len(text)}`,
};

const REQUIREMENTS = `# requirements.txt
torch>=2.1.0
torchvision>=0.16.0
transformers>=4.36.0
sentencepiece>=0.1.99
pillow>=10.0.0
opencv-python>=4.8.0
albumentations>=1.3.1
pandas>=2.0.0
numpy>=1.24.0
tqdm>=4.66.0
jiwer>=3.0.3
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6`;

const DATASETS = [
  {
    name: "IAM Handwriting Database",
    desc: "1,539 scanned pages, 13,353 isolated lines from 657 writers. Gold standard for cursive English.",
    size: "~1.7 GB",
    url: "https://fki.tic.heia-fr.ch/databases/iam-handwriting-database",
    icon: "solar:notebook-bookmark-bold",
  },
  {
    name: "CVL Database",
    desc: "310 writers, 7 different texts. Excellent for diverse writer styles.",
    size: "~2.5 GB",
    url: "https://zenodo.org/record/1492267",
    icon: "solar:users-group-rounded-bold",
  },
  {
    name: "EMNIST Letters",
    desc: "145,600 handwritten character images across 26 classes. Great for character-level pretraining.",
    size: "~535 MB",
    url: "https://www.nist.gov/itl/products-and-services/emnist-dataset",
    icon: "solar:letter-bold",
  },
  {
    name: "RIMES Database",
    desc: "12,723 pages of French handwriting — useful for transfer learning to mixed scripts.",
    size: "~1.2 GB",
    url: "http://www.a2ialab.com/doku.php?id=rimes_database:start",
    icon: "solar:document-bold",
  },
];

const MODELS = [
  {
    name: "TrOCR (Microsoft)",
    badge: "Recommended",
    desc: "Transformer encoder-decoder pretrained on 684M synthetic + IAM. Best out-of-the-box accuracy for cursive English.",
    icon: "solar:cpu-bolt-bold",
    color: "from-blue-500 to-indigo-600",
  },
  {
    name: "CRNN + CTC",
    desc: "Lightweight CNN-RNN with CTC loss. Great when you need a small, fast model on edge devices.",
    icon: "solar:layers-bold",
    color: "from-purple-500 to-pink-500",
  },
  {
    name: "Tesseract LSTM",
    desc: "Open-source classic. Fine-tune via lstmtraining for printed-style handwriting and forms.",
    icon: "solar:eye-bold",
    color: "from-emerald-500 to-teal-600",
  },
  {
    name: "EasyOCR Custom",
    desc: "PyTorch-based, easy custom training. Good middle-ground between TrOCR and Tesseract.",
    icon: "solar:bolt-bold",
    color: "from-orange-500 to-rose-500",
  },
];

// ---------- Component ----------
export default function MlPipeline({ dark = false }) {
  const [activeTab, setActiveTab] = useState("structure");
  const [copied, setCopied] = useState("");

  const handleCopy = async (key, code) => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(key);
      setTimeout(() => setCopied(""), 1800);
    } catch {
      setCopied("err");
      setTimeout(() => setCopied(""), 1800);
    }
  };

  const cardBase = dark
    ? "bg-slate-900/60 border-slate-800"
    : "bg-white/80 border-slate-200";

  return (
    <section
      id="ml-pipeline"
      className={`relative py-20 ${dark ? "bg-slate-950/40" : "bg-gradient-to-b from-transparent via-purple-50/30 to-transparent"}`}
    >
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        {/* Header */}
        <div className="text-center mb-12">
          <div
            className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium mb-4 ${
              dark
                ? "bg-blue-500/10 text-blue-300 border border-blue-500/20"
                : "bg-blue-100 text-blue-700"
            }`}
          >
            <iconify-icon icon="solar:code-square-bold" style={{ fontSize: "14px" }}></iconify-icon>
            Machine Learning Pipeline
          </div>
          <h2
            className={`text-3xl sm:text-4xl font-semibold tracking-tight ${
              dark ? "text-white" : "text-slate-900"
            }`}
          >
            Fine-tune your own <span className="gradient-text">Handwriting OCR</span>
          </h2>
          <p
            className={`mt-3 text-base max-w-2xl mx-auto ${
              dark ? "text-slate-400" : "text-slate-600"
            }`}
          >
            A complete, beginner-friendly Python pipeline to fine-tune TrOCR on cursive
            and mixed English handwriting — from dataset prep to a deployable FastAPI server.
          </p>
        </div>

        {/* Recommended models */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-12">
          {MODELS.map((m) => (
            <div
              key={m.name}
              className={`relative rounded-2xl p-5 border transition-all hover:-translate-y-1 ${cardBase}`}
            >
              {m.badge && (
                <span className="absolute top-3 right-3 text-[10px] px-2 py-0.5 rounded-full bg-gradient-to-r from-blue-600 to-purple-600 text-white font-medium">
                  {m.badge}
                </span>
              )}
              <div
                className={`w-11 h-11 rounded-xl bg-gradient-to-br ${m.color} flex items-center justify-center mb-3 shadow-lg`}
              >
                <iconify-icon icon={m.icon} style={{ color: "white", fontSize: "22px" }}></iconify-icon>
              </div>
              <div className={`font-semibold ${dark ? "text-white" : "text-slate-900"}`}>
                {m.name}
              </div>
              <div className={`text-xs mt-1 leading-relaxed ${dark ? "text-slate-400" : "text-slate-600"}`}>
                {m.desc}
              </div>
            </div>
          ))}
        </div>

        {/* Datasets */}
        <div className={`rounded-2xl border ${cardBase} backdrop-blur p-6 mb-10`}>
          <div className="flex items-center gap-2 mb-4">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center">
              <iconify-icon icon="solar:database-bold" style={{ color: "white", fontSize: "18px" }}></iconify-icon>
            </div>
            <h3 className={`text-lg font-semibold tracking-tight ${dark ? "text-white" : "text-slate-900"}`}>
              Recommended Datasets
            </h3>
          </div>
          <div className="grid sm:grid-cols-2 gap-3">
            {DATASETS.map((d) => (
              <a
                key={d.name}
                href={d.url}
                target="_blank"
                rel="noopener noreferrer"
                className={`group flex items-start gap-3 p-4 rounded-xl border transition-all hover:-translate-y-0.5 ${
                  dark
                    ? "bg-slate-950/40 border-slate-800 hover:border-emerald-500/40"
                    : "bg-slate-50/80 border-slate-200 hover:border-emerald-400/60"
                }`}
              >
                <div
                  className={`w-10 h-10 rounded-lg flex items-center justify-center shrink-0 ${
                    dark ? "bg-emerald-500/10" : "bg-emerald-100"
                  }`}
                >
                  <iconify-icon icon={d.icon} style={{ color: "#10b981", fontSize: "22px" }}></iconify-icon>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2">
                    <div className={`font-medium ${dark ? "text-white" : "text-slate-900"}`}>
                      {d.name}
                    </div>
                    <span
                      className={`text-[10px] px-2 py-0.5 rounded-full ${
                        dark ? "bg-slate-800 text-slate-400" : "bg-slate-200 text-slate-600"
                      }`}
                    >
                      {d.size}
                    </span>
                  </div>
                  <div className={`text-xs mt-1 ${dark ? "text-slate-400" : "text-slate-600"}`}>
                    {d.desc}
                  </div>
                  <div className="text-xs mt-2 inline-flex items-center gap-1 text-emerald-500 opacity-0 group-hover:opacity-100 transition-opacity">
                    Visit dataset
                    <iconify-icon icon="solar:arrow-right-up-linear" style={{ fontSize: "12px" }}></iconify-icon>
                  </div>
                </div>
              </a>
            ))}
          </div>
        </div>

        {/* Code Tabs */}
        <div className={`rounded-2xl border ${cardBase} backdrop-blur overflow-hidden`}>
          {/* Tab bar */}
          <div
            className={`flex overflow-x-auto scrollbar-thin border-b ${
              dark ? "border-slate-800 bg-slate-950/40" : "border-slate-200 bg-slate-50/60"
            }`}
          >
            {TABS.map((t) => {
              const active = activeTab === t.id;
              return (
                <button
                  key={t.id}
                  onClick={() => setActiveTab(t.id)}
                  className={`relative shrink-0 inline-flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
                    active
                      ? dark
                        ? "text-white"
                        : "text-slate-900"
                      : dark
                      ? "text-slate-400 hover:text-slate-200"
                      : "text-slate-500 hover:text-slate-800"
                  }`}
                >
                  <iconify-icon icon={t.icon} style={{ fontSize: "16px" }}></iconify-icon>
                  {t.label}
                  {active && (
                    <span className="absolute bottom-0 left-2 right-2 h-0.5 rounded-full bg-gradient-to-r from-blue-500 to-purple-500"></span>
                  )}
                </button>
              );
            })}
          </div>

          {/* Code body */}
          <div className="relative">
            <div
              className={`flex items-center justify-between px-4 py-2 text-xs border-b ${
                dark
                  ? "bg-slate-950/60 border-slate-800 text-slate-400"
                  : "bg-slate-100/80 border-slate-200 text-slate-500"
              }`}
            >
              <div className="flex items-center gap-2 font-mono">
                <span className="w-2.5 h-2.5 rounded-full bg-red-400"></span>
                <span className="w-2.5 h-2.5 rounded-full bg-yellow-400"></span>
                <span className="w-2.5 h-2.5 rounded-full bg-green-400"></span>
                <span className="ml-2">
                  {activeTab === "structure" ? "project tree" : `${activeTab}.py`}
                </span>
              </div>
              <button
                onClick={() => handleCopy(activeTab, CODE[activeTab])}
                className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${
                  dark
                    ? "bg-slate-800 hover:bg-slate-700 text-slate-200"
                    : "bg-white hover:bg-slate-200 text-slate-700 border border-slate-200"
                }`}
              >
                <iconify-icon
                  icon={copied === activeTab ? "solar:check-circle-bold" : "solar:copy-bold"}
                  style={{ fontSize: "14px" }}
                ></iconify-icon>
                {copied === activeTab ? "Copied!" : "Copy"}
              </button>
            </div>

            <pre className="code-block">
              <code>{CODE[activeTab]}</code>
            </pre>
          </div>
        </div>

        {/* Requirements + Run instructions */}
        <div className="grid md:grid-cols-2 gap-6 mt-10">
          <div className={`rounded-2xl border ${cardBase} backdrop-blur p-6`}>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-orange-500 to-rose-500 flex items-center justify-center">
                  <iconify-icon icon="solar:box-bold" style={{ color: "white", fontSize: "18px" }}></iconify-icon>
                </div>
                <h3 className={`text-base font-semibold tracking-tight ${dark ? "text-white" : "text-slate-900"}`}>
                  requirements.txt
                </h3>
              </div>
              <button
                onClick={() => handleCopy("req", REQUIREMENTS)}
                className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${
                  dark ? "bg-slate-800 hover:bg-slate-700 text-slate-200" : "bg-slate-100 hover:bg-slate-200 text-slate-700"
                }`}
              >
                <iconify-icon
                  icon={copied === "req" ? "solar:check-circle-bold" : "solar:copy-bold"}
                  style={{ fontSize: "14px" }}
                ></iconify-icon>
                {copied === "req" ? "Copied!" : "Copy"}
              </button>
            </div>
            <pre className="code-block code-block--sm">
              <code>{REQUIREMENTS}</code>
            </pre>
          </div>

          <div className={`rounded-2xl border ${cardBase} backdrop-blur p-6`}>
            <div className="flex items-center gap-2 mb-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                <iconify-icon icon="solar:play-circle-bold" style={{ color: "white", fontSize: "18px" }}></iconify-icon>
              </div>
              <h3 className={`text-base font-semibold tracking-tight ${dark ? "text-white" : "text-slate-900"}`}>
                Run the pipeline
              </h3>
            </div>
            <ol className={`space-y-3 text-sm ${dark ? "text-slate-300" : "text-slate-700"}`}>
              {[
                { t: "Install deps", c: "pip install -r requirements.txt" },
                { t: "Prepare CSVs", c: "Build data/train.csv, val.csv, test.csv with image_path,text" },
                { t: "Train", c: "python src/train.py" },
                { t: "Evaluate", c: "python src/evaluate.py" },
                { t: "Serve API", c: "uvicorn api.main:app --reload --port 8000" },
              ].map((step, i) => (
                <li key={i} className="flex gap-3">
                  <span
                    className={`shrink-0 w-6 h-6 rounded-md text-xs font-semibold flex items-center justify-center bg-gradient-to-br from-blue-600 to-purple-600 text-white`}
                  >
                    {i + 1}
                  </span>
                  <div className="min-w-0">
                    <div className="font-medium">{step.t}</div>
                    <code
                      className={`mt-0.5 inline-block text-xs px-2 py-0.5 rounded font-mono break-all ${
                        dark ? "bg-slate-950 text-purple-300" : "bg-slate-100 text-purple-700"
                      }`}
                    >
                      {step.c}
                    </code>
                  </div>
                </li>
              ))}
            </ol>
          </div>
        </div>

        {/* Metrics info */}
        <div
          className={`mt-10 rounded-2xl p-6 border ${
            dark
              ? "bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-pink-500/10 border-purple-500/20"
              : "bg-gradient-to-r from-blue-50 via-purple-50 to-pink-50 border-purple-200"
          }`}
        >
          <div className="flex items-start gap-4">
            <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shrink-0">
              <iconify-icon icon="solar:chart-2-bold" style={{ color: "white", fontSize: "22px" }}></iconify-icon>
            </div>
            <div>
              <h4 className={`font-semibold tracking-tight ${dark ? "text-white" : "text-slate-900"}`}>
                How accuracy is measured
              </h4>
              <p className={`text-sm mt-1 ${dark ? "text-slate-300" : "text-slate-700"}`}>
                We track <strong>CER</strong> (Character Error Rate) and <strong>WER</strong> (Word Error
                Rate) on the validation set every epoch. The model with the lowest CER is saved to{" "}
                <code className={`px-1.5 py-0.5 rounded font-mono text-xs ${dark ? "bg-slate-900 text-purple-300" : "bg-white text-purple-700"}`}>
                  checkpoints/best_model
                </code>
                . A CER of 0.05 means roughly 95% of characters are predicted correctly.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}