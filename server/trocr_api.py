from contextlib import asynccontextmanager
from io import BytesIO

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


MODEL_ID = "microsoft/trocr-base-handwritten"
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading {MODEL_ID} on {DEVICE}...")

    app.state.processor = TrOCRProcessor.from_pretrained(MODEL_ID)
    app.state.model = VisionEncoderDecoderModel.from_pretrained(
        MODEL_ID,
        use_safetensors=True,
    )

    app.state.model.to(DEVICE)
    app.state.model.eval()

    print("TrOCR model loaded successfully.")
    yield


app = FastAPI(
    title="ROCT TrOCR API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE.type,
    }


@app.post("/api/trocr")
async def recognize_handwriting(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, PNG and WEBP images are supported.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Maximum supported file size is 10 MB.",
        )

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    processor = app.state.processor
    model = app.state.model

    pixel_values = processor(
        images=image,
        return_tensors="pt",
    ).pixel_values.to(DEVICE)

    with torch.inference_mode():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=128,
            num_beams=4,
            early_stopping=True,
        )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()

    return {
        "text": text,
        "model": MODEL_ID,
        "device": DEVICE.type,
    }
