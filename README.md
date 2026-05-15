# Image to Text OCR AI

An AI-powered OCR web app with a React + Vite frontend and a FastAPI backend.

## OCR engines

1. **Tesseract.js** — browser-only OCR. No backend required.
2. **TrOCR Handwritten** — Python backend required.
3. **GOT-OCR 2.0** — Python backend required.

## Project structure

```text
.
├── backend/
│   ├── main.py
│   └── requirements.txt
├── src/
│   ├── App.jsx
│   ├── main.jsx
│   ├── index.css
│   └── components/
│       └── MlPipeline.jsx
├── index.html
├── package.json
└── vite.config.js
```

## Run frontend

```bash
npm install
npm run dev
```

Open the Vite URL, usually:

```text
http://localhost:5173
```

## Run backend

Open a second terminal:

```bash
cd backend
python -m venv .venv
```

Activate the environment:

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies and start FastAPI:

```bash
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Check backend:

```text
http://localhost:8000/
```

## Notes

- Tesseract.js works directly in the browser.
- GOT-OCR 2.0 and TrOCR need the backend running at `localhost:8000`.
- The first backend request may take time because Hugging Face models need to download.
- GOT-OCR and TrOCR can require high RAM/VRAM. Use Tesseract.js for low-end systems.
