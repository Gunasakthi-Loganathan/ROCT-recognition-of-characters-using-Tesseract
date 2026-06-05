import React, { useState, useEffect, useRef, useCallback } from "react";
import { geminiAutoCorrectText } from "./geminiCorrector";
import { autoCorrectText } from "./ocrTextUtils";

// ---------- Tesseract.js Loader ----------
const TESSERACT_CDN =
  "https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js";

function loadTesseract() {
  return new Promise((resolve, reject) => {
    if (typeof window === "undefined") return reject(new Error("No window"));

    if (window.Tesseract) return resolve(window.Tesseract);

    const existing = document.querySelector(`script[src="${TESSERACT_CDN}"]`);

    if (existing) {
      existing.addEventListener("load", () => resolve(window.Tesseract));
      existing.addEventListener("error", () =>
        reject(new Error("Failed to load Tesseract"))
      );
      return;
    }

    const script = document.createElement("script");
    script.src = TESSERACT_CDN;
    script.async = true;
    script.onload = () => resolve(window.Tesseract);
    script.onerror = () => reject(new Error("Failed to load Tesseract"));
    document.head.appendChild(script);
  });
}

// ---------- OCR Engine ----------
const OCR_ENGINES = {
  tesseract: {
    label: "Tesseract.js",
    description: "Browser-based OCR with Gemini AI auto-correction.",
    icon: "solar:scanner-bold",
    backend: false,
  },
};

// ---------- OCR Modes (used by Tesseract) ----------
const OCR_MODES = {
  auto: {
    label: "Auto (Best Accuracy)",
    psm: "6",
    description: "Tries multiple modes and picks the best confidence",
  },
  printed: {
    label: "Printed Text",
    psm: "6",
    description: "Best for clear printed paragraphs",
  },
  handwritten: {
    label: "Handwritten Text",
    psm: "6",
    description: "Best for normal handwriting",
  },
  cursive: {
    label: "Cursive Text",
    psm: "11",
    description: "Best for joined or irregular handwriting",
  },
  singleLine: {
    label: "Single Line",
    psm: "7",
    description: "Best for one line of text",
  },
  sparse: {
    label: "Sparse Text",
    psm: "11",
    description: "Best when text is scattered",
  },
  rawLine: {
    label: "Raw Line",
    psm: "13",
    description: "Best for difficult single-line OCR",
  },
};

const SOFT_MODES = new Set(["cursive", "handwritten", "auto"]);


function useDarkMode() {
  const [dark, setDark] = useState(() => {
    if (typeof window === "undefined") return false;
    const saved = localStorage.getItem("ocr-theme");
    if (saved) return saved === "dark";
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  });

  useEffect(() => {
    localStorage.setItem("ocr-theme", dark ? "dark" : "light");
  }, [dark]);

  return [dark, setDark];
}

function preprocessImage(file, mode = "printed") {
  return new Promise((resolve, reject) => {
    const image = new Image();
    const url = URL.createObjectURL(file);
    image.onload = () => {
      try {
        URL.revokeObjectURL(url);
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        const scaleFactor = Math.max(1, Math.min(2.5, 1800 / image.width));
        canvas.width = image.width * scaleFactor;
        canvas.height = image.height * scaleFactor;
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        let totalBrightness = 0;
        let minV = 255;
        let maxV = 0;
        for (let i = 0; i < data.length; i += 4) {
          const gray =
            0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
          totalBrightness += gray;
          if (gray < minV) minV = gray;
          if (gray > maxV) maxV = gray;
          data[i] = gray;
          data[i + 1] = gray;
          data[i + 2] = gray;
        }
        const avgBrightness = totalBrightness / (data.length / 4);
        const isSoftMode = SOFT_MODES.has(mode);
        if (isSoftMode) {
          const contrast = 1.25;
          const range = Math.max(1, maxV - minV);
          for (let i = 0; i < data.length; i += 4) {
            let v = data[i];
            v = ((v - minV) / range) * 255;
            v = (v - 128) * contrast + 128;
            v = Math.max(0, Math.min(255, v));
            data[i] = v;
            data[i + 1] = v;
            data[i + 2] = v;
            data[i + 3] = 255;
          }
        } else {
          const contrast = 1.65;
          const threshold = avgBrightness;
          for (let i = 0; i < data.length; i += 4) {
            let value = data[i];
            value = (value - 128) * contrast + 128;
            value = value > threshold ? 255 : 0;
            data[i] = value;
            data[i + 1] = value;
            data[i + 2] = value;
            data[i + 3] = 255;
          }
        }
        ctx.putImageData(imageData, 0, 0);
        canvas.toBlob(
          (blob) => {
            if (!blob) {
              reject(new Error("Failed to preprocess image"));
              return;
            }
            resolve(blob);
          },
          "image/png",
          1
        );
      } catch (error) {
        reject(error);
      }
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Failed to load image"));
    };
    image.src = url;
  });
}

function cropImageFile(file, rectRatio) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    const url = URL.createObjectURL(file);
    image.onload = () => {
      try {
        URL.revokeObjectURL(url);
        const sx = Math.max(0, Math.round(rectRatio.x * image.width));
        const sy = Math.max(0, Math.round(rectRatio.y * image.height));
        const sw = Math.max(1, Math.round(rectRatio.w * image.width));
        const sh = Math.max(1, Math.round(rectRatio.h * image.height));
        const canvas = document.createElement("canvas");
        canvas.width = sw;
        canvas.height = sh;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(image, sx, sy, sw, sh, 0, 0, sw, sh);
        canvas.toBlob(
          (blob) => {
            if (!blob) return reject(new Error("Failed to crop"));
            const croppedFile = new File(
              [blob],
              (file.name || "image") + "-cropped.png",
              { type: "image/png" }
            );
            resolve(croppedFile);
          },
          "image/png",
          1
        );
      } catch (e) {
        reject(e);
      }
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Failed to load image"));
    };
    image.src = url;
  });
}

// ---------- Navbar ----------
function Navbar({ dark, setDark }) {
  return (
    <nav
      className={`sticky top-0 z-50 ${
        dark
          ? "bg-slate-950/70 border-slate-800"
          : "bg-white/70 border-slate-200"
      } border-b backdrop-blur transition-colors`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-purple-500/30">
            <iconify-icon
              icon="solar:scanner-2-bold"
              style={{ color: "white", fontSize: "20px" }}
            ></iconify-icon>
          </div>
          <div>
            <div
              className={`font-semibold tracking-tight text-base ${
                dark ? "text-white" : "text-slate-900"
              }`}
            >
              Image to Text <span className="text-purple-500">OCR AI</span>
            </div>
            <div
              className={`text-xs ${
                dark ? "text-slate-400" : "text-slate-500"
              }`}
            >
              Powered by Machine Learning
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <a
            href="#how-it-works"
            className={`hidden sm:inline-flex text-sm font-medium px-3 py-2 rounded-lg transition-colors ${
              dark
                ? "text-slate-300 hover:text-white hover:bg-slate-800"
                : "text-slate-600 hover:text-slate-900 hover:bg-slate-100"
            }`}
          >
            How it works
          </a>

          <button
            onClick={() => setDark(!dark)}
            aria-label="Toggle dark mode"
            className={`w-10 h-10 rounded-lg flex items-center justify-center transition-all ${
              dark
                ? "bg-slate-800 hover:bg-slate-700 text-yellow-300"
                : "bg-slate-100 hover:bg-slate-200 text-slate-700"
            }`}
          >
            <iconify-icon
              icon={dark ? "solar:sun-bold" : "solar:moon-bold"}
              style={{ fontSize: "20px" }}
            ></iconify-icon>
          </button>
        </div>
      </div>
    </nav>
  );
}

function Hero({ dark }) {
  return (
    <section className="relative overflow-hidden pt-16 pb-12 sm:pt-20 sm:pb-16">
      <div className="absolute inset-0 -z-10 overflow-hidden">
        <div
          className={`absolute top-10 -left-20 w-96 h-96 rounded-full blur-3xl opacity-30 ${
            dark ? "bg-blue-600" : "bg-blue-300"
          }`}
        ></div>
        <div
          className={`absolute -bottom-10 -right-20 w-96 h-96 rounded-full blur-3xl opacity-30 ${
            dark ? "bg-purple-600" : "bg-purple-300"
          }`}
        ></div>
      </div>

      <div className="max-w-4xl mx-auto px-4 sm:px-6 text-center">
        <div
          className={`inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-medium mb-6 ${
            dark
              ? "bg-slate-800/80 text-slate-300 border border-slate-700"
              : "bg-white/80 text-slate-700 border border-slate-200"
          } backdrop-blur`}
        >
          <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></span>
          AI-Powered OCR · Tesseract.js · Gemini Correction
        </div>

        <h1
          className={`text-4xl sm:text-5xl lg:text-6xl font-semibold tracking-tight mb-6 ${
            dark ? "text-white" : "text-slate-900"
          }`}
        >
          Convert Written Text from <br className="hidden sm:block" />
          <span className="bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">
            Images to Digital Text
          </span>
        </h1>

        <p
          className={`text-base sm:text-lg max-w-2xl mx-auto leading-relaxed ${
            dark ? "text-slate-400" : "text-slate-600"
          }`}
        >
          Upload an image, run Tesseract.js OCR in the browser, and improve the
          extracted text with Gemini AI correction.
        </p>
      </div>
    </section>
  );
}

// ---------- Crop Overlay (unchanged) ----------
function CropOverlay({ dark, imageUrl, onCancel, onApply }) {
  const containerRef = useRef(null);
  const imgRef = useRef(null);
  const [imgRect, setImgRect] = useState({ w: 0, h: 0 });
  const [rect, setRect] = useState(null);
  const [drag, setDrag] = useState(null);

  const handleImgLoad = () => {
    if (imgRef.current) {
      setImgRect({
        w: imgRef.current.clientWidth,
        h: imgRef.current.clientHeight,
      });
      const w = imgRef.current.clientWidth * 0.8;
      const h = imgRef.current.clientHeight * 0.6;
      setRect({
        x: (imgRef.current.clientWidth - w) / 2,
        y: (imgRef.current.clientHeight - h) / 2,
        w,
        h,
      });
    }
  };

  const getPos = (e) => {
    const bounds = imgRef.current.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: Math.max(0, Math.min(bounds.width, clientX - bounds.left)),
      y: Math.max(0, Math.min(bounds.height, clientY - bounds.top)),
    };
  };

  const onMouseDown = (e) => {
    e.preventDefault();
    const p = getPos(e);
    setDrag({ startX: p.x, startY: p.y });
    setRect({ x: p.x, y: p.y, w: 0, h: 0 });
  };

  const onMouseMove = (e) => {
    if (!drag) return;
    const p = getPos(e);
    const x = Math.min(drag.startX, p.x);
    const y = Math.min(drag.startY, p.y);
    const w = Math.abs(p.x - drag.startX);
    const h = Math.abs(p.y - drag.startY);
    setRect({ x, y, w, h });
  };

  const onMouseUp = () => setDrag(null);

  const handleApply = () => {
    if (!rect || rect.w < 5 || rect.h < 5 || !imgRect.w || !imgRect.h) {
      onCancel();
      return;
    }
    onApply({
      x: rect.x / imgRect.w,
      y: rect.y / imgRect.h,
      w: rect.w / imgRect.w,
      h: rect.h / imgRect.h,
    });
  };

  return (
    <div className="fixed inset-0 z-[60] bg-slate-950/80 backdrop-blur-md flex items-center justify-center p-4">
      <div
        className={`w-full max-w-4xl rounded-2xl border ${
          dark ? "bg-slate-900 border-slate-800" : "bg-white border-slate-200"
        } shadow-2xl overflow-hidden`}
      >
        <div
          className={`px-5 py-3 flex items-center justify-between border-b ${
            dark ? "border-slate-800" : "border-slate-200"
          }`}
        >
          <div className="flex items-center gap-2">
            <iconify-icon
              icon="solar:crop-bold"
              style={{ color: "#a855f7", fontSize: "20px" }}
            ></iconify-icon>
            <h3
              className={`font-semibold tracking-tight ${
                dark ? "text-white" : "text-slate-900"
              }`}
            >
              Crop Text Region
            </h3>
          </div>
          <button
            onClick={onCancel}
            className={`w-8 h-8 rounded-lg flex items-center justify-center ${
              dark
                ? "hover:bg-slate-800 text-slate-300"
                : "hover:bg-slate-100 text-slate-600"
            }`}
            aria-label="Close"
          >
            <iconify-icon
              icon="solar:close-circle-bold"
              style={{ fontSize: "20px" }}
            ></iconify-icon>
          </button>
        </div>

        <div
          ref={containerRef}
          className="relative bg-slate-950 flex items-center justify-center select-none"
          style={{ maxHeight: "70vh" }}
        >
          <div
            className="relative"
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onMouseUp={onMouseUp}
            onMouseLeave={onMouseUp}
            onTouchStart={onMouseDown}
            onTouchMove={onMouseMove}
            onTouchEnd={onMouseUp}
            style={{ cursor: "crosshair" }}
          >
            <img
              ref={imgRef}
              src={imageUrl}
              alt="To crop"
              onLoad={handleImgLoad}
              className="block max-w-full max-h-[70vh] pointer-events-none"
              draggable={false}
            />
            {rect && (
              <>
                <div
                  className="absolute bg-slate-950/60 pointer-events-none"
                  style={{ left: 0, top: 0, right: 0, height: rect.y }}
                />
                <div
                  className="absolute bg-slate-950/60 pointer-events-none"
                  style={{
                    left: 0,
                    top: rect.y + rect.h,
                    right: 0,
                    bottom: 0,
                  }}
                />
                <div
                  className="absolute bg-slate-950/60 pointer-events-none"
                  style={{ left: 0, top: rect.y, width: rect.x, height: rect.h }}
                />
                <div
                  className="absolute bg-slate-950/60 pointer-events-none"
                  style={{
                    left: rect.x + rect.w,
                    top: rect.y,
                    right: 0,
                    height: rect.h,
                  }}
                />
                <div
                  className="absolute border-2 border-purple-400 shadow-[0_0_0_1px_rgba(168,85,247,0.4)] pointer-events-none"
                  style={{
                    left: rect.x,
                    top: rect.y,
                    width: rect.w,
                    height: rect.h,
                  }}
                />
              </>
            )}
          </div>
        </div>

        <div
          className={`px-5 py-4 flex flex-col sm:flex-row items-center justify-between gap-3 border-t ${
            dark ? "border-slate-800" : "border-slate-200"
          }`}
        >
          <p
            className={`text-xs ${
              dark ? "text-slate-400" : "text-slate-500"
            }`}
          >
            Drag on the image to draw a selection around the text region.
          </p>
          <div className="flex gap-2">
            <button
              onClick={onCancel}
              className={`px-4 py-2 rounded-xl text-sm font-medium transition-all ${
                dark
                  ? "bg-slate-800 text-slate-200 hover:bg-slate-700"
                  : "bg-slate-100 text-slate-700 hover:bg-slate-200"
              }`}
            >
              Cancel
            </button>
            <button
              onClick={handleApply}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:shadow-lg hover:shadow-purple-500/30 transition-all"
            >
              <iconify-icon
                icon="solar:crop-bold"
                style={{ fontSize: "16px" }}
              ></iconify-icon>
              Apply Crop
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------- OCR Workspace ----------
function OcrWorkspace({ dark }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [processedPreviewUrl, setProcessedPreviewUrl] = useState(null);

  const [originalText, setOriginalText] = useState("");
  const [correctedText, setCorrectedText] = useState("");

  const [ocrEngine, setOcrEngine] = useState("tesseract");
  const [ocrMode, setOcrMode] = useState("auto");
  const [usePreprocessing, setUsePreprocessing] = useState(true);

  const [averageConfidence, setAverageConfidence] = useState(null);
  const [lowConfidenceWords, setLowConfidenceWords] = useState([]);
  const [bestPsmUsed, setBestPsmUsed] = useState(null);
  const [modelUsed, setModelUsed] = useState(null);

  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState("");

  const [error, setError] = useState("");
  const [toast, setToast] = useState("");
  const [isDragging, setIsDragging] = useState(false);

  const [showCrop, setShowCrop] = useState(false);

  const fileInputRef = useRef(null);

  const cardBase = dark
    ? "bg-slate-900/60 border-slate-800"
    : "bg-white/80 border-slate-200";

  const showToast = (msg) => {
    setToast(msg);
    setTimeout(() => setToast(""), 2500);
  };

  const resetTextData = () => {
    setOriginalText("");
    setCorrectedText("");
    setAverageConfidence(null);
    setLowConfidenceWords([]);
    setBestPsmUsed(null);
    setModelUsed(null);
  };

  const handleFile = (selectedFile) => {
    setError("");
    resetTextData();

    if (!selectedFile) return;

    const validTypes = ["image/jpeg", "image/jpg", "image/png", "image/webp"];

    if (!validTypes.includes(selectedFile.type)) {
      setError("Invalid file type. Please upload JPG, JPEG, PNG, or WEBP.");
      return;
    }

    if (selectedFile.size > 10 * 1024 * 1024) {
      setError("File is too large. Maximum size is 10MB.");
      return;
    }

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    if (processedPreviewUrl) URL.revokeObjectURL(processedPreviewUrl);

    const url = URL.createObjectURL(selectedFile);

    setFile(selectedFile);
    setPreviewUrl(url);
    setProcessedPreviewUrl(null);
  };

  const onInputChange = (e) => handleFile(e.target.files?.[0]);

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const onDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    handleFile(e.dataTransfer.files?.[0]);
  }, []);

  const runOcrPass = async (Tesseract, imageForOcr, psm, onProgress) => {
    const { data } = await Tesseract.recognize(imageForOcr, "eng", {
      logger: (m) => {
        if (m.status === "recognizing text" && onProgress) {
          onProgress(Math.round(m.progress * 100));
        }
      },
      tessedit_pageseg_mode: psm,
      preserve_interword_spaces: "1",
    });

    const text = (data?.text || "").trim();
    const words = (data?.words || []).filter((w) => w.text?.trim());
    const avg =
      words.length > 0
        ? words.reduce((s, w) => s + Number(w.confidence || 0), 0) /
          words.length
        : 0;

    return { text, words, avg, psm };
  };

  const handleConvert = async () => {
    if (!file) {
      setError("Please upload an image first.");
      return;
    }

    setError("");
    resetTextData();
    setIsProcessing(true);
    setProgress(0);
    setStatusMessage("");

    try {
      // ---------- Tesseract.js (browser) ----------
      setStatusMessage("Loading Tesseract.js…");
      const Tesseract = await loadTesseract();

      let imageForOcr = file;

      if (usePreprocessing) {
        setStatusMessage("Preprocessing image…");
        const processedBlob = await preprocessImage(file, ocrMode);
        imageForOcr = processedBlob;
        if (processedPreviewUrl) URL.revokeObjectURL(processedPreviewUrl);
        setProcessedPreviewUrl(URL.createObjectURL(processedBlob));
      } else {
        if (processedPreviewUrl) URL.revokeObjectURL(processedPreviewUrl);
        setProcessedPreviewUrl(null);
      }

      setStatusMessage("Running Tesseract OCR…");

      let best;
      if (ocrMode === "auto") {
        const psms = ["6", "11", "13"];
        const results = [];
        for (let i = 0; i < psms.length; i++) {
          const psm = psms[i];
          const passResult = await runOcrPass(
            Tesseract,
            imageForOcr,
            psm,
            (p) => {
              const base = (i / psms.length) * 100;
              const slice = p / psms.length;
              setProgress(Math.min(100, Math.round(base + slice)));
            }
          );
          results.push(passResult);
        }
        const valid = results.filter((r) => r.text.length > 0);
        const pool = valid.length > 0 ? valid : results;
        best = pool.reduce((a, b) => (b.avg > a.avg ? b : a), pool[0]);
        setBestPsmUsed(best.psm);
      } else {
        const selectedMode = OCR_MODES[ocrMode];
        best = await runOcrPass(
          Tesseract,
          imageForOcr,
          selectedMode.psm,
          (p) => setProgress(p)
        );
        setBestPsmUsed(best.psm);
      }

      const text = best.text;
      if (!text) {
        setError("No readable text found. Try a clearer image or crop region.");
        return;
      }

      const words = best.words;
      if (words.length > 0) {
        const avg = best.avg;
        setAverageConfidence(Number.isFinite(avg) ? Math.round(avg) : null);
        const lowWords = words
          .filter((word) => Number(word.confidence || 0) < 75)
          .map((word) => ({
            text: word.text,
            confidence: Math.round(Number(word.confidence || 0)),
          }))
          .slice(0, 15);
        setLowConfidenceWords(lowWords);
      }

      setOriginalText(text);

      setStatusMessage("Applying rule-based correction…");
      const ruleCorrected = autoCorrectText(text);

      setStatusMessage("Correcting text with Gemini AI…");
      setProgress(95);

      try {
        const geminiResult = await geminiAutoCorrectText(ruleCorrected);

        setCorrectedText(geminiResult.correctedText);
        setModelUsed(`tesseract.js + ${geminiResult.model}`);
        showToast("Text extracted and corrected with Gemini!");
      } catch (geminiError) {
        console.error(geminiError);

        setCorrectedText(ruleCorrected);
        setModelUsed("tesseract.js + rule correction");
        setError("Gemini correction failed. Rule-based correction was applied instead.");
      }
    } catch (err) {
      console.error(err);
      setError(err.message || "Something went wrong while processing.");
    } finally {
      setIsProcessing(false);
      setProgress(0);
      setStatusMessage("");
    }
  };

  const handleAutoCorrect = async () => {
    if (!originalText.trim()) {
      setError("No OCR text available to correct.");
      return;
    }

    try {
      setError("");
      setIsProcessing(true);
      setProgress(50);
      setStatusMessage("Applying rule-based correction…");

      const ruleCorrected = autoCorrectText(originalText);

      setStatusMessage("Correcting text with Gemini AI…");
      setProgress(95);

      const geminiResult = await geminiAutoCorrectText(ruleCorrected);

      setCorrectedText(geminiResult.correctedText);
      setModelUsed(`tesseract.js + ${geminiResult.model}`);
      showToast("Gemini auto-correction applied!");
    } catch (err) {
      console.error(err);

      setCorrectedText(autoCorrectText(originalText));
      setModelUsed("tesseract.js + rule correction");
      setError("Gemini correction failed. Rule-based correction was applied instead.");
    } finally {
      setIsProcessing(false);
      setProgress(0);
      setStatusMessage("");
    }
  };

  const handleCopy = async (text) => {
    if (!text.trim()) return;
    try {
      await navigator.clipboard.writeText(text);
      showToast("Copied to clipboard!");
    } catch {
      showToast("Failed to copy.");
    }
  };

  const handleDownload = (text, filename) => {
    if (!text.trim()) return;
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    showToast("Downloaded as TXT!");
  };

  const handleClear = () => {
    setFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    if (processedPreviewUrl) URL.revokeObjectURL(processedPreviewUrl);
    setPreviewUrl(null);
    setProcessedPreviewUrl(null);
    resetTextData();
    setError("");
    setProgress(0);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleApplyCrop = async (rectRatio) => {
    if (!file) {
      setShowCrop(false);
      return;
    }
    try {
      const cropped = await cropImageFile(file, rectRatio);
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      if (processedPreviewUrl) URL.revokeObjectURL(processedPreviewUrl);
      setFile(cropped);
      setPreviewUrl(URL.createObjectURL(cropped));
      setProcessedPreviewUrl(null);
      resetTextData();
      setShowCrop(false);
      showToast("Cropped! Now click Convert to OCR the region.");
    } catch (e) {
      console.error(e);
      setError("Failed to crop image.");
      setShowCrop(false);
    }
  };

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      if (processedPreviewUrl) URL.revokeObjectURL(processedPreviewUrl);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const showPreprocessedBadge = usePreprocessing && !!processedPreviewUrl;
  const isLowConfidence =
    averageConfidence !== null && averageConfidence < 60;

  const isTesseract = ocrEngine === "tesseract";

  return (
    <section className="max-w-7xl mx-auto px-4 sm:px-6 pb-16">
      {toast && (
        <div className="fixed top-20 left-1/2 -translate-x-1/2 z-50">
          <div className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-blue-600 to-purple-600 text-white text-sm font-medium shadow-xl shadow-purple-500/30 flex items-center gap-2">
            <iconify-icon
              icon="solar:check-circle-bold"
              style={{ fontSize: "18px" }}
            ></iconify-icon>
            {toast}
          </div>
        </div>
      )}

      {showCrop && previewUrl && (
        <CropOverlay
          dark={dark}
          imageUrl={previewUrl}
          onCancel={() => setShowCrop(false)}
          onApply={handleApplyCrop}
        />
      )}

      <div className="grid lg:grid-cols-2 gap-6">
        {/* ---------- LEFT: Upload ---------- */}
        <div
          className={`rounded-2xl border ${cardBase} backdrop-blur p-6 shadow-xl shadow-blue-500/5`}
        >
          <div className="flex items-center gap-2 mb-4">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center">
              <iconify-icon
                icon="solar:gallery-add-bold"
                style={{ color: "white", fontSize: "18px" }}
              ></iconify-icon>
            </div>

            <h2
              className={`text-lg font-semibold tracking-tight ${
                dark ? "text-white" : "text-slate-900"
              }`}
            >
              Upload Image
            </h2>
          </div>

          {/* OCR Engine selector */}
          <div className="mb-4">
            <label
              className={`text-xs font-medium ${
                dark ? "text-slate-300" : "text-slate-700"
              }`}
            >
              OCR Engine
            </label>
            <div className="mt-1 grid sm:grid-cols-1 gap-2">
              {Object.entries(OCR_ENGINES).map(([key, eng]) => {
                const active = ocrEngine === key;
                return (
                  <button
                    key={key}
                    onClick={() => setOcrEngine(key)}
                    disabled={isProcessing}
                    className={`flex items-center gap-2 px-3 py-2.5 rounded-xl border text-left transition-all disabled:opacity-50 ${
                      active
                        ? "bg-gradient-to-br from-blue-600 to-purple-600 border-transparent text-white shadow-lg shadow-purple-500/30"
                        : dark
                        ? "bg-slate-950/40 border-slate-800 text-slate-300 hover:border-purple-500/40"
                        : "bg-white border-slate-200 text-slate-700 hover:border-purple-400/60"
                    }`}
                  >
                    <iconify-icon
                      icon={eng.icon}
                      style={{ fontSize: "18px" }}
                    ></iconify-icon>
                    <span className="text-xs font-medium leading-tight">
                      {eng.label}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {!previewUrl ? (
            <label
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              className={`relative flex flex-col items-center justify-center text-center rounded-2xl border-2 border-dashed cursor-pointer transition-all p-10 min-h-[240px] ${
                isDragging
                  ? "border-purple-500 bg-purple-500/10 scale-[1.01]"
                  : dark
                  ? "border-slate-700 hover:border-purple-500 hover:bg-slate-800/40"
                  : "border-slate-300 hover:border-purple-500 hover:bg-purple-50/50"
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/jpg,image/png,image/webp"
                onChange={onInputChange}
                className="hidden"
              />

              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500/20 to-purple-500/20 flex items-center justify-center mb-4">
                <iconify-icon
                  icon="solar:cloud-upload-bold"
                  style={{ color: "#8b5cf6", fontSize: "32px" }}
                ></iconify-icon>
              </div>

              <div
                className={`font-medium ${
                  dark ? "text-white" : "text-slate-900"
                }`}
              >
                Drop image here or click to browse
              </div>

              <div
                className={`text-xs mt-1 ${
                  dark ? "text-slate-400" : "text-slate-500"
                }`}
              >
                Supports JPG, JPEG, PNG, WEBP · Max 10MB
              </div>
            </label>
          ) : (
            <div className="space-y-4">
              <div
                className={`relative rounded-2xl overflow-hidden border ${
                  dark ? "border-slate-800" : "border-slate-200"
                } bg-slate-950/40`}
              >
                <img
                  src={processedPreviewUrl || previewUrl}
                  alt="Uploaded preview"
                  className="w-full max-h-[340px] object-contain"
                />

                {showPreprocessedBadge && isTesseract && (
                  <div className="absolute top-3 left-3 px-3 py-1 rounded-full bg-purple-600 text-white text-xs font-medium">
                    Preprocessed Preview
                  </div>
                )}

                <button
                  onClick={() => setShowCrop(true)}
                  disabled={isProcessing}
                  className="absolute top-3 right-3 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-slate-950/70 text-white text-xs font-medium backdrop-blur hover:bg-slate-950/90 transition-colors disabled:opacity-50"
                >
                  <iconify-icon
                    icon="solar:crop-bold"
                    style={{ fontSize: "14px" }}
                  ></iconify-icon>
                  Crop
                </button>

                {isProcessing && (
                  <div className="absolute inset-0 bg-slate-950/70 backdrop-blur-sm flex flex-col items-center justify-center gap-3">
                    <div className="relative w-16 h-16">
                      <div className="absolute inset-0 rounded-full border-4 border-purple-500/30"></div>
                      <div className="absolute inset-0 rounded-full border-4 border-t-purple-500 animate-spin"></div>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <iconify-icon
                          icon="solar:magic-stick-3-bold"
                          style={{ color: "#a855f7", fontSize: "22px" }}
                        ></iconify-icon>
                      </div>
                    </div>

                    <div className="text-white text-sm font-medium text-center px-4">
                      {statusMessage || "AI scanning text…"}{" "}
                      {progress > 0 && `${progress}%`}
                    </div>

                    {progress > 0 && (
                      <div className="w-48 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all"
                          style={{ width: `${progress}%` }}
                        ></div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Tesseract-only mode controls */}
              {isTesseract && (
                <div className="grid sm:grid-cols-2 gap-3">
                  <div>
                    <label
                      className={`text-xs font-medium ${
                        dark ? "text-slate-300" : "text-slate-700"
                      }`}
                    >
                      OCR Mode
                    </label>

                    <select
                      value={ocrMode}
                      onChange={(e) => setOcrMode(e.target.value)}
                      disabled={isProcessing}
                      className={`mt-1 w-full rounded-xl px-3 py-2.5 text-sm border outline-none ${
                        dark
                          ? "bg-slate-950 border-slate-800 text-slate-100"
                          : "bg-white border-slate-200 text-slate-900"
                      }`}
                    >
                      {Object.entries(OCR_MODES).map(([key, mode]) => (
                        <option key={key} value={key}>
                          {mode.label}
                        </option>
                      ))}
                    </select>

                    <p
                      className={`text-xs mt-1 ${
                        dark ? "text-slate-500" : "text-slate-500"
                      }`}
                    >
                      {OCR_MODES[ocrMode].description}
                    </p>
                  </div>

                  <div>
                    <label
                      className={`text-xs font-medium ${
                        dark ? "text-slate-300" : "text-slate-700"
                      }`}
                    >
                      Preprocessing
                    </label>

                    <button
                      onClick={() => setUsePreprocessing(!usePreprocessing)}
                      disabled={isProcessing}
                      className={`mt-1 w-full rounded-xl px-3 py-2.5 text-sm font-medium border transition-all ${
                        usePreprocessing
                          ? "bg-purple-600 border-purple-600 text-white"
                          : dark
                          ? "bg-slate-950 border-slate-800 text-slate-300"
                          : "bg-white border-slate-200 text-slate-700"
                      }`}
                    >
                      {usePreprocessing ? "Enabled" : "Disabled"}
                    </button>

                    <p
                      className={`text-xs mt-1 ${
                        dark ? "text-slate-500" : "text-slate-500"
                      }`}
                    >
                      Soft mode for cursive · binarization for printed.
                    </p>
                  </div>
                </div>
              )}

              <div className="flex flex-wrap gap-2">
                <button
                  onClick={handleConvert}
                  disabled={isProcessing}
                  className="flex-1 min-w-[160px] inline-flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white text-sm font-medium hover:shadow-lg hover:shadow-purple-500/40 hover:scale-[1.02] transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
                >
                  <iconify-icon
                    icon="solar:magic-stick-3-bold"
                    style={{ fontSize: "18px" }}
                  ></iconify-icon>
                  {isProcessing
                    ? "Processing…"
                    : `Convert with ${OCR_ENGINES[ocrEngine].label}`}
                </button>

                <button
                  onClick={() => setShowCrop(true)}
                  disabled={isProcessing}
                  className={`inline-flex items-center justify-center gap-2 px-4 py-3 rounded-xl text-sm font-medium transition-all disabled:opacity-50 ${
                    dark
                      ? "bg-slate-800 text-slate-200 hover:bg-slate-700"
                      : "bg-slate-100 text-slate-700 hover:bg-slate-200"
                  }`}
                >
                  <iconify-icon
                    icon="solar:crop-bold"
                    style={{ fontSize: "16px" }}
                  ></iconify-icon>
                  Crop
                </button>

                <button
                  onClick={handleClear}
                  disabled={isProcessing}
                  className={`inline-flex items-center justify-center gap-2 px-4 py-3 rounded-xl text-sm font-medium transition-all disabled:opacity-50 ${
                    dark
                      ? "bg-slate-800 text-slate-200 hover:bg-slate-700"
                      : "bg-slate-100 text-slate-700 hover:bg-slate-200"
                  }`}
                >
                  <iconify-icon
                    icon="solar:trash-bin-trash-bold"
                    style={{ fontSize: "16px" }}
                  ></iconify-icon>
                  Clear
                </button>
              </div>
            </div>
          )}

          {error && (
            <div className="mt-4 flex items-start gap-2 px-4 py-3 rounded-xl bg-red-500/10 border border-red-500/30 text-red-500 text-sm">
              <iconify-icon
                icon="solar:danger-triangle-bold"
                style={{ fontSize: "18px", marginTop: "1px" }}
              ></iconify-icon>
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* ---------- RIGHT: Result ---------- */}
        <div
          className={`rounded-2xl border ${cardBase} backdrop-blur p-6 shadow-xl shadow-purple-500/5`}
        >
          <div className="flex items-center justify-between mb-4 gap-2">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                <iconify-icon
                  icon="solar:document-text-bold"
                  style={{ color: "white", fontSize: "18px" }}
                ></iconify-icon>
              </div>

              <h2
                className={`text-lg font-semibold tracking-tight ${
                  dark ? "text-white" : "text-slate-900"
                }`}
              >
                OCR Result
              </h2>
            </div>

            <div className="flex items-center gap-2 flex-wrap justify-end">
              {modelUsed && (
                <span
                  className={`text-xs px-2 py-1 rounded-full font-mono ${
                    dark
                      ? "bg-slate-800 text-slate-300"
                      : "bg-slate-100 text-slate-700"
                  }`}
                  title="Model used"
                >
                  {modelUsed.length > 28
                    ? modelUsed.slice(0, 28) + "…"
                    : modelUsed}
                </span>
              )}

              {bestPsmUsed && ocrMode === "auto" && isTesseract && (
                <span
                  className={`text-xs px-2 py-1 rounded-full ${
                    dark
                      ? "bg-slate-800 text-slate-300"
                      : "bg-slate-100 text-slate-700"
                  }`}
                >
                  PSM {bestPsmUsed}
                </span>
              )}

              {averageConfidence !== null && (
                <span
                  className={`text-xs px-2 py-1 rounded-full ${
                    averageConfidence >= 80
                      ? "bg-green-500/10 text-green-500"
                      : averageConfidence >= 60
                      ? "bg-yellow-500/10 text-yellow-500"
                      : "bg-red-500/10 text-red-500"
                  }`}
                >
                  Confidence: {averageConfidence}%
                </span>
              )}
            </div>
          </div>

          {isLowConfidence && (
            <div
              className={`mb-4 flex items-start gap-2 px-4 py-3 rounded-xl border text-sm ${
                dark
                  ? "bg-red-500/10 border-red-500/30 text-red-300"
                  : "bg-red-50 border-red-200 text-red-700"
              }`}
            >
              <iconify-icon
                icon="solar:danger-triangle-bold"
                style={{ fontSize: "18px", marginTop: "1px" }}
              ></iconify-icon>
              <div>
                <div className="font-medium">
                  Low confidence ({averageConfidence}%)
                </div>
                <div className="text-xs mt-0.5 opacity-90">
                  Try cropping tighter around the text, improving image clarity, or
                  changing the OCR mode.
                </div>
              </div>
            </div>
          )}

          <div className="grid gap-4">
            <div>
              <div className="flex items-center justify-between mb-2">
                <label
                  className={`text-sm font-medium ${
                    dark ? "text-slate-300" : "text-slate-700"
                  }`}
                >
                  Original OCR Text
                </label>

                {originalText && (
                  <span
                    className={`text-xs ${
                      dark ? "text-slate-500" : "text-slate-500"
                    }`}
                  >
                    {originalText.length} chars
                  </span>
                )}
              </div>

              <textarea
                value={originalText}
                readOnly
                placeholder={
                  isProcessing
                    ? statusMessage || "AI is reading your image…"
                    : "Original OCR text will appear here (read-only)."
                }
                className={`w-full h-[150px] rounded-xl p-4 text-sm resize-none border outline-none transition-all font-mono leading-relaxed ${
                  dark
                    ? "bg-slate-950/50 border-slate-800 text-slate-100 placeholder:text-slate-500"
                    : "bg-slate-50 border-slate-200 text-slate-900 placeholder:text-slate-400"
                }`}
              />
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <label
                  className={`text-sm font-medium ${
                    dark ? "text-slate-300" : "text-slate-700"
                  }`}
                >
                  Corrected Text
                </label>

                <button
                  onClick={handleAutoCorrect}
                  disabled={!originalText || isProcessing}
                  className="text-xs px-3 py-1.5 rounded-lg bg-purple-600 text-white font-medium disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  Gemini Correct Text
                </button>
              </div>

              <textarea
                value={correctedText}
                onChange={(e) => setCorrectedText(e.target.value)}
                placeholder="Corrected text will appear here. You can edit it manually."
                className={`w-full h-[150px] rounded-xl p-4 text-sm resize-none border outline-none transition-all font-mono leading-relaxed ${
                  dark
                    ? "bg-slate-950/50 border-slate-800 text-slate-100 placeholder:text-slate-500 focus:border-purple-500"
                    : "bg-slate-50 border-slate-200 text-slate-900 placeholder:text-slate-400 focus:border-purple-500 focus:bg-white"
                }`}
              />
            </div>
          </div>

          {lowConfidenceWords.length > 0 && (
            <div
              className={`mt-4 rounded-xl border p-3 ${
                dark
                  ? "border-yellow-500/20 bg-yellow-500/5"
                  : "border-yellow-300 bg-yellow-50"
              }`}
            >
              <div
                className={`text-xs font-semibold mb-2 ${
                  dark ? "text-yellow-300" : "text-yellow-700"
                }`}
              >
                Low-confidence words
              </div>

              <div className="flex flex-wrap gap-2">
                {lowConfidenceWords.map((word, index) => (
                  <span
                    key={`${word.text}-${index}`}
                    className={`text-xs px-2 py-1 rounded-full ${
                      dark
                        ? "bg-slate-800 text-slate-300"
                        : "bg-white text-slate-700 border border-slate-200"
                    }`}
                  >
                    {word.text} · {word.confidence}%
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="grid sm:grid-cols-2 gap-2 mt-4">
            <button
              onClick={() => handleCopy(correctedText || originalText)}
              disabled={!correctedText && !originalText}
              className={`inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed ${
                dark
                  ? "bg-slate-800 text-slate-200 hover:bg-slate-700"
                  : "bg-slate-100 text-slate-700 hover:bg-slate-200"
              }`}
            >
              <iconify-icon
                icon="solar:copy-bold"
                style={{ fontSize: "16px" }}
              ></iconify-icon>
              Copy Final Text
            </button>

            <button
              onClick={() =>
                handleDownload(
                  correctedText || originalText,
                  "corrected-ocr-text.txt"
                )
              }
              disabled={!correctedText && !originalText}
              className="inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:shadow-lg hover:shadow-purple-500/30 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <iconify-icon
                icon="solar:download-bold"
                style={{ fontSize: "16px" }}
              ></iconify-icon>
              Download Final Text
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}

function HowItWorks({ dark }) {
  const steps = [
    {
      icon: "solar:upload-square-bold",
      title: "Upload Image",
      desc: "Upload a JPG, PNG, or WEBP image containing printed, handwritten, or cursive text.",
      color: "from-blue-500 to-indigo-600",
    },
    {
      icon: "solar:cpu-bolt-bold",
      title: "Run OCR",
      desc: "Use Tesseract.js directly in the browser; no FastAPI backend is needed.",
      color: "from-indigo-500 to-purple-600",
    },
    {
      icon: "solar:document-text-bold",
      title: "Extract + Correct",
      desc: "Recognized text is auto-corrected with rules and a custom dictionary.",
      color: "from-purple-500 to-pink-600",
    },
  ];

  return (
    <section
      id="how-it-works"
      className={`py-20 ${dark ? "bg-slate-900/30" : "bg-white/40"}`}
    >
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        <div className="text-center mb-14">
          <div
            className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium mb-4 ${
              dark
                ? "bg-purple-500/10 text-purple-300 border border-purple-500/20"
                : "bg-purple-100 text-purple-700"
            }`}
          >
            <iconify-icon
              icon="solar:cpu-bolt-bold"
              style={{ fontSize: "14px" }}
            ></iconify-icon>
            How it works
          </div>

          <h2
            className={`text-3xl sm:text-4xl font-semibold tracking-tight ${
              dark ? "text-white" : "text-slate-900"
            }`}
          >
            Powerful OCR in{" "}
            <span className="bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">
              3 steps
            </span>
          </h2>

          <p
            className={`mt-3 text-base max-w-2xl mx-auto ${
              dark ? "text-slate-400" : "text-slate-600"
            }`}
          >
            A simple frontend-only pipeline: browser OCR, preprocessing, confidence
            scoring, and Gemini AI text correction.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {steps.map((step, index) => (
            <div
              key={step.title}
              className={`relative rounded-2xl p-6 border backdrop-blur transition-all hover:-translate-y-1 hover:shadow-2xl ${
                dark
                  ? "bg-slate-900/60 border-slate-800 hover:shadow-purple-500/10"
                  : "bg-white/80 border-slate-200 hover:shadow-purple-500/20"
              }`}
            >
              <div
                className={`absolute -top-3 -left-3 w-9 h-9 rounded-xl bg-gradient-to-br ${step.color} text-white flex items-center justify-center font-semibold text-sm shadow-lg`}
              >
                {index + 1}
              </div>

              <div
                className={`w-12 h-12 rounded-2xl mb-4 flex items-center justify-center bg-gradient-to-br ${step.color} shadow-lg`}
              >
                <iconify-icon
                  icon={step.icon}
                  style={{ color: "white", fontSize: "24px" }}
                ></iconify-icon>
              </div>

              <h3
                className={`text-lg font-semibold tracking-tight mb-2 ${
                  dark ? "text-white" : "text-slate-900"
                }`}
              >
                {step.title}
              </h3>

              <p
                className={`text-sm leading-relaxed ${
                  dark ? "text-slate-400" : "text-slate-600"
                }`}
              >
                {step.desc}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Features({ dark }) {
  const items = [
    {
      icon: "solar:settings-bold",
      title: "Browser OCR",
      desc: "Tesseract.js reads text directly in the browser.",
    },
    {
      icon: "solar:gallery-edit-bold",
      title: "Preprocessing",
      desc: "Soft pipeline for cursive · binarization for printed.",
    },
    {
      icon: "solar:chart-bold",
      title: "Confidence Score",
      desc: "Shows average OCR confidence and weak words.",
    },
    {
      icon: "solar:pen-new-square-bold",
      title: "Gemini Auto Correction",
      desc: "Uses Gemini AI plus OCR cleanup rules.",
    },
  ];

  return (
    <section className="max-w-6xl mx-auto px-4 sm:px-6 py-16">
      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {items.map((item) => (
          <div
            key={item.title}
            className={`rounded-2xl p-5 border transition-all hover:-translate-y-0.5 ${
              dark
                ? "bg-slate-900/60 border-slate-800 hover:border-purple-500/40"
                : "bg-white/80 border-slate-200 hover:border-purple-400/60"
            }`}
          >
            <div
              className={`w-10 h-10 rounded-xl flex items-center justify-center mb-3 ${
                dark ? "bg-purple-500/10" : "bg-purple-100"
              }`}
            >
              <iconify-icon
                icon={item.icon}
                style={{ color: "#8b5cf6", fontSize: "22px" }}
              ></iconify-icon>
            </div>

            <div
              className={`font-medium ${
                dark ? "text-white" : "text-slate-900"
              }`}
            >
              {item.title}
            </div>

            <div
              className={`text-xs mt-1 ${
                dark ? "text-slate-400" : "text-slate-500"
              }`}
            >
              {item.desc}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function Footer({ dark }) {
  return (
    <footer
      className={`border-t ${
        dark ? "border-slate-800 bg-slate-950" : "border-slate-200 bg-white/60"
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
            <iconify-icon
              icon="solar:scanner-2-bold"
              style={{ color: "white", fontSize: "16px" }}
            ></iconify-icon>
          </div>

          <span
            className={`text-sm font-medium ${
              dark ? "text-slate-300" : "text-slate-700"
            }`}
          >
            OCR AI Tool · Tesseract.js · Gemini AI Correction
          </span>
        </div>

        <div className={`text-xs ${dark ? "text-slate-500" : "text-slate-500"}`}>
          © {new Date().getFullYear()} Image to Text OCR AI
        </div>
      </div>
    </footer>
  );
}

export default function App() {
  const [dark, setDark] = useDarkMode();

  const bg = dark
    ? "bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-slate-100"
    : "bg-gradient-to-br from-blue-50 via-white to-purple-50 text-slate-900";

  return (
    <div className={`min-h-screen ${bg} transition-colors duration-300`}>
      <Navbar dark={dark} setDark={setDark} />
      <Hero dark={dark} />
      <OcrWorkspace dark={dark} />
      <HowItWorks dark={dark} />
      <Features dark={dark} />
      <Footer dark={dark} />
    </div>
  );
}
