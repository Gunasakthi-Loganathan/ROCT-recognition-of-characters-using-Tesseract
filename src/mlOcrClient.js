const ML_API_BASE_URL =
  import.meta.env?.VITE_ML_API_BASE_URL || "http://localhost:9000";

const ML_RECOGNIZE_ENDPOINT = `${ML_API_BASE_URL}/api/ocr`;
const DEFAULT_TIMEOUT_MS = 30000;

export async function recognizeWithMlBackend(file, { engine = "ml", timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  if (!file) {
    throw new Error("No image file provided for ML OCR.");
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  const body = new FormData();
  body.append("file", file, file.name || "upload");
  body.append("engine", engine);

  try {
    const response = await fetch(ML_RECOGNIZE_ENDPOINT, {
      method: "POST",
      body,
      signal: controller.signal,
    });

    let data = null;
    try {
      data = await response.json();
    } catch {
      data = null;
    }

    if (!response.ok) {
      throw new Error(
        data?.message || data?.error || `ML OCR backend returned HTTP ${response.status}`
      );
    }

    return {
      text: String(data?.text || "").trim(),
      confidence: Number.isFinite(Number(data?.confidence))
        ? Number(data.confidence)
        : null,
      engine: data?.engine || "ml",
      model: data?.model || "unknown-ml-model",
      latencyMs: Number.isFinite(Number(data?.latency_ms))
        ? Number(data.latency_ms)
        : null,
      raw: data,
    };
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new Error("ML OCR request timed out. Try Tesseract or a smaller image.");
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}
