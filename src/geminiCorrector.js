const API_BASE_URL =
  import.meta.env?.VITE_API_BASE_URL || "http://localhost:8000";

const GEMINI_CORRECT_ENDPOINT = `${API_BASE_URL}/api/correct`;
const DEFAULT_TIMEOUT_MS = 15000;

export async function geminiAutoCorrectText(text, { timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  const cleanText = text.trim();

  if (!cleanText) {
    return {
      correctedText: "",
      model: "gemini-2.5-flash",
      engine: "local-empty-input",
    };
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(GEMINI_CORRECT_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text: cleanText,
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      let message = `Gemini backend returned HTTP ${response.status}`;

      try {
        const data = await response.json();
        message = data.message || data.error || message;
      } catch {
        // ignore JSON parse error
      }

      throw new Error(message);
    }

    const data = await response.json();

    return {
      correctedText: data.correctedText || cleanText,
      model: data.model || "gemini-2.5-flash",
      engine: data.engine || "gemini",
    };
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new Error("Gemini correction timed out. Local OCR text is still available.");
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}
