const API_BASE_URL =
  import.meta.env?.VITE_API_BASE_URL || "http://localhost:8000";

const GEMINI_CORRECT_ENDPOINT = `${API_BASE_URL}/api/correct`;

export async function geminiAutoCorrectText(text) {
  const cleanText = text.trim();

  if (!cleanText) {
    return {
      correctedText: "",
      model: "gemini-2.5-flash",
    };
  }

  const response = await fetch(GEMINI_CORRECT_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text: cleanText,
    }),
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
  };
}