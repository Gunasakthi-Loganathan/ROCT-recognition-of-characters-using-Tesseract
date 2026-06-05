const TROCR_API_BASE_URL =
  import.meta.env.VITE_TROCR_API_URL || "http://localhost:8001";

export async function recognizeWithTrocr(file) {
  if (!(file instanceof File || file instanceof Blob)) {
    throw new TypeError("A valid image file is required.");
  }

  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${TROCR_API_BASE_URL}/api/trocr`, {
    method: "POST",
    body: formData,
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(
      data.detail ||
        data.error ||
        `TrOCR server returned HTTP ${response.status}`
    );
  }

  return {
    text: String(data.text || "").trim(),
    model: data.model || "microsoft/trocr-base-handwritten",
    device: data.device || "unknown",
  };
}
