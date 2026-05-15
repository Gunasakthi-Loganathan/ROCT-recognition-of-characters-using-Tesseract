import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { GoogleGenAI } from "@google/genai";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 8000;

const allowedOrigins = [
  "http://localhost:5173",
  "http://127.0.0.1:5173",
  process.env.CLIENT_ORIGIN,
].filter(Boolean);

app.use(
  cors({
    origin: function (origin, callback) {
      if (!origin || allowedOrigins.includes(origin)) {
        return callback(null, true);
      }

      return callback(new Error("Not allowed by CORS"));
    },
  })
);

app.use(express.json({ limit: "2mb" }));

if (!process.env.GEMINI_API_KEY) {
  console.warn("Warning: GEMINI_API_KEY is missing in .env file");
}

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
});

const GEMINI_MODELS = [
  "gemini-2.5-flash-lite",
  "gemini-2.0-flash",
  "gemini-2.5-flash",
];

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRetryableGeminiError(error) {
  const message = String(error?.message || "").toLowerCase();
  const status = error?.status;

  return (
    status === 429 ||
    status === 500 ||
    status === 502 ||
    status === 503 ||
    status === 504 ||
    message.includes("unavailable") ||
    message.includes("high demand") ||
    message.includes("overloaded") ||
    message.includes("rate limit")
  );
}

async function generateWithRetry(prompt) {
  let lastError = null;

  for (const model of GEMINI_MODELS) {
    for (let attempt = 1; attempt <= 3; attempt++) {
      try {
        console.log(`Trying ${model}, attempt ${attempt}...`);

        const response = await ai.models.generateContent({
          model,
          contents: prompt,
        });

        const correctedText = response.text?.trim();

        if (correctedText) {
          return {
            correctedText,
            model,
          };
        }

        throw new Error("Gemini returned empty text");
      } catch (error) {
        lastError = error;

        console.error(
          `Gemini error with ${model}, attempt ${attempt}:`,
          error?.message || error
        );

        if (!isRetryableGeminiError(error)) {
          throw error;
        }

        await sleep(1000 * attempt);
      }
    }
  }

  throw lastError || new Error("All Gemini models failed");
}

app.get("/", (req, res) => {
  res.json({
    status: "ok",
    message: "Gemini OCR correction backend is running",
  });
});

app.post("/api/correct", async (req, res) => {
  try {
    const { text } = req.body;

    if (!text || !text.trim()) {
      return res.status(400).json({
        error: "No text provided",
      });
    }

    const prompt = `
You are an OCR auto-correction assistant.

Task:
Correct OCR mistakes in the given text.

Rules:
- Correct spelling mistakes.
- Correct OCR character mistakes such as 0/o, 1/l/I, rn/m.
- Improve spacing and punctuation.
- Preserve the original meaning.
- Do not add new information.
- Do not explain anything.
- Return only the corrected text.

OCR text:
${text}
`;

    const result = await generateWithRetry(prompt);

    res.json({
      correctedText: result.correctedText,
      model: result.model,
    });
  } catch (error) {
    console.error("Final Gemini correction error:", error);

    res.status(500).json({
      error: "Gemini correction failed",
      message:
        error?.message ||
        "Gemini is busy or unavailable. Please try again later.",
    });
  }
});

app.listen(PORT, () => {
  console.log(`Gemini backend running at http://localhost:${PORT}`);
});