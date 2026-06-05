import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { GoogleGenAI } from "@google/genai";

function loadEnv() {
  dotenv.config({ quiet: true });
}

loadEnv();

export const PORT = process.env.PORT || 8000;
export const MAX_TEXT_LENGTH = Number(process.env.MAX_CORRECTION_TEXT_LENGTH || 8000);
export const GEMINI_TIMEOUT_MS = Number(process.env.GEMINI_TIMEOUT_MS || 15000);

const allowedOrigins = [
  "http://localhost:5173",
  "http://127.0.0.1:5173",
  process.env.CLIENT_ORIGIN,
].filter(Boolean);

export const GEMINI_MODELS = [
  "gemini-2.5-flash-lite",
  "gemini-2.0-flash",
  "gemini-2.5-flash",
];

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function withTimeout(promise, ms) {
  let timeoutId;
  const timeout = new Promise((_, reject) => {
    timeoutId = setTimeout(() => reject(new Error("Gemini request timed out")), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timeoutId));
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
    message.includes("rate limit") ||
    message.includes("timed out")
  );
}

function getGeminiClient() {
  if (!process.env.GEMINI_API_KEY) {
    return null;
  }
  return new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
}

async function generateWithRetry(prompt, ai = getGeminiClient()) {
  if (!ai) {
    const error = new Error("Gemini correction is not configured. Set GEMINI_API_KEY on the backend.");
    error.status = 503;
    throw error;
  }

  let lastError = null;
  for (const model of GEMINI_MODELS) {
    for (let attempt = 1; attempt <= 3; attempt++) {
      try {
        const response = await withTimeout(
          ai.models.generateContent({ model, contents: prompt }),
          GEMINI_TIMEOUT_MS
        );
        const correctedText = response.text?.trim();
        if (correctedText) {
          return { correctedText, model };
        }
        throw new Error("Gemini returned empty text");
      } catch (error) {
        lastError = error;
        console.error(`Gemini correction failed for ${model} attempt ${attempt}: ${error?.message || "unknown error"}`);
        if (!isRetryableGeminiError(error)) {
          throw error;
        }
        await sleep(1000 * attempt);
      }
    }
  }
  throw lastError || new Error("All Gemini models failed");
}

function buildCorrectionPrompt(text) {
  return `You are an OCR auto-correction assistant.

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
${text}`;
}

export function createApp() {
  const app = express();

  app.use(
    cors({
      origin(origin, callback) {
        if (!origin || allowedOrigins.includes(origin)) {
          return callback(null, true);
        }
        return callback(new Error("Not allowed by CORS"));
      },
    })
  );

  app.use(express.json({ limit: "2mb" }));

  app.get("/", (req, res) => {
    res.json({
      status: "ok",
      message: "Gemini OCR correction backend is running",
      geminiConfigured: Boolean(process.env.GEMINI_API_KEY),
    });
  });

  app.post("/api/correct", async (req, res) => {
    try {
      const text = typeof req.body?.text === "string" ? req.body.text.trim() : "";
      if (!text) {
        return res.status(400).json({ error: "No text provided" });
      }
      if (text.length > MAX_TEXT_LENGTH) {
        return res.status(413).json({ error: `Text is too large. Maximum length is ${MAX_TEXT_LENGTH} characters.` });
      }

      const result = await generateWithRetry(buildCorrectionPrompt(text));
      return res.json({ correctedText: result.correctedText, model: result.model, engine: "gemini" });
    } catch (error) {
      const status = Number(error?.status || 500);
      const safeStatus = status >= 400 && status < 600 ? status : 500;
      console.error(`Final Gemini correction error: ${error?.message || "unknown error"}`);
      return res.status(safeStatus).json({
        error: "Gemini correction failed",
        message: error?.message || "Gemini is busy or unavailable. Please try again later.",
      });
    }
  });

  app.use((err, req, res, next) => {
    if (res.headersSent) return next(err);
    console.error(`Request rejected: ${err?.message || "unknown error"}`);
    return res.status(403).json({ error: "Request is not allowed by server policy" });
  });

  return app;
}

export function startServer(port = PORT) {
  const app = createApp();
  return app.listen(port, () => {
    if (!process.env.GEMINI_API_KEY) {
      console.warn("Warning: GEMINI_API_KEY is missing. Local browser OCR still works, but Gemini correction is disabled.");
    }
    console.log(`Gemini backend running at http://localhost:${port}`);
  });
}

if (process.argv[1] && import.meta.url === new URL(process.argv[1], "file:").href) {
  startServer();
}

export { buildCorrectionPrompt, generateWithRetry, isRetryableGeminiError };
