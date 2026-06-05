import test, { afterEach } from "node:test";
import assert from "node:assert/strict";
import { recognizeWithMlBackend } from "../mlOcrClient.js";

const originalFetch = globalThis.fetch;
const originalFormData = globalThis.FormData;

class FakeFormData {
  constructor() {
    this.values = [];
  }

  append(key, value, filename) {
    this.values.push({ key, value, filename });
  }
}

afterEach(() => {
  globalThis.fetch = originalFetch;
  globalThis.FormData = originalFormData;
});

test("recognizeWithMlBackend returns normalized backend OCR result", async () => {
  globalThis.FormData = FakeFormData;
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({
      text: " A ",
      confidence: 0.91,
      engine: "ml",
      model: "fixture-model",
      latency_ms: 12.5,
    }),
  });

  const result = await recognizeWithMlBackend({ name: "sample.png" });

  assert.deepEqual(result, {
    text: "A",
    confidence: 0.91,
    engine: "ml",
    model: "fixture-model",
    latencyMs: 12.5,
    raw: {
      text: " A ",
      confidence: 0.91,
      engine: "ml",
      model: "fixture-model",
      latency_ms: 12.5,
    },
  });
});

test("recognizeWithMlBackend surfaces backend errors", async () => {
  globalThis.FormData = FakeFormData;
  globalThis.fetch = async () => ({
    ok: false,
    status: 503,
    json: async () => ({ message: "model is not configured" }),
  });

  await assert.rejects(
    () => recognizeWithMlBackend({ name: "sample.png" }),
    /model is not configured/
  );
});
