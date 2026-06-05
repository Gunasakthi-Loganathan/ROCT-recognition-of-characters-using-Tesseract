import test, { afterEach } from "node:test";
import assert from "node:assert/strict";
import { geminiAutoCorrectText } from "../geminiCorrector.js";

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

test("geminiAutoCorrectText does not call backend for blank text", async () => {
  let called = false;
  globalThis.fetch = async () => {
    called = true;
  };
  assert.deepEqual(await geminiAutoCorrectText("   "), {
    correctedText: "",
    model: "gemini-2.5-flash",
  });
  assert.equal(called, false);
});

test("geminiAutoCorrectText returns backend result", async () => {
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({ correctedText: "Hello world", model: "gemini-test" }),
  });
  assert.deepEqual(await geminiAutoCorrectText(" hello "), {
    correctedText: "Hello world",
    model: "gemini-test",
  });
});

test("geminiAutoCorrectText throws backend error messages", async () => {
  globalThis.fetch = async () => ({
    ok: false,
    status: 500,
    json: async () => ({ message: "backend failed" }),
  });
  await assert.rejects(() => geminiAutoCorrectText("hello"), /backend failed/);
});
