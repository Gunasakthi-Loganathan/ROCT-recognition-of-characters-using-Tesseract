import test from "node:test";
import assert from "node:assert/strict";
import { createApp } from "./server.js";

function listen(app) {
  return new Promise((resolve) => {
    const server = app.listen(0, () => resolve(server));
  });
}

async function request(server, path, options = {}) {
  const address = server.address();
  const response = await fetch(`http://127.0.0.1:${address.port}${path}`, options);
  let body = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  return { status: response.status, body };
}

test("health endpoint reports missing Gemini configuration without crashing", async () => {
  const originalKey = process.env.GEMINI_API_KEY;
  delete process.env.GEMINI_API_KEY;
  const server = await listen(createApp());
  try {
    const response = await request(server, "/");
    assert.equal(response.status, 200);
    assert.equal(response.body.geminiConfigured, false);
  } finally {
    server.close();
    if (originalKey) process.env.GEMINI_API_KEY = originalKey;
  }
});

test("correction endpoint validates empty text", async () => {
  const server = await listen(createApp());
  try {
    const response = await request(server, "/api/correct", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "   " }),
    });
    assert.equal(response.status, 400);
  } finally {
    server.close();
  }
});

test("correction endpoint returns explicit 503 when Gemini key is missing", async () => {
  const originalKey = process.env.GEMINI_API_KEY;
  delete process.env.GEMINI_API_KEY;
  const server = await listen(createApp());
  try {
    const response = await request(server, "/api/correct", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "hello" }),
    });
    assert.equal(response.status, 503);
    assert.match(response.body.message, /GEMINI_API_KEY/);
  } finally {
    server.close();
    if (originalKey) process.env.GEMINI_API_KEY = originalKey;
  }
});
