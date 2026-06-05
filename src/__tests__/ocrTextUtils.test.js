import test from "node:test";
import assert from "node:assert/strict";
import { autoCorrectText, correctSingleWord, levenshteinDistance } from "../ocrTextUtils.js";

test("levenshteinDistance computes edit distance", () => {
  assert.equal(levenshteinDistance("kitten", "sitting"), 3);
});

test("autoCorrectText corrects common OCR mistakes", () => {
  assert.equal(autoCorrectText("teh machne leaming modle"), "the machine learning model");
});

test("OCR word correction preserves punctuation and capitalization", () => {
  assert.equal(correctSingleWord("Algoritm"), "Algorithm");
  assert.equal(autoCorrectText("i can test."), "I can test.");
});
