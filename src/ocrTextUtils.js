export const CUSTOM_DICTIONARY = [
  "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with", "this", "these", "those", "you", "your", "we", "our", "they", "their", "i", "am", "can", "could", "should", "would", "may", "might", "must", "do", "does", "did", "done", "have", "had", "not", "no", "yes", "if", "else", "then", "than", "when", "where", "why", "how", "what", "which", "who", "whom", "because", "therefore", "hence", "thus", "example", "important", "definition", "explain", "describe", "machine", "learning", "artificial", "intelligence", "algorithm", "dataset", "data", "model", "training", "testing", "test", "validation", "classification", "regression", "clustering", "neural", "network", "deep", "supervised", "unsupervised", "reinforcement", "feature", "label", "accuracy", "precision", "recall", "error", "loss", "function", "gradient", "descent", "python", "java", "react", "javascript", "html", "css", "database", "compiler", "encryption", "decryption", "security", "cloud", "server", "client", "computer", "science", "engineering", "digital", "image", "text", "ocr", "recognition", "processing", "system", "input", "output",
];

export const COMMON_OCR_REPLACEMENTS = [
  [/\bteh\b/gi, "the"],
  [/\bths\b/gi, "this"],
  [/\bthls\b/gi, "this"],
  [/\bwlth\b/gi, "with"],
  [/\bmachne\b/gi, "machine"],
  [/\bleaming\b/gi, "learning"],
  [/\bmodle\b/gi, "model"],
  [/\balgorthm\b/gi, "algorithm"],
  [/\balgoritm\b/gi, "algorithm"],
  [/\bdatasett\b/gi, "dataset"],
  [/\bdatabse\b/gi, "database"],
  [/\brecieve\b/gi, "receive"],
  [/\bseperate\b/gi, "separate"],
  [/\bdefinaton\b/gi, "definition"],
  [/\bclasification\b/gi, "classification"],
  [/\bneuraI\b/g, "neural"],
  [/\bintelligance\b/gi, "intelligence"],
];

export function levenshteinDistance(a, b) {
  const matrix = Array.from({ length: a.length + 1 }, () =>
    Array(b.length + 1).fill(0)
  );
  for (let i = 0; i <= a.length; i++) matrix[i][0] = i;
  for (let j = 0; j <= b.length; j++) matrix[0][j] = j;
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      matrix[i][j] = Math.min(
        matrix[i - 1][j] + 1,
        matrix[i][j - 1] + 1,
        matrix[i - 1][j - 1] + cost
      );
    }
  }
  return matrix[a.length][b.length];
}

export function correctSingleWord(word) {
  if (!word || word.length <= 2) return word;
  const originalWord = word;
  const lower = word.toLowerCase();
  const cleaned = lower.replace(/[^a-z]/g, "");
  if (!cleaned || cleaned.length <= 2) return originalWord;
  if (CUSTOM_DICTIONARY.includes(cleaned)) return originalWord;
  let bestMatch = cleaned;
  let bestDistance = Infinity;
  for (const dictWord of CUSTOM_DICTIONARY) {
    if (Math.abs(dictWord.length - cleaned.length) > 2) continue;
    const distance = levenshteinDistance(cleaned, dictWord);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestMatch = dictWord;
    }
  }
  const maxAllowedDistance = cleaned.length <= 5 ? 1 : 2;
  if (bestDistance <= maxAllowedDistance) {
    if (originalWord[0] === originalWord[0].toUpperCase()) {
      return bestMatch.charAt(0).toUpperCase() + bestMatch.slice(1);
    }
    return bestMatch;
  }
  return originalWord;
}

export function autoCorrectText(text) {
  if (!text.trim()) return "";
  let corrected = text;
  for (const [pattern, replacement] of COMMON_OCR_REPLACEMENTS) {
    corrected = corrected.replace(pattern, replacement);
  }
  corrected = corrected
    .replace(/0/g, "o")
    .replace(/(?<=\b)[1|](?=[a-z])/gi, "l")
    .replace(/vv/g, "w")
    .replace(/rn/g, "m");
  corrected = corrected
    .split(/(\s+)/)
    .map((token) => {
      if (/^\s+$/.test(token)) return token;
      const punctuationStart = token.match(/^[^a-zA-Z]+/)?.[0] || "";
      const punctuationEnd = token.match(/[^a-zA-Z]+$/)?.[0] || "";
      const word = token
        .replace(/^[^a-zA-Z]+/, "")
        .replace(/[^a-zA-Z]+$/, "");
      if (!word) return token;
      const fixed = correctSingleWord(word);
      return punctuationStart + fixed + punctuationEnd;
    })
    .join("");
  corrected = corrected
    .replace(/\bi am\b/gi, "I am")
    .replace(/\bi have\b/gi, "I have")
    .replace(/\bi will\b/gi, "I will")
    .replace(/\bi can\b/gi, "I can");
  return corrected;
}
