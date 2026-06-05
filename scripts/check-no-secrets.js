import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";

const ROOT = process.cwd();
const SKIP_DIRS = new Set([".git", "node_modules", "dist", "coverage", ".venv", "venv", "__pycache__"]);
const SKIP_FILES = new Set(["package-lock.json", ".env.example"]);
const SECRET_FILE_PATTERNS = [/^\.env(?:\.|$)/i, /id_rsa/i, /id_dsa/i, /\.pem$/i, /\.p12$/i, /\.pfx$/i];
const SECRET_PATTERNS = [
  { name: "private key", pattern: /-----BEGIN [A-Z ]*PRIVATE KEY-----/ },
  { name: "generic secret assignment", pattern: /(?:api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"]{12,}['\"]/i },
  { name: "AWS access key", pattern: /AKIA[0-9A-Z]{16}/ },
  { name: "Google API key", pattern: /AIza[0-9A-Za-z_-]{35}/ },
  { name: "GitHub token", pattern: /gh[pousr]_[0-9A-Za-z_]{36,}/ },
];

const findings = [];

function scanPath(path) {
  const stat = statSync(path);
  const rel = relative(ROOT, path) || ".";
  if (stat.isDirectory()) {
    const name = rel.split(/[\\/]/).pop();
    if (SKIP_DIRS.has(name)) return;
    for (const entry of readdirSync(path)) scanPath(join(path, entry));
    return;
  }
  if (!stat.isFile()) return;
  const fileName = rel.split(/[\\/]/).pop();
  if (SKIP_FILES.has(fileName)) return;
  if (SECRET_FILE_PATTERNS.some((pattern) => pattern.test(fileName))) {
    findings.push(`${rel}: credential-like file must not be committed`);
    return;
  }
  if (stat.size > 1024 * 1024) return;
  const content = readFileSync(path, "utf8");
  SECRET_PATTERNS.forEach(({ name, pattern }) => {
    if (pattern.test(content)) findings.push(`${rel}: potential ${name}`);
  });
}

scanPath(ROOT);

if (findings.length) {
  console.error("Secret scan failed:");
  findings.forEach((finding) => console.error(`- ${finding}`));
  process.exit(1);
}

console.log("No committed credentials detected.");
