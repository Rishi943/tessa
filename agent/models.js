// Meta Model API models for pi (OpenAI-compatible chat completions).
// Prices are USD per million tokens, from dev.meta.ai/docs/pricing-rate-limits (2026-10-27).
// "-contributor" runs the same model far cheaper in exchange for Meta training on the traffic.

const BASE_URL = "https://api.meta.ai/v1";

const COMPAT = {
  supportsStore: false,
  supportsDeveloperRole: false,
  supportsReasoningEffort: false,
};

function meta(id, name, input, output, cacheRead) {
  return {
    id,
    name,
    api: "openai-completions",
    provider: "meta",
    baseUrl: BASE_URL,
    reasoning: false,
    input: ["text"],
    cost: { input, output, cacheRead, cacheWrite: 0 },
    contextWindow: 128000,
    maxTokens: 16384,
    compat: COMPAT,
  };
}

try {
  process.loadEnvFile(new URL("../.env", import.meta.url));
} catch {
  // no .env: rely on the environment
}

// Muse Glimmer 30B, open weights, served by serve/glimmer.py on Modal. It's paid by GPU time, not
// tokens, so the token prices are zero; reasoning strength goes in the system prompt (glimmerPrompt).
function glimmer() {
  return {
    id: "muse-glimmer",
    name: "Muse Glimmer 30B",
    api: "openai-completions",
    provider: "glimmer",
    baseUrl: `${(process.env.GLIMMER_URL ?? "").replace(/\/$/, "")}/v1`,
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 32768,
    maxTokens: 16384,
    compat: COMPAT,
  };
}

// Muse Spark Open, Meta's 816B (42B active) open-weight MoE, on Baseten. The hackathon key only works with
// this exact id. Prices from baseten.co/pricing (2026-10-29); the $100 hackathon credit pays for it.
function sparkOpen() {
  return {
    id: "meta-models/muse-spark-open-1026",
    name: "Muse Spark Open",
    api: "openai-completions",
    provider: "baseten",
    baseUrl: "https://inference.baseten.co/v1",
    reasoning: false,
    input: ["text"],
    cost: { input: 1.25, output: 4.25, cacheRead: 0.15, cacheWrite: 0 },
    contextWindow: 128000,
    maxTokens: 16384,
    compat: COMPAT,
  };
}

const KEYS = { glimmer: "GLIMMER_API_KEY", baseten: "BASETEN_API_KEY", meta: "META_API_KEY" };

export function glimmerPrompt(prompt, effort) {
  return effort ? `${prompt}\n\nReasoning strength: ${effort}.` : prompt;
}

export const MODELS = {
  "muse-glimmer": glimmer(),
  "muse-spark-open": sparkOpen(),
  "muse-spark-1.3": meta("muse-spark-1.3", "Muse Spark 1.3", 1.25, 4.25, 0.15),
  "muse-spark-1.3-contributor": meta("muse-spark-1.3-contributor", "Muse Spark 1.3 (contributor)", 0.1, 0.2, 0.002),
  "muse-spark-1.2": meta("muse-spark-1.2", "Muse Spark 1.2", 1.25, 4.25, 0.15),
  "muse-spark-1.2-contributor": meta("muse-spark-1.2-contributor", "Muse Spark 1.2 (contributor)", 0.1, 0.2, 0.002),
  "muse-spark-1.1": meta("muse-spark-1.1", "Muse Spark 1.1", 1.25, 4.25, 0.15),
  "muse-wonder-1.0": meta("muse-wonder-1.0", "Muse Wonder 1.0", 4.0, 20.0, 0.2),
};

export function apiKey(id) {
  const name = KEYS[MODELS[id]?.provider] ?? "META_API_KEY";
  const key = process.env[name];
  if (!key) throw new Error(`${name} is not set`);
  return key;
}
