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

export const MODELS = {
  "muse-spark-1.3": meta("muse-spark-1.3", "Muse Spark 1.3", 1.25, 4.25, 0.15),
  "muse-spark-1.3-contributor": meta("muse-spark-1.3-contributor", "Muse Spark 1.3 (contributor)", 0.1, 0.2, 0.002),
  "muse-spark-1.2": meta("muse-spark-1.2", "Muse Spark 1.2", 1.25, 4.25, 0.15),
  "muse-spark-1.2-contributor": meta("muse-spark-1.2-contributor", "Muse Spark 1.2 (contributor)", 0.1, 0.2, 0.002),
  "muse-spark-1.1": meta("muse-spark-1.1", "Muse Spark 1.1", 1.25, 4.25, 0.15),
  "muse-wonder-1.0": meta("muse-wonder-1.0", "Muse Wonder 1.0", 4.0, 20.0, 0.2),
};

export function apiKey() {
  try {
    process.loadEnvFile(new URL("../.env", import.meta.url));
  } catch {
    // no .env: rely on the environment
  }
  const key = process.env.META_API_KEY;
  if (!key) throw new Error("META_API_KEY is not set");
  return key;
}
