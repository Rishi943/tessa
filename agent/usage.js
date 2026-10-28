// One line per model call (model, purpose, tokens), so the demo can show how much of Tessa's model work
// runs on the open-weight model. serve/usage.py sums it, with the site's Ask Tessa calls (D1) and Modal's
// GPU hours.
import { appendFileSync, mkdirSync } from "node:fs";

const dir = new URL("../data/usage/", import.meta.url);

export function logUsage(model, purpose, usage, ms) {
  mkdirSync(dir, { recursive: true });
  appendFileSync(new URL("model_calls.jsonl", dir), JSON.stringify({
    at: new Date().toISOString(), model, purpose, input: usage?.input ?? 0, output: usage?.output ?? 0, ms: ms ?? null,
  }) + "\n");
}
