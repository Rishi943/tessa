// Muse Spark 1.3 (contributor) on the fine-tuning test TOIs, same prompt (v5), effort (low) and anonymised
// checks as base Glimmer's test dossiers: the closed model's row in finetune/evaluate.py.
// Run: node finetune/spark_test.js [--jobs 12]   then   python -m finetune.evaluate --models base=data/finetune/traces spark=data/finetune/traces_spark
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { anonymize } from "../agent/anonymize.js";
import { inventedNumbers, parseDossier, systemPrompt, writeDossier } from "../agent/dossier.js";

const MODEL = "muse-spark-1.3-contributor";
const VERSION = "v5";
const args = process.argv.slice(2);
const at = args.indexOf("--jobs");
const jobs = at >= 0 ? Number(args[at + 1]) : 12;
const vetDir = new URL("../data/finetune/vet/", import.meta.url);
const outDir = new URL("../data/finetune/traces_spark/", import.meta.url);
mkdirSync(outDir, { recursive: true });
const prompt = systemPrompt(VERSION);

const todo = [];
for (const f of readdirSync(vetDir).filter((f) => f.endsWith(".json"))) {
  const d = JSON.parse(readFileSync(new URL(f, vetDir), "utf8"));
  if (d.checks.code_verdict !== "agent" || d.label.split !== "test") continue;
  const path = new URL(`${d.toi}.0.json`, outDir);
  if (!existsSync(path)) todo.push({ d, path });
}
console.log(`${todo.length} Spark dossiers to write`);

let next = 0, done = 0, failed = 0, cost = 0;
async function worker() {
  while (next < todo.length) {
    const { d, path } = todo[next++];
    try {
      const checks = anonymize(d.checks);
      const run = await writeDossier(checks, MODEL, VERSION, "low", "eval (fine-tune test)");
      const out = { toi: d.toi, sample: 0, split: d.label.split, target: d.label.target, ...run };
      try {
        out.dossier = parseDossier(run.raw);
        out.verdict = out.dossier.verdict;
        out.invented = inventedNumbers(out.dossier, checks, prompt);
      } catch (e) {
        out.verdict = "unparseable";
        out.error = e.message;
      }
      writeFileSync(path, JSON.stringify(out, null, 1));
      cost += run.cost;
    } catch (e) {
      failed++;
      console.log(`${d.toi}: ${e.message.slice(0, 160)}`);
    }
    if (++done % 50 === 0) console.log(`${done}/${todo.length}, ${failed} failed, $${cost.toFixed(3)}`);
  }
}
await Promise.all(Array.from({ length: jobs }, worker));
console.log(`done: ${done}, ${failed} failed, $${cost.toFixed(3)}`);
