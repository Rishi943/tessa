// Base Glimmer's dossiers on the fine-tuning TOIs, from the anonymised checks: several samples per star in
// train and val (the correct ones become reasoning traces to train on), one per star in test (the base
// model's score). Only TOIs the code rules pass to the agent, as in the survey. Existing files are kept.
// Run: node finetune/traces.js [--samples 4] [--jobs 48] [--splits train,val,test] [--limit N] [--hint]
//   [--export prompts.jsonl | --import FOLDER]   (run the prompts inside Modal: finetune/remote_traces.py)
//   --hint: one more dossier, at high strength, for each train/val TOI none of whose samples got the outcome
//   right, with the outcome in the system prompt. The file keeps the plain system prompt, which is what it
//   trains on; finetune/dataset.py drops hinted traces whose reasoning mentions the hint.
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { complete } from "@earendil-works/pi-ai";
import { anonymize } from "../agent/anonymize.js";
import { inventedNumbers, parseDossier, systemPrompt, userMessage } from "../agent/dossier.js";
import { MODELS, apiKey, glimmerPrompt } from "../agent/models.js";
import { logUsage } from "../agent/usage.js";

const args = process.argv.slice(2);
const opt = (name, fallback) => {
  const at = args.indexOf(name);
  return at >= 0 ? args.splice(at, 2)[1] : fallback;
};
const samples = Number(opt("--samples", "4"));
const jobs = Number(opt("--jobs", "48"));
const splits = opt("--splits", "train,val,test").split(",");
const limit = Number(opt("--limit", "0"));
const hintMode = args.includes("--hint");
const HINT_SAMPLE = 10;
const OUTCOME = {
  gate2: "Follow-up observations later confirmed this signal as a planet.",
  reject: "Follow-up observations later showed this signal is not a planet.",
  flag: "Follow-up observations left this signal ambiguous: neither confirmed nor ruled out.",
};
const hintFor = (target) => `${OUTCOME[target]} Write the dossier you would write from the checks alone: reason only `
  + "from the numbers given, as if you did not know this, and never mention follow-up, outcomes or this note.";
const VERSION = "v5";
const MODEL = "muse-glimmer";

const vetDir = new URL("../data/finetune/vet/", import.meta.url);
const outDir = new URL("../data/finetune/traces/", import.meta.url);
mkdirSync(outDir, { recursive: true });
const prompt = systemPrompt(VERSION);

// Half low, half high in training, so the tuned model keeps both ways of working; test at low, the dossiers' level.
const effortsFor = (split) => (split === "test" ? ["low"] : Array.from({ length: samples }, (_, i) => (i < samples / 2 ? "low" : "high")));

// TOIs with at least one sample that got the outcome right (and finished, with no invented numbers)
const solved = new Set();
if (hintMode) {
  for (const f of readdirSync(outDir).filter((f) => f.endsWith(".json"))) {
    const t = JSON.parse(readFileSync(new URL(f, outDir), "utf8"));
    if (!t.hint && t.verdict === t.target && !t.invented?.length && t.stop === "stop") solved.add(t.toi);
  }
}

let todo = [];
for (const f of readdirSync(vetDir).filter((f) => f.endsWith(".json"))) {
  const d = JSON.parse(readFileSync(new URL(f, vetDir), "utf8"));
  if (d.checks.code_verdict !== "agent" || !splits.includes(d.label.split)) continue;
  if (hintMode) {
    const path = new URL(`${d.toi}.${HINT_SAMPLE}.json`, outDir);
    if (d.label.split !== "test" && !solved.has(d.toi) && !existsSync(path))
      todo.push({ d, i: HINT_SAMPLE, effort: "high", path, hint: hintFor(d.label.target) });
    continue;
  }
  effortsFor(d.label.split).forEach((effort, i) => {
    const path = new URL(`${d.toi}.${i}.json`, outDir);
    if (!existsSync(path)) todo.push({ d, i, effort, path });
  });
}
if (limit) todo = todo.slice(0, limit);
console.log(`${todo.length} dossiers to write`);

// --export FILE: the prompts as JSONL for finetune/remote_traces.py, which runs them inside Modal (no laptop
// connection needed); --import FOLDER: its raw replies, parsed into trace files as below.
const exportPath = opt("--export", "");
const importDir = opt("--import", "");
if (exportPath) {
  const lines = todo.map(({ d, i, effort, hint }) => {
    const plain = glimmerPrompt(prompt, effort);
    return JSON.stringify({ id: `${d.toi}.${i}`, toi: d.toi, sample: i, effort, split: d.label.split,
      target: d.label.target, disp: d.label.disp, plain, ...(hint && { hint }),
      system: hint ? glimmerPrompt(`${prompt}\n\n${hint}`, effort) : plain, user: userMessage(anonymize(d.checks)) });
  });
  writeFileSync(exportPath, lines.join("\n") + "\n");
  console.log(`exported ${lines.length} prompts to ${exportPath}`);
  process.exit(0);
}
if (importDir) {
  const dir = pathToFileURL(`${resolve(importDir)}/`);
  let n = 0, matched = 0;
  for (const f of readdirSync(dir).filter((f) => f.endsWith(".json"))) {
    const r = JSON.parse(readFileSync(new URL(f, dir), "utf8"));
    const d = JSON.parse(readFileSync(new URL(`${r.toi}.json`, vetDir), "utf8"));
    const checks = anonymize(d.checks);
    const run = { toi: r.toi, sample: r.sample, effort: r.effort, split: r.split, target: r.target, disp: r.disp,
      ms: r.ms, tokens: r.tokens, stop: r.finish_reason === "length" ? "length" : "stop",
      system: r.plain, user: r.user, ...(r.hint && { hint: r.hint }), thinking: r.thinking ?? "", text: r.text ?? "" };
    try {
      run.dossier = parseDossier(run.text);
      run.verdict = run.dossier.verdict;
      run.invented = inventedNumbers(run.dossier, checks, prompt);
    } catch (e) {
      run.verdict = "unparseable";
      run.error = e.message;
    }
    writeFileSync(new URL(`${r.toi}.${r.sample}.json`, outDir), JSON.stringify(run, null, 1));
    logUsage(MODEL, r.split === "test" ? "eval (fine-tune test)" : "fine-tune traces", r.tokens, r.ms);
    n++;
    if (run.verdict === run.target) matched++;
  }
  console.log(`imported ${n} dossiers, ${matched} match the outcome`);
  process.exit(0);
}

async function one({ d, i, effort, path, hint }) {
  const checks = anonymize(d.checks);
  const plain = glimmerPrompt(prompt, effort);
  const context = {
    systemPrompt: hint ? glimmerPrompt(`${prompt}\n\n${hint}`, effort) : plain,
    messages: [{ role: "user", content: userMessage(checks), timestamp: Date.now() }],
    tools: [],
  };
  const t0 = Date.now();
  const msg = await complete(MODELS[MODEL], context, { apiKey: apiKey(MODEL) });
  logUsage(MODEL, d.label.split === "test" ? "eval (fine-tune test)" : "fine-tune traces", msg.usage, Date.now() - t0);
  if (msg.stopReason === "error") throw new Error(msg.errorMessage);
  const thinking = msg.content.filter((b) => b.type === "thinking").map((b) => b.thinking).join("");
  const text = msg.content.filter((b) => b.type === "text").map((b) => b.text).join("");
  const run = { toi: d.toi, sample: i, effort, split: d.label.split, target: d.label.target, disp: d.label.disp,
    ms: Date.now() - t0, tokens: { input: msg.usage.input, output: msg.usage.output }, stop: msg.stopReason,
    system: plain, user: context.messages[0].content, ...(hint && { hint }), thinking, text };
  try {
    run.dossier = parseDossier(text);
    run.verdict = run.dossier.verdict;
    run.invented = inventedNumbers(run.dossier, checks, prompt);
  } catch (e) {
    run.verdict = "unparseable";
    run.error = e.message;
  }
  writeFileSync(path, JSON.stringify(run, null, 1));
  return run;
}

let next = 0, done = 0, right = 0, failed = 0;
const started = Date.now();
async function worker() {
  while (next < todo.length) {
    const job = todo[next++];
    try {
      const r = await one(job);
      if (r.verdict === r.target) right++;
    } catch (e) {
      failed++;
      console.log(`${job.d.toi}.${job.i}: ${e.message.slice(0, 160)}`);
    }
    if (++done % 50 === 0) console.log(`${done}/${todo.length}, ${right} match the outcome, ${failed} failed, ${((Date.now() - started) / 60000).toFixed(1)} min`);
  }
}
await Promise.all(Array.from({ length: jobs }, worker));
console.log(`done: ${done}, ${right} match the outcome, ${failed} failed`);
