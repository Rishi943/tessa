// Write dossiers for every code-vetted star in a folder that reached the agent, a few at a time,
// and summarise them. Existing dossiers are kept, so a rerun only fills gaps.
// Run: node agent/batch.js VET_DIR [--prompt v4] [--model muse-spark-1.3-contributor] [--jobs 6]
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { inventedNumbers, parseDossier, systemPrompt, writeDossier } from "./dossier.js";

const args = process.argv.slice(2);
const opt = (name, fallback) => {
  const at = args.indexOf(name);
  return at >= 0 ? args.splice(at, 2)[1] : fallback;
};
const version = opt("--prompt", "v4");
const model = opt("--model", "muse-spark-1.3-contributor");
const jobs = Number(opt("--jobs", "6"));
const [dir] = args;

const outDir = new URL("../data/dossiers/", import.meta.url);
mkdirSync(outDir, { recursive: true });
const stars = readdirSync(dir)
  .filter((f) => /^\d+\.json$/.test(f))
  .map((f) => JSON.parse(readFileSync(join(dir, f), "utf8")))
  .filter((c) => c.code_verdict === "agent");

async function one(checks) {
  const path = new URL(`${checks.tic}.${model}.${version}.json`, outDir);
  if (existsSync(path)) return JSON.parse(readFileSync(path, "utf8"));
  for (let attempt = 1; ; attempt++) {
    try {
      const run = await writeDossier(checks, model, version);
      run.dossier = parseDossier(run.raw);
      run.invented = inventedNumbers(run.dossier, checks, systemPrompt(version));
      writeFileSync(path, JSON.stringify(run, null, 1));
      return run;
    } catch (e) {
      if (attempt === 2) return { error: String(e.message || e), cost: 0 };
    }
  }
}

const rows = [];
let next = 0;
async function worker() {
  while (next < stars.length) {
    const checks = stars[next++];
    const run = await one(checks);
    rows.push({ tic: checks.tic, verdict: run.dossier?.verdict ?? "error", cost: run.cost ?? 0,
      seconds: (run.ms ?? 0) / 1000, invented: (run.invented ?? []).length,
      checks: (run.dossier?.reasons ?? []).map((r) => r.check).join(" "), error: run.error ?? "" });
    console.log(`${rows.length}/${stars.length} TIC ${checks.tic}: ${rows.at(-1).verdict}`);
  }
}
await Promise.all(Array.from({ length: jobs }, worker));

const csv = ["tic,verdict,cost,seconds,invented,checks,error",
  ...rows.map((r) => [r.tic, r.verdict, r.cost.toFixed(5), r.seconds.toFixed(1), r.invented, `"${r.checks}"`, `"${r.error}"`].join(","))];
writeFileSync(join(dir, `dossiers.${model}.${version}.csv`), csv.join("\n") + "\n");
const count = (v) => rows.filter((r) => r.verdict === v).length;
const total = rows.reduce((s, r) => s + r.cost, 0);
console.log(`gate2 ${count("gate2")}, flag ${count("flag")}, reject ${count("reject")}, error ${count("error")}; $${total.toFixed(4)}`);
