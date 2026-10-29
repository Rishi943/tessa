// Write a vetting dossier for one flagged signal with each given model, and score it:
// verdict, cost, and how many numbers it cites that aren't in its inputs (invented values).
// Run: node agent/dossier.js CHECKS.json [--prompt v2] [model ...]
//   CHECKS.json comes from `python -m pipeline.vet TIC --results results.csv > CHECKS.json`.
import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { complete } from "@earendil-works/pi-ai";
import { MODELS, apiKey, glimmerPrompt } from "./models.js";
import { logUsage } from "./usage.js";

// System prompts live in agent/prompts/vetting.<version>.md so versions can be compared.
export function systemPrompt(version) {
  return readFileSync(new URL(`./prompts/vetting.${version}.md`, import.meta.url), "utf8");
}

// Four significant figures for measurements; integers (TIC and Gaia IDs, sectors, counts) stay exact.
function round(v) {
  return typeof v === "number" && !Number.isInteger(v) ? Number(v.toPrecision(4)) : v;
}

// What prompt v6 reads beyond v5's checks: System 1's score and drivers, the similar TOIs in words, and the
// two fractions its odd/even and secondary rules use. The raw clues stay out: System 1 has read them.
const OUTCOME = { gate2: "planet", reject: "not a planet", flag: "ambiguous" };
export function systemOneView(checks) {
  const c = structuredClone(checks);
  delete c.clues;
  const { depth_odd_ppm: a, depth_even_ppm: b } = c.odd_even ?? {};
  const deeper = Math.max(Math.abs(a ?? 0), Math.abs(b ?? 0));
  if (a != null && b != null && deeper) c.odd_even.depth_difference = Math.abs(a - b) / deeper;
  if (c.secondary?.depth_ppm != null && c.refit?.depth_ppm) c.secondary.depth_over_dip = c.secondary.depth_ppm / c.refit.depth_ppm;
  if (c.system1) {
    const { p_planet, planet_rank, drivers } = c.system1;
    c.system1 = { p_planet, planet_rank, drivers: drivers.map((d) => ({ check: d.label, value: d.value, log_odds: d.log_odds })) };
  }
  if (c.similar) c.similar = c.similar.map((s) => ({ outcome: OUTCOME[s.target], ...(s.reason && s.reason !== "unknown" && { reason: s.reason }) }));
  return c;
}

// The input a prompt version reads: v6 on sees System 1's summary in place of the raw clues.
export function promptInput(checks, version) {
  return Number(version.slice(1)) >= 6 ? systemOneView(checks) : checks;
}

export function userMessage(checks) {
  const compact = JSON.parse(JSON.stringify(checks, (k, v) => round(v)));
  return `Flagged signal and the code checks already run on it:\n${JSON.stringify(compact)}`;
}

function numbersIn(text) {
  // "10-22" is a range and "4101+/-718" an error bar, not -22 or -718; 5.4e-05 is one number
  return (text.match(/(?<![\w./+])-?\d+(?:\.\d+)?(?:e[-+]?\d+)?/gi) || []).map(Number);
}

// Numbers in the dossier's reasons that match nothing in its inputs (checks or prompt) within 3%,
// also as a percentage. Small integers are skipped: they are counts ("7 sectors").
export function inventedNumbers(dossier, checks, prompt) {
  const known = numbersIn(JSON.stringify(checks) + prompt);
  const rank = checks.system1?.planet_rank;
  if (typeof rank === "number") known.push(1 - rank); // "less planet-like than 96% of planets"
  const near = (a, b) => Math.abs(a - b) <= Math.max(0.15, 0.03 * Math.abs(b));
  const cited = dossier.reasons.flatMap((r) => numbersIn(`${r.value} ${r.why}`));
  return cited.filter((x) => !(Number.isInteger(x) && Math.abs(x) <= 10) && !known.some((k) => near(x, k) || near(x, 100 * k)));
}

export function parseDossier(text) {
  const body = text.replace(/^```(?:json)?\s*|\s*```$/g, "").trim();
  const d = JSON.parse(body.slice(body.indexOf("{"), body.lastIndexOf("}") + 1));
  if (!["reject", "flag", "gate2"].includes(d.verdict)) throw new Error(`bad verdict ${d.verdict}`);
  if (!Array.isArray(d.reasons)) throw new Error("reasons missing");
  if (!d.reasons.every((r) => typeof r?.check === "string" && typeof r?.why === "string")) throw new Error("a reason lacks check or why");
  return d;
}

// effort sets Muse Spark's reasoning_effort ("low" ... "xhigh"); without it the model picks its own level.
// Glimmer takes the same levels as a line in its system prompt.
export async function writeDossier(checks, id, version, effort, purpose = "dossier") {
  const glimmer = MODELS[id].provider === "glimmer";
  const model = effort && !glimmer ? { ...MODELS[id], reasoning: true, compat: { ...MODELS[id].compat, supportsReasoningEffort: true } } : MODELS[id];
  const context = {
    systemPrompt: glimmer ? glimmerPrompt(systemPrompt(version), effort) : systemPrompt(version),
    messages: [{ role: "user", content: userMessage(promptInput(checks, version)), timestamp: Date.now() }],
    tools: [],
  };
  const t0 = Date.now();
  const msg = await complete(model, context, { apiKey: apiKey(id), ...(effort && !glimmer && { reasoningEffort: effort }) });
  logUsage(id, purpose, msg.usage, Date.now() - t0);
  if (msg.stopReason === "error") throw new Error(msg.errorMessage);
  const text = msg.content.filter((b) => b.type === "text").map((b) => b.text).join("");
  return {
    model: id, prompt: version, ...(effort && { effort }), ms: Date.now() - t0,
    tokens: { input: msg.usage.input, cached: msg.usage.cacheRead, output: msg.usage.output }, cost: msg.usage.cost.total,
    raw: text,
  };
}

if (import.meta.url === `file:///${process.argv[1].replace(/\\/g, "/")}`) {
  const args = process.argv.slice(2);
  const at = args.indexOf("--prompt");
  const version = at >= 0 ? args.splice(at, 2)[1] : "v2";
  const [path, ...ids] = args;
  const checks = JSON.parse(readFileSync(path, "utf8"));
  const outDir = new URL("../data/dossiers/", import.meta.url);
  mkdirSync(outDir, { recursive: true });
  for (const id of ids.length ? ids : ["muse-spark-1.3-contributor"]) {
    const run = await writeDossier(checks, id, version);
    let line = `${id.padEnd(28)} $${run.cost.toFixed(4)}  ${run.tokens.input}/${run.tokens.output} tok  ${(run.ms / 1000).toFixed(1)} s  `;
    try {
      run.dossier = parseDossier(run.raw);
      const invented = inventedNumbers(run.dossier, checks, systemPrompt(version));
      line += `${run.dossier.verdict.padEnd(6)} ${run.dossier.reasons.length} reasons, invented numbers: ${invented.length ? invented.join(", ") : "none"}`;
    } catch (e) {
      line += `UNPARSEABLE (${e.message})`;
    }
    writeFileSync(new URL(`${checks.tic}.${id}.${version}.json`, outDir), JSON.stringify(run, null, 1));
    console.log(line);
  }
}
