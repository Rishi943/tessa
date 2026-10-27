// Investigate flagged stars: the agent picks tools (pipeline/tools.py) to settle each flag, then
// writes a dossier. Tool results are measured by code and saved as evidence that gate 2 reads,
// so the agent decides what to look at, never what passes.
// Run: node agent/investigate.js VET_DIR [--model muse-spark-1.3-contributor] [--jobs 3] [--tic TIC ...]
//   Without --tic it takes every star VET_DIR/status.csv (from pipeline/gate2.py) calls flagged.
import { execFile } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { Agent } from "@earendil-works/pi-agent-core";
import { Type } from "@earendil-works/pi-ai";
import { inventedNumbers, parseDossier, userMessage } from "./dossier.js";
import { MODELS, apiKey } from "./models.js";

const VERSION = "inv1";
const PROMPT = "investigate.v1.md";
const MAX_CALLS = 5;
const ROOT = fileURLToPath(new URL("..", import.meta.url));
const PYTHON = process.env.TESSA_PYTHON ?? join(ROOT, process.platform === "win32" ? ".venv/Scripts/python.exe" : ".venv/bin/python");

const TOOLS = {
  centroid_all: "Difference imaging on every TESS sector (the first pass used the latest six): where on the sky the dip happens. For unresolved blends and offset or inconclusive centroids. Slow: downloads images.",
  sector_depths_by_source: "Transit depth per sector with the light-curve product it came from (SPOC 2-min, TESS-SPOC, QLP). For depths that change between sectors.",
  transit_depths: "Depth of every single transit: outliers, and how much of the signal the top three carry. For signals that might come from a few events.",
  stellar_radius: "Gaia DR3's radius for the star against the TIC's, with the expected transit duration and planet size from each. For transits too long for the star.",
  rotation: "The star's spot rotation period per sector, and whether the dip repeats at a multiple of it. For stars that swing at the signal period.",
  predict: "Next TESS sectors and nights on the ground when the transit can be seen, with the timing error.",
  known: "Fresh check of TOI, CTOI, planet and binary lists, VSX, Gaia and SIMBAD for this signal.",
};

function python(name, tic) {
  return new Promise((resolve, reject) => {
    execFile(PYTHON, ["-m", "pipeline.tools", name, String(tic)],
      { cwd: ROOT, timeout: 20 * 60 * 1000, maxBuffer: 1 << 24, env: { ...process.env, PYTHONIOENCODING: "utf-8" } },
      (err, stdout) => (err ? reject(new Error(`${name}: ${err.message.split("\n").slice(-3).join(" ")}`)) : resolve(stdout.trim())));
  });
}

const tools = Object.entries(TOOLS).map(([name, description]) => ({
  name, label: name, description,
  parameters: Type.Object({ tic: Type.Integer({ description: "TESS Input Catalog ID of the star under investigation" }) }),
  execute: async (_id, params) => {
    const out = await python(name, params.tic);
    return { content: [{ type: "text", text: out }], details: JSON.parse(out) };
  },
}));

function flagsOf(dir, tic) {
  const line = readFileSync(join(dir, "status.csv"), "utf8").split("\n").find((l) => l.startsWith(`${tic},`));
  const reasons = line?.match(/^\d+,\w+,"?(.*?)"?\r?$/)?.[1] ?? "";
  return reasons.split("; ").filter(Boolean);
}

export async function investigate(checks, flags, modelId) {
  const prompt = readFileSync(new URL(`./prompts/${PROMPT}`, import.meta.url), "utf8");
  let calls = 0;
  const used = [];
  const agent = new Agent({
    initialState: { systemPrompt: prompt, model: MODELS[modelId], tools, messages: [] },
    getApiKey: async () => apiKey(),
    toolExecution: "sequential",
    beforeToolCall: async ({ args }) => {
      if (args.tic !== checks.tic) return { block: true, reason: `tools are for TIC ${checks.tic} only` };
      if (++calls > MAX_CALLS) return { block: true, reason: "tool budget used up: give your verdict now" };
    },
    afterToolCall: async ({ toolCall, result, isError }) => {
      used.push({ tool: toolCall.name, ok: !isError, result: isError ? String(result?.content?.[0]?.text ?? "") : result.details });
    },
  });
  const t0 = Date.now();
  await agent.prompt(`Star: TIC ${checks.tic}\nFlags raised by the code:\n- ${flags.join("\n- ")}\n\n${userMessage(checks)}`);
  const assistant = agent.state.messages.filter((m) => m.role === "assistant");
  const last = assistant.at(-1);
  if (last?.stopReason === "error") throw new Error(last.errorMessage);
  const raw = last.content.filter((b) => b.type === "text").map((b) => b.text).join("");
  const dossier = parseDossier(raw);
  const known = { ...checks, tools: used.map((u) => u.result) };
  return {
    model: modelId, prompt: VERSION, ms: Date.now() - t0, flags, tools: used, raw, dossier,
    cost: assistant.reduce((s, m) => s + (m.usage?.cost?.total ?? 0), 0),
    tokens: { input: assistant.reduce((s, m) => s + (m.usage?.input ?? 0), 0), output: assistant.reduce((s, m) => s + (m.usage?.output ?? 0), 0) },
    invented: inventedNumbers(dossier, known, prompt),
  };
}

if (import.meta.url === `file:///${process.argv[1].replace(/\\/g, "/")}`) {
  const args = process.argv.slice(2);
  const opt = (name, fallback) => {
    const at = args.indexOf(name);
    return at >= 0 ? args.splice(at, 2)[1] : fallback;
  };
  const model = opt("--model", "muse-spark-1.3-contributor");
  const jobs = Number(opt("--jobs", "3"));
  const at = args.indexOf("--tic");
  const only = at >= 0 ? args.splice(at).slice(1).map(Number) : null;
  const [dir] = args;
  const status = readFileSync(join(dir, "status.csv"), "utf8").split("\n").slice(1).filter(Boolean);
  const tics = only ?? status.filter((l) => l.split(",")[1] === "flagged").map((l) => Number(l.split(",")[0]));
  const outDir = join(ROOT, "data", "dossiers");
  mkdirSync(outDir, { recursive: true });

  const rows = [];
  let next = 0;
  async function worker() {
    while (next < tics.length) {
      const tic = tics[next++];
      const path = join(outDir, `${tic}.${model}.${VERSION}.json`);
      let run;
      if (existsSync(path)) run = JSON.parse(readFileSync(path, "utf8"));
      else {
        try {
          const checks = JSON.parse(readFileSync(join(dir, `${tic}.json`), "utf8"));
          run = await investigate(checks, flagsOf(dir, tic), model);
          writeFileSync(path, JSON.stringify(run, null, 1));
        } catch (e) {
          run = { error: String(e.message || e), cost: 0, tools: [] };
        }
      }
      rows.push({ tic, verdict: run.dossier?.verdict ?? "error", cost: run.cost ?? 0,
        tools: (run.tools ?? []).map((t) => t.tool).join(" "), invented: (run.invented ?? []).length, error: run.error ?? "" });
      const r = rows.at(-1);
      console.log(`${rows.length}/${tics.length} TIC ${tic}: ${r.verdict} [${r.tools}] $${r.cost.toFixed(4)}${r.error ? " " + r.error : ""}`);
    }
  }
  await Promise.all(Array.from({ length: jobs }, worker));
  const csv = ["tic,verdict,cost,tools,invented,error",
    ...rows.map((r) => [r.tic, r.verdict, r.cost.toFixed(5), `"${r.tools}"`, r.invented, `"${r.error}"`].join(","))];
  writeFileSync(join(dir, `investigations.${model}.${VERSION}.csv`), csv.join("\n") + "\n");
  const count = (v) => rows.filter((r) => r.verdict === v).length;
  console.log(`gate2 ${count("gate2")}, flag ${count("flag")}, reject ${count("reject")}, error ${count("error")}; ` +
    `$${rows.reduce((s, r) => s + r.cost, 0).toFixed(4)}`);
}
