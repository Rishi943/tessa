// One plain sentence for every rejected star, for its "case closed" page, twenty stars per call.
// Run: node agent/caseclosed.js VET_DIR [--model muse-spark-1.3-contributor]
//   Reads VET_DIR/status.csv (pipeline/gate2.py); writes VET_DIR/caseclosed.json, keeping earlier lines.
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { complete } from "@earendil-works/pi-ai";
import { MODELS, apiKey } from "./models.js";

const PER_CALL = 20;
const args = process.argv.slice(2);
const at = args.indexOf("--model");
const model = at >= 0 ? args.splice(at, 2)[1] : "muse-spark-1.3-contributor";
const [dir] = args;

function parseCsv(text) {
  const rows = [];
  for (const line of text.split(/\r?\n/).slice(1).filter(Boolean)) {
    const m = line.match(/^(\d+),([^,]+),"?(.*?)"?$/);
    if (m) rows.push({ tic: m[1], status: m[2], reasons: m[3] });
  }
  return rows;
}

const outPath = join(dir, "caseclosed.json");
const notes = existsSync(outPath) ? JSON.parse(readFileSync(outPath, "utf8")) : {};
const todo = parseCsv(readFileSync(join(dir, "status.csv"), "utf8")).filter((r) => r.status === "rejected" && !notes[r.tic]);
const prompt = readFileSync(new URL("./prompts/caseclosed.v1.md", import.meta.url), "utf8");
let cost = 0;
for (let i = 0; i < todo.length; i += PER_CALL) {
  const batch = todo.slice(i, i + PER_CALL);
  const content = batch.map((r) => `TIC ${r.tic}: ${r.reasons}`).join("\n");
  const msg = await complete(MODELS[model], { systemPrompt: prompt, messages: [{ role: "user", content, timestamp: Date.now() }], tools: [] },
    { apiKey: apiKey() });
  cost += msg.usage.cost.total;
  if (msg.stopReason === "error") {
    console.log(`batch ${i / PER_CALL}: ${msg.errorMessage}`);
    continue;
  }
  const text = msg.content.filter((b) => b.type === "text").map((b) => b.text).join("");
  const got = JSON.parse(text.slice(text.indexOf("{"), text.lastIndexOf("}") + 1));
  for (const r of batch) if (got[r.tic]) notes[r.tic] = got[r.tic];
  writeFileSync(outPath, JSON.stringify(notes, null, 1));
  console.log(`${Math.min(i + PER_CALL, todo.length)}/${todo.length}`);
}
const metaPath = join(dir, "caseclosed.meta.json");  // running cost, for the site's funnel
const meta = existsSync(metaPath) ? JSON.parse(readFileSync(metaPath, "utf8")) : { model, cost_usd: 0 };
meta.cost_usd += cost;
writeFileSync(metaPath, JSON.stringify(meta, null, 1));
console.log(`${Object.keys(notes).length} notes; $${cost.toFixed(4)}`);
