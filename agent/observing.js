// The "how to check" note for every candidate gate 2 found: next chances, telescope size, what
// would prove it wrong. Run: node agent/observing.js VET_DIR [--model muse-spark-1.3-contributor]
//   Reads VET_DIR/candidates/TIC.json (pipeline/gate2.py) and adds an "observing" note to each.
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { complete } from "@earendil-works/pi-ai";
import { inventedNumbers } from "./dossier.js";
import { MODELS, apiKey } from "./models.js";

const args = process.argv.slice(2);
const at = args.indexOf("--model");
const model = at >= 0 ? args.splice(at, 2)[1] : "muse-spark-1.3-contributor";
const [dir] = args;
const prompt = readFileSync(new URL("./prompts/observing.v1.md", import.meta.url), "utf8");

for (const file of readdirSync(join(dir, "candidates")).filter((f) => f.endsWith(".json"))) {
  const path = join(dir, "candidates", file);
  const c = JSON.parse(readFileSync(path, "utf8"));
  const w = c.when;
  const r = (v, d) => Number(v.toFixed(d));  // as precise as it is worth saying
  const input = {
    tic: c.tic, period_d: r(w.ephemeris.period, 5), period_err_d: Number(w.ephemeris.period_err.toPrecision(2)),
    depth_ppm: Math.round(c.signal.depth_ppm), duration_h: r(w.ephemeris.t14_h, 1), planet_radius_earth: r(c.size.rp_rearth, 1),
    star_tmag: r(c.star.tmag, 1), star_teff_k: Math.round(c.star.teff), distance_pc: Math.round(c.star.distance),
    timing_err_now_min: w.timing_err_now_min,
    tess_sectors: w.tess_sectors.filter((s) => s.transits && !s.lost).slice(0, 3),
    ground_nights: w.ground.slice(0, 6).map((g) => ({ site: g.site, from_utc: g.from_utc, to_utc: g.to_utc, timing_err_min: g.timing_err_min })),
    ground_nights_in_90_days: w.ground_nights, ground_snr_per_transit: w.ground_snr_per_transit,
  };
  const msg = await complete(MODELS[model], { systemPrompt: prompt, tools: [],
    messages: [{ role: "user", content: JSON.stringify(input), timestamp: Date.now() }] }, { apiKey: apiKey() });
  if (msg.stopReason === "error") {
    console.log(`TIC ${c.tic}: ${msg.errorMessage}`);
    continue;
  }
  const text = msg.content.filter((b) => b.type === "text").map((b) => b.text).join("");
  const note = JSON.parse(text.slice(text.indexOf("{"), text.lastIndexOf("}") + 1));
  const invented = inventedNumbers({ reasons: [{ value: "", why: JSON.stringify(note) }] }, input, prompt);
  c.observing = { ...note, model, prompt: "observing.v1", cost_usd: msg.usage.cost.total, invented };
  writeFileSync(path, JSON.stringify(c, null, 1));
  console.log(`TIC ${c.tic}: ${note.headline}\n  ${note.telescope}\n  invented: ${JSON.stringify(invented)}; $${msg.usage.cost.total.toFixed(5)}`);
}
