// What pure noise scores. Shuffling a light curve's fluxes keeps every value but destroys any timing (no
// repeating dip, no slow drifts), so the best "dip" the search finds afterwards is what chance alone gives on
// the same grid of trial orbits. Compare with the real data's upside-down scores, which keep the slow drifts.
// Run from site/: node scripts/search-shuffle.mjs [path to parity folder] [shuffles per star]
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { blsChunk, decode, gate, grid, starShape, topPeaks } from "../src/search/core.ts";

const dir = process.argv[2] ?? "../data/site/search/parity", n = Number(process.argv[3] ?? 2);
let seed = 7;
const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
for (const f of readdirSync(dir).filter((x) => x.endsWith(".json"))) {
  const ref = JSON.parse(readFileSync(join(dir, f), "utf-8"));
  const buf = readFileSync(join(dir, `${ref.tic}.bin`));
  const lc = decode(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  const { rad, mass } = starShape(ref.rad, ref.mass);
  const g = grid(lc.t[lc.t.length - 1] - lc.tRef, rad, mass);
  for (let k = 0; k < n; k++) {
    const idx = [...lc.y.keys()];
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    const y = Float64Array.from(idx, (i) => lc.y[i]), w = Float64Array.from(idx, (i) => lc.w[i]);
    const shuffled = { ...lc, y, w };
    const { stat, code } = blsChunk(lc.t.map((x) => x - lc.tRef), y, w, g, 0, g.freq.length);
    const v = gate(topPeaks(shuffled, g, stat, code, 0), topPeaks(shuffled, g, stat, code, 1), 14.3);
    console.log(`TIC ${ref.tic} shuffle ${k + 1}: best dip ${(v.dip?.snr ?? v.topDip).toFixed(2)}, best bump ${(v.bump?.snr ?? v.topBump).toFixed(2)} (real: dip ${ref.dip_snr}, bump ${ref.bump_snr || ref.top_bump_snr})`);
  }
}
