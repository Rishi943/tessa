// Runs the browser search (src/search/core.ts) on survey stars whose GPU search results are known, and prints
// both side by side. Run from site/: node scripts/search-parity.mjs [path to data/site/search/parity]
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { blsChunk, checkPeak, decode, gate, grid, starShape, topPeaks } from "../src/search/core.ts";

const dir = process.argv[2] ?? "../data/site/search/parity";
for (const f of readdirSync(dir).filter((x) => x.endsWith(".json"))) {
  const ref = JSON.parse(readFileSync(join(dir, f), "utf-8"));
  const buf = readFileSync(join(dir, `${ref.tic}.bin`));
  const lc = decode(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  const { rad, mass } = starShape(ref.rad, ref.mass);
  const rel = lc.t.map((x) => x - lc.tRef);
  const g = grid(lc.t[lc.t.length - 1] - lc.tRef, rad, mass);
  const t0 = performance.now();
  const { stat, code } = blsChunk(rel, lc.y, lc.w, g, 0, g.freq.length);
  const secs = (performance.now() - t0) / 1000;
  const v = gate(topPeaks(lc, g, stat, code, 0), topPeaks(lc, g, stat, code, 1), 14.3);
  console.log(`TIC ${ref.tic}: ${g.freq.length} trial orbits (GPU ${ref.n_freq}) in ${secs.toFixed(1)} s on one core`);
  console.log(`  dip  ${v.dip?.snr.toFixed(3)} at ${v.dip?.period.toFixed(5)} d, ${v.dip?.transits} transits   GPU ${ref.dip_snr} at ${ref.dip_period}, ${ref.dip_transits}`);
  console.log(`  bump ${(v.bump?.snr ?? v.topBump).toFixed(3)}   GPU ${ref.bump_snr || ref.top_bump_snr}   gate: ${v.pass ? "pass" : "fail"} (GPU ${ref.gate1})`);
  void checkPeak;
}
