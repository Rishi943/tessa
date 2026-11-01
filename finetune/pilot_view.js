// The input prompt v10 reads, for finetune/pilot.py: argv[3] is a file of [{toi, checks, extra}] (Windows pipes fail on a big batch), stdout [{toi, user}].
// The production anonymised view with System 1's output, the code verdict, its reasons and the tools' summary
// removed; `extra` (measurements the vet files leave out) goes in under "more".
import { readFileSync } from "node:fs";
import { anonymize } from "../agent/anonymize.js";
import { systemOneView, userMessage } from "../agent/dossier.js";

// v10 only: values models kept computing by hand. Each is skipped when its inputs are missing.
const sig = (v, n) => Number(v.toPrecision(n));
function derived(c) {
  const { depth_odd_ppm: odd, depth_even_ppm: even } = c.odd_even ?? {};
  if (Number.isFinite(odd) && Number.isFinite(even)) {
    c.odd_even.depth_difference_ppm = Math.abs(odd - even);
    const [hi, lo] = [Math.max(Math.abs(odd), Math.abs(even)), Math.min(Math.abs(odd), Math.abs(even))];
    if (lo > 0) c.odd_even.depth_ratio = sig(hi / lo, 2);
  }
  const d = (c.sectors?.per_sector ?? []).map((x) => x.depth_ppm).filter(Number.isFinite);
  const mean = d.length ? d.reduce((a, b) => a + b, 0) / d.length : 0;
  if (d.length > 1 && mean > 0) c.sectors.depth_change_pct = (100 * (Math.max(...d) - Math.min(...d))) / mean;
  if (Number.isFinite(c.signal?.period)) c.signal.period_x2 = sig(2 * sig(c.signal.period, 4), 3);
}

const version = process.argv[2] ?? "v10";
const jobs = JSON.parse(readFileSync(process.argv[3], "utf8"));
const out = jobs.map(({ toi, checks, extra, score }) => {
  const c = anonymize(systemOneView(checks, version));
  for (const k of ["system1", "code_verdict", "reasons", "tools"]) delete c[k];
  if (Number(version.slice(1)) >= 10) derived(c);
  if (extra && Object.keys(extra).length) c.more = extra;
  if (score !== undefined) c.xgboost = { planet_rank: score };
  return { toi, user: userMessage(c) };
});
process.stdout.write(JSON.stringify(out));
