// One of the visitor's cores: it gets the light curve and the grid once, then runs the box search on ranges of
// trial orbits as they are handed out.
import { blsChunk } from "./core";
import type { Grid } from "./core";

let lc: { t: Float64Array; y: Float64Array; w: Float64Array; g: Grid } | null = null;

self.onmessage = (e: MessageEvent) => {
  const d = e.data;
  if (d.type === "init") { lc = d; return; }
  const { stat, code } = blsChunk(lc!.t, lc!.y, lc!.w, lc!.g, d.i0, d.i1);
  (self as unknown as Worker).postMessage({ i0: d.i0, stat, code }, [stat.buffer, code.buffer]);
};
