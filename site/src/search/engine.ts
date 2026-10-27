// Runs the search for one star on every core the visitor has, a range of trial orbits at a time, and reports
// progress as it goes; then picks the peaks and runs gate 1 on the main thread.
import { gate, grid, topPeaks } from "./core";
import type { Grid, LC, Verdict } from "./core";

export interface Run { g: Grid; stat: Float32Array; verdict: Verdict; seconds: number; cores: number }
const CHUNK = 1500;  // trial orbits per message: small enough for smooth progress, big enough to keep cores busy

export const cores = () => Math.max(1, Math.min((navigator.hardwareConcurrency || 4) - 1, 12));

export function run(lc: LC, rad: number, mass: number, floor: number,
  onProgress: (done: number, total: number, stat: Float32Array, g: Grid) => void): { done: Promise<Run>; stop(): void } {
  const t0 = performance.now();
  const rel = lc.t.map((x) => x - lc.tRef);
  const g = grid(rel[rel.length - 1], rad, mass);
  const n = g.freq.length, stat = new Float32Array(2 * n), code = new Int32Array(2 * n);
  const k = cores(), workers: Worker[] = [];
  let next = 0, finished = 0, stopped = false;
  const done = new Promise<Run>((resolve, reject) => {
    if (!n) { reject(new Error("this light curve is too short to search")); return; }
    const give = (w: Worker) => {
      if (next >= n) return;
      const i0 = next, i1 = Math.min(n, next + CHUNK);
      next = i1;
      w.postMessage({ i0, i1 });
    };
    for (let i = 0; i < k; i++) {
      const w = new Worker(new URL("./bls.worker.ts", import.meta.url), { type: "module" });
      w.onmessage = (e: MessageEvent) => {
        if (stopped) return;
        const { i0, stat: s, code: c } = e.data as { i0: number; stat: Float32Array; code: Int32Array };
        stat.set(s, 2 * i0); code.set(c, 2 * i0);
        finished += s.length / 2;
        onProgress(finished, n, stat, g);
        if (finished >= n) {
          workers.forEach((x) => x.terminate());
          const verdict = gate(topPeaks(lc, g, stat, code, 0), topPeaks(lc, g, stat, code, 1), floor);
          resolve({ g, stat, verdict, seconds: (performance.now() - t0) / 1000, cores: k });
        } else give(w);
      };
      w.onerror = (e) => { workers.forEach((x) => x.terminate()); reject(new Error(e.message || "the search stopped")); };
      w.postMessage({ type: "init", t: rel, y: lc.y, w: lc.w, g });
      workers.push(w);
      give(w);
    }
  });
  return { done, stop() { stopped = true; workers.forEach((x) => x.terminate()); } };
}
