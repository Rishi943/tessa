// "Search a star": a visitor runs Tessa's search on one held-back star nobody has searched, in their own browser,
// and sees it happen: the odds, their star on the map, every trial orbit scored as it goes, then the verdict.
import { axes } from "../methods";
import { rich } from "../tourcards";
import { el, int } from "../util";
import { decode, starShape } from "./core";
import type { Grid, LC } from "./core";
import { cores, run } from "./engine";
import type { Run } from "./engine";

export interface SearchStar { tic: number; ra: number; dec: number; dist: number; tmag: number; teff: number | null; rad: number | null; mass: number | null }
export interface SearchOpts { floor: number; odds: string; start?: boolean; onStar(s: SearchStar): void; onLeave(): void; onAgain(): void }

const css = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const LY = 3.26156;
const HANDLE = "tessa.handle";
const kind = (teff: number | null) => (teff == null ? "a small star" : teff < 3900 ? "a red dwarf" : teff < 5300 ? "an orange dwarf" : "a star");

export function searchCard(o: SearchOpts): { node: HTMLElement; dispose(): void } {
  const text = el("p", { class: "t-text", "aria-live": "polite" });
  const big = el("p", { class: "t-big" }), sub = el("p", { class: "t-sub" });
  const leave = el("button", { type: "button", class: "u-bar-btn" }, "Back to the map");
  leave.addEventListener("click", () => o.onLeave());
  const plot = el("canvas", { class: "m-plot short" }), fold = el("canvas", { class: "m-graph" });
  const stage = el("div", { class: "t-stage" }, el("div", { class: "m-show" }, plot, fold));
  const handle = el("input", { class: "s-handle", type: "text", maxlength: 24, placeholder: "Your name or @handle (optional)", "aria-label": "Your name or handle, shown as the star's searcher" });
  try { handle.value = localStorage.getItem(HANDLE) ?? ""; } catch { /* private mode */ }
  const go = el("button", { type: "button", class: "u-btn" }, "Start the search");
  const ctrl = el("div", { class: "t-ctrl" }, handle, go);
  const node = el("section", { class: "t-card bare", "aria-label": "Search a star" },
    el("div", { class: "t-side" },
      el("div", { class: "t-head" }, el("div", { class: "t-top" }, el("span", { class: "t-where" }, el("span", { class: "u-dot", style: "background: var(--ok)" }),
        el("b", {}, "Search a star")), leave)),
      el("div", { class: "t-say" }, text), el("div", { class: "t-stat" }, big, sub), ctrl),
    stage);
  const say = (s: string) => text.replaceChildren(...rich(s));
  const stat = (a: string, b: string) => { big.textContent = a; sub.textContent = b; };

  say(`Your computer runs Tessa's search on one star nobody has searched yet. About ${o.odds} stars gives a [[candidate]], so most searches find nothing, and that still counts: every star searched is one fewer to go.`);
  fetch("/api/search/stats").then((r) => r.json()).then((s: { left: number }) => stat(int(s.left), "stars left to search")).catch(() => {});

  let job: { stop(): void } | null = null, dead = false;
  if (o.start) queueMicrotask(() => go.click());
  go.addEventListener("click", async () => {
    try { localStorage.setItem(HANDLE, handle.value.trim()); } catch { /* private mode */ }
    ctrl.replaceChildren();
    try {
      const res = await fetch("/api/search/next");
      const body = await res.json() as { star?: SearchStar; error?: string };
      if (!body.star) throw new Error(body.error ?? "no star free right now");
      const star = body.star;
      o.onStar(star);
      const buf = await (await fetch(`/data/search/${star.tic}.bin`)).arrayBuffer();
      if (dead) return;
      const lc = decode(buf);
      const sectors = new Set(lc.sector).size, years = (lc.t[lc.t.length - 1] - lc.t[0]) / 365.25;
      say(`Your star: TIC ${star.tic}, ${kind(star.teff)} ${int(star.dist * LY)} light-years away. [[TESS]] watched it in ${sectors} [[sector|sectors]]${years >= 1 ? ` over ${Math.round(years)} years` : ""}: ${int(lc.t.length)} measurements of its [[light curve|light]].`);
      stat(`TIC ${star.tic}`, "your star");
      node.classList.remove("bare");
      await new Promise((r) => setTimeout(r, 2500));
      if (dead) return;
      const { rad, mass } = starShape(star.rad, star.mass);
      const k = cores();
      const running = run(lc, rad, mass, o.floor, (done, total, s, g) => {
        say(`Trying ${int(total)} possible [[orbit|orbits]] on ${k} of your computer's cores: ${Math.floor((100 * done) / total)}%.`);
        periodogram(plot, s, g, o.floor, null);
      });
      job = running;
      const r = await running.done;
      if (dead) return;
      verdict(star, lc, r);
    } catch (e) {
      say(`The search couldn't run: ${(e as Error).message}. Try again in a minute.`);
      ctrl.replaceChildren(again());
    }
  });

  function verdict(star: SearchStar, lc: LC, r: Run) {
    const v = r.verdict, who = handle.value.trim();
    periodogram(plot, r.stat, r.g, o.floor, v.dip ? v.dip.period : null);
    if (v.dip) foldPlot(fold, lc, v.dip.period, v.dip.t0, v.dip.dur_h / 24);
    say(v.pass
      ? `This star cleared the bar: a dip every ${v.dip!.period.toFixed(3)} days, ${Math.round(v.dip!.depth_ppm)} [[ppm]] deep, [[score]] ${v.dip!.snr.toFixed(1)}. Tessa checks it next${who ? `, and credits ${who}` : ""}.`
      : `No dip strong enough this time: the best [[score|scored]] ${(v.dip?.snr ?? v.topDip).toFixed(1)}, and the bar is ${o.floor}. Most stars show no planet we can see, so this is a real result: TIC ${star.tic} is searched.`);
    stat(`${r.seconds.toFixed(0)} s`, `${int(r.g.freq.length)} orbits on ${r.cores} cores`);
    const share = `${location.origin}/search`;
    const post = `I searched TIC ${star.tic} for planets in my browser with Tessa: ${v.pass ? "it cleared the bar!" : "no planet this time, one more star searched."}`;
    ctrl.replaceChildren(again(),
      el("a", { class: "u-btn", href: `https://x.com/intent/post?text=${encodeURIComponent(post)}&url=${encodeURIComponent(share)}`, target: "_blank", rel: "noopener" }, "Share"));
    fetch("/api/search/result", { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ tic: star.tic, handle: who, pass: v.pass, dip: v.dip, topBump: v.bump?.snr ?? v.topBump, nFreq: r.g.freq.length, seconds: r.seconds }) })
      .then((x) => x.json()).then((x: { first?: boolean; left?: number }) => {
        if (x.left != null) sub.textContent = `${int(r.g.freq.length)} orbits on ${r.cores} cores. ${int(x.left)} stars left.`;
      }).catch(() => {});
  }
  function again() {
    const b = el("button", { type: "button", class: "u-btn" }, "Search another star");
    b.addEventListener("click", () => o.onAgain());
    return b;
  }
  function dispose() { dead = true; job?.stop(); node.remove(); }
  return { node, dispose };
}

// every trial orbit's best dip score, by period, filling in as the cores report; the bar dashed across
function periodogram(c: HTMLCanvasElement, stat: Float32Array, g: Grid, floor: number, mark: number | null) {
  const { g: x, w, B, T, L } = axes(c, "dip score", "");
  const lo = Math.log(0.5), hi = Math.log(15), X = (p: number) => L + 2 + ((Math.log(p) - lo) / (hi - lo)) * (w - L - 4);
  const top = Math.max(floor * 1.6, 1);
  const Y = (s: number) => B - Math.min(s / top, 1) * (B - T - 8);
  const cols = new Float32Array(Math.ceil(w));
  for (let m = 0; m < g.freq.length; m++) {
    const s = Math.sqrt(stat[2 * m]);
    if (!s) continue;
    const i = Math.round(X(1 / g.freq[m]));
    if (s > cols[i]) cols[i] = s;
  }
  x.strokeStyle = "rgba(196, 195, 189, 0.55)"; x.lineWidth = 1; x.beginPath();
  cols.forEach((s, i) => { if (s) { x.moveTo(i + 0.5, B); x.lineTo(i + 0.5, Y(s)); } });
  x.stroke();
  x.strokeStyle = css("--ink"); x.setLineDash([4, 4]); x.beginPath(); x.moveTo(L, Y(floor)); x.lineTo(w, Y(floor)); x.stroke(); x.setLineDash([]);
  x.fillStyle = css("--ink"); x.fillText(`the bar: ${floor}`, w - 96, Y(floor) - 6);
  if (mark) {
    x.fillStyle = "#5eead4"; x.beginPath(); x.arc(X(mark), T + 10, 4, 0, 7); x.fill();
    x.fillText(`best: ${mark.toFixed(3)} d`, Math.min(X(mark) + 8, w - 110), T + 14);
  }
  // the orbit lengths along the bottom, in days on a log scale
  for (const p of [0.5, 1, 2, 5, 10, 15]) { x.fillStyle = css("--muted"); x.fillText(p === 15 ? "15 days" : String(p), p === 15 ? X(p) - 52 : X(p) - 4, B + 14); }
}

// the light curve folded on the best orbit, binned, around the dip
function foldPlot(c: HTMLCanvasElement, lc: LC, period: number, t0: number, dur: number) {
  const half = Math.min(dur * 3, period / 2), nb = 60;
  const sw = new Float64Array(nb), sy = new Float64Array(nb);
  for (let j = 0; j < lc.t.length; j++) {
    const ph = ((((lc.t[j] - t0) / period + 0.5) % 1) + 1) % 1 - 0.5, d = ph * period;
    if (Math.abs(d) >= half) continue;
    const b = Math.floor(((d + half) / (2 * half)) * nb);
    sw[b] += lc.w[j]; sy[b] += lc.w[j] * lc.y[j];
  }
  const ys = [...sy].map((s, i) => (sw[i] ? (s / sw[i]) * 1e6 : NaN));
  const fin = ys.filter(Number.isFinite), span = Math.max(...fin.map(Math.abs), 1) * 1.3;
  const { g, w, B, T, L } = axes(c, "brightness", "hours from mid-dip →");
  const X = (i: number) => L + 2 + ((i + 0.5) / nb) * (w - L - 4), Y = (v: number) => (T + B) / 2 - (v / span) * ((B - T) / 2);
  g.strokeStyle = "#5eead4"; g.lineWidth = 2; g.beginPath();
  ys.forEach((v, i) => { if (Number.isFinite(v)) { if (i && Number.isFinite(ys[i - 1])) g.lineTo(X(i), Y(v)); else g.moveTo(X(i), Y(v)); } });
  g.stroke();
  g.fillStyle = css("--muted"); g.fillText(`folded on ${period.toFixed(3)} days, ±${(half * 24).toFixed(1)} h`, L + 6, T + 10);
}
