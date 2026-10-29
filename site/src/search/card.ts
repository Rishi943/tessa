// "Search a star": a visitor runs Tessa's search on held-back stars nobody has searched, in their own browser,
// one star or a batch, and sees it happen: the odds, each star on the map, every trial orbit scored as it goes,
// then the verdict. The star being searched stays described in the corner the whole time.
import { axes } from "../methods";
import { rich } from "../tourcards";
import { el, int } from "../util";
import { decode, starShape } from "./core";
import type { Grid, LC, Verdict } from "./core";
import { cores, run } from "./engine";
import type { Run } from "./engine";

export interface SearchStar { tic: number; ra: number; dec: number; dist: number; tmag: number; teff: number | null; rad: number | null; mass: number | null }
// start: a batch size to begin at once (Ask Tessa's button, /search?go=10)
export interface SearchOpts { floor: number; odds: string; start?: number; onStar(s: SearchStar): void; onLeave(): void; onAgain(): void }

// The running batch, for Ask Tessa: she reads it to answer "how's my search going?", and hears when a star clears
// the bar or the batch ends (a "tessa:search" event on window)
interface Done { tic: number; cleared: boolean; best_score: number; period_days: number | null }
let state: { star: number; of: number; done: Done[]; minutes_left: number } | null = null;
export const searchState = () => state;
const tell = (detail: { kind: "cleared" | "finished"; tic?: number; period?: number; k: number; n: number; found: number }) =>
  dispatchEvent(new CustomEvent("tessa:search", { detail }));

const css = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const LY = 3.26156;
const HANDLE = "tessa.handle";
const TIERS = [1, 10, 50, 100];
const kind = (teff: number | null) => (teff == null ? "a small star" : teff < 3900 ? "a red dwarf" : teff < 5300 ? "an orange dwarf" : "a star");
const stars = (n: number) => (n === 1 ? "1 star" : `${int(n)} stars`);
const mins = (s: number) => (s < 90 ? `${Math.max(Math.round(s / 5) * 5, 5)} s` : `${Math.round(s / 60)} min`);

// why a star didn't clear gate 1, in the order the gate checks
function why(v: Verdict, floor: number, tic: number) {
  if (!v.dip) return `No repeating dip on TIC ${tic}: its strongest dip, [[score]] ${v.topDip.toFixed(1)}, didn't come back often enough to be an orbit.`;
  if (v.dip.snr < floor) return `No dip strong enough on TIC ${tic}: the best [[score|scored]] ${v.dip.snr.toFixed(1)}, and [[bar|the bar]] is ${floor}.`;
  const up = v.bump?.snr || v.topBump;
  return `TIC ${tic}'s best dip [[score|scored]] ${v.dip.snr.toFixed(1)}, over [[bar|the bar]], but upside down the star scores ${up.toFixed(1)}, nearly as high: that's the star's own flicker, not a planet.`;
}

export function searchCard(o: SearchOpts): { node: HTMLElement; dispose(): void } {
  // the search full screen: the star on the left, Tessa's steps on the right as a transcript, newest at the bottom
  const log = el("div", { class: "s-log", "aria-live": "polite" });
  const kick = el("p", { class: "s-kick" });
  const big = el("p", { class: "t-big" }), sub = el("p", { class: "t-sub" }), facts = el("p", { class: "s-facts" });
  const leave = el("button", { type: "button", class: "u-bar-btn" }, "Back to the map");
  leave.addEventListener("click", () => o.onLeave());
  const plot = el("canvas", { class: "m-plot short" }), fold = el("canvas", { class: "m-graph" });
  const stage = el("div", { class: "t-stage" }, el("div", { class: "m-show" }, plot, fold));
  const handle = el("input", { class: "s-handle", type: "text", maxlength: 24, placeholder: "Your name or @handle (optional)", "aria-label": "Your name or handle, shown as the stars' searcher" });
  try { handle.value = localStorage.getItem(HANDLE) ?? ""; } catch { /* private mode */ }
  // about 15 s a star on a 12-core laptop, slower with fewer cores; a batch re-estimates from its own pace
  const perStar = 15 * Math.max(12 / cores(), 1);
  const tiers = TIERS.map((n) => {
    const b = el("button", { type: "button", class: "u-btn s-tier" }, n === 1 ? "1 star" : `${n} stars`, el("small", {}, `about ${mins(n * perStar)}`));
    b.addEventListener("click", () => batch(n));
    return b;
  });
  const ctrl = el("div", { class: "t-ctrl" }, handle, ...tiers);
  const node = el("section", { class: "t-card s-full bare", "aria-label": "Search a star" },
    el("div", { class: "t-head" }, el("div", { class: "t-top" }, el("span", { class: "t-where" }, el("span", { class: "u-dot", style: "background: var(--ok)" }),
      el("b", {}, "Search a star")), leave)),
    el("div", { class: "t-stat" }, kick, big, sub, facts), ctrl,
    el("div", { class: "s-col" }, log, stage));
  // each step is a line of the transcript; a step still running (the orbit count) updates in place
  const begun = performance.now();
  const clock = () => { const s = Math.round((performance.now() - begun) / 1000); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
  let live: HTMLElement | null = null;
  const entry = (s: string) => {
    const p = el("p", { class: "s-say" }, ...rich(s));
    log.append(el("div", { class: "s-step" }, el("div", { class: "who" }, el("b", {}, "Tessa"), clock()), p));
    while (log.children.length > 8) log.firstElementChild!.remove();
    return p;
  };
  const say = (s: string) => { live = null; entry(s); };
  const sayLive = (s: string) => { if (live) live.replaceChildren(...rich(s)); else live = entry(s); };
  const stat = (a: string, b: string, c = "") => { big.textContent = a; sub.textContent = b; facts.textContent = c; };

  say(`Your computer runs Tessa's search on stars nobody has searched yet. About ${o.odds} stars gives a [[candidate]], so most searches find nothing, and that still counts: every star searched is one fewer to go. How many?`);
  fetch("/api/search/stats").then((r) => r.json()).then((s: { left: number }) => stat(int(s.left), "stars left to search")).catch(() => {});

  let job: { stop(): void } | null = null, dead = false, stopAfter = false;
  if (o.start) queueMicrotask(() => batch(o.start!));

  async function batch(n: number) {
    try { localStorage.setItem(HANDLE, handle.value.trim()); } catch { /* private mode */ }
    const who = handle.value.trim();
    const stop = el("button", { type: "button", class: "u-btn" }, "Stop after this star");
    stop.addEventListener("click", () => { stopAfter = true; stop.disabled = true; stop.textContent = "Stopping after this star"; });
    ctrl.replaceChildren(...(n > 1 ? [stop] : []));
    const t0 = performance.now(), found: number[] = [];
    let done = 0, err = "";
    state = { star: 1, of: n, done: [], minutes_left: Math.round((n * perStar) / 60) };
    for (let i = 0; i < n && !dead && !stopAfter; i++) {
      try {
        const res = await fetch("/api/search/next");
        const body = await res.json() as { star?: SearchStar; error?: string };
        if (!body.star) throw new Error(body.error ?? "no star free right now");
        const pace = done ? (performance.now() - t0) / 1000 / done : perStar;
        if (state) { state.star = i + 1; state.minutes_left = Math.round(((n - i) * pace) / 60); }
        const r = await one(body.star, n > 1 ? i + 1 : 0, n, pace);
        if (!r) return;
        done++;
        if (r.verdict.pass) found.push(body.star.tic);
        post(body.star, r, who);
        const v = r.verdict;
        state?.done.push({ tic: body.star.tic, cleared: v.pass, best_score: Number((v.dip?.snr ?? v.topDip).toFixed(1)), period_days: v.dip ? Number(v.dip.period.toFixed(3)) : null });
        if (v.pass) tell({ kind: "cleared", tic: body.star.tic, period: v.dip!.period, k: i + 1, n, found: found.length });
        if (n > 1 && i < n - 1) await new Promise((x) => setTimeout(x, r.verdict.pass ? 4000 : 1500));
      } catch (e) {
        err = `The search couldn't go on: ${(e as Error).message}. Try again in a minute.`;
        say(err);
        break;
      }
    }
    state = null;
    if (dead) return;
    if (done) tell({ kind: "finished", k: done, n, found: found.length });
    if (n > 1 && done && !err) {
      const secs = (performance.now() - t0) / 1000;
      say(found.length
        ? `Done: you searched ${stars(done)} in ${mins(secs)}, and ${found.length === 1 ? "one cleared" : `${found.length} cleared`} [[bar|the bar]]: ${found.map((t) => `TIC ${t}`).join(", ")}. Tessa checks ${found.length === 1 ? "it" : "them"} next${who ? `, crediting ${who}` : ""}.`
        : `Done: you searched ${stars(done)} in ${mins(secs)}. None cleared [[bar|the bar]] this time, which is what most stars do: ${done} fewer to go.`);
    }
    ctrl.replaceChildren(again(), share(done, found.length));
  }

  // one star: fly to it, describe it, run the search with live progress, show the verdict
  async function one(star: SearchStar, k: number, n: number, pace: number): Promise<Run | null> {
    o.onStar(star);
    const buf = await (await fetch(`/data/search/${star.tic}.bin`)).arrayBuffer();
    if (dead) return null;
    const lc = decode(buf);
    const sectors = new Set(lc.sector).size, years = (lc.t[lc.t.length - 1] - lc.t[0]) / 365.25;
    // the star stays described in the corner for the whole search
    kick.textContent = k ? `Star ${k} of your ${n}` : "Your star";
    stat(`TIC ${star.tic}`, `${kind(star.teff)}, ${int(star.dist * LY)} light-years away`,
      `${sectors} TESS sector${sectors > 1 ? "s" : ""}${years >= 1 ? ` over ${Math.round(years)} years` : ""}, ${int(lc.t.length)} measurements`);
    node.classList.remove("bare");
    const head = k ? `Star ${k} of ${n}. ` : "";
    if (!k) {
      say(`Your star: TIC ${star.tic}, ${kind(star.teff)} ${int(star.dist * LY)} light-years away. [[TESS]] watched it in ${sectors} [[sector|sectors]]${years >= 1 ? ` over ${Math.round(years)} years` : ""}: ${int(lc.t.length)} measurements of its [[light curve|light]].`);
      await new Promise((r) => setTimeout(r, 3500));
      if (dead) return null;
    }
    const { rad, mass } = starShape(star.rad, star.mass);
    const c = cores();
    const left = k ? ` About ${mins((n - k + 1) * pace)} to go.` : "";
    const running = run(lc, rad, mass, o.floor, (done, total, s, g) => {
      sayLive(`${head}Trying ${int(total)} possible [[orbit|orbits]] on ${c} of your computer's cores: ${Math.floor((100 * done) / total)}%.${left}`);
      periodogram(plot, s, g, o.floor, null);
    });
    job = running;
    const r = await running.done;
    if (dead) return null;
    const v = r.verdict;
    periodogram(plot, r.stat, r.g, o.floor, v.dip ? v.dip.period : null);
    if (v.dip) foldPlot(fold, lc, v.dip.period, v.dip.t0, v.dip.dur_h / 24);
    say(head + (v.pass
      ? `TIC ${star.tic} cleared [[bar|the bar]]: a dip every ${v.dip!.period.toFixed(3)} days, ${Math.round(v.dip!.depth_ppm)} [[ppm]] deep, [[score]] ${v.dip!.snr.toFixed(1)}. Tessa checks it next.`
      : `${why(v, o.floor, star.tic)} Most stars show no planet we can see, so this is a real result.`));
    return r;
  }

  function post(star: SearchStar, r: Run, who: string) {
    const v = r.verdict;
    fetch("/api/search/result", { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ tic: star.tic, handle: who, pass: v.pass, dip: v.dip, topBump: v.bump?.snr ?? v.topBump, nFreq: r.g.freq.length, seconds: r.seconds }) })
      .catch(() => {});
  }
  function again() {
    const b = el("button", { type: "button", class: "u-btn" }, "Search more stars");
    b.addEventListener("click", () => o.onAgain());
    return b;
  }
  function share(n: number, found: number) {
    const text = n > 1
      ? `I searched ${n} stars for planets in my browser with Tessa${found ? `, and ${found} cleared the bar!` : "."}`
      : `I searched a star for planets in my browser with Tessa${found ? ": it cleared the bar!" : "."}`;
    return el("a", { class: "u-btn", href: `https://x.com/intent/post?text=${encodeURIComponent(text)}&url=${encodeURIComponent(`${location.origin}/search`)}`, target: "_blank", rel: "noopener" }, "Share");
  }
  function dispose() { dead = true; state = null; job?.stop(); node.remove(); }
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
