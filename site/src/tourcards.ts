// The tour's cards: what Tessa did, one sentence at a time, each with its picture. Every number comes from the
// data files (summary.json and the candidate's case file), so the words stay true after the next batch.
import type { Planets, StarPage, Summary } from "./data";
import { openAsk } from "./ask";
import { axes, fit, method, smooth, trace } from "./methods";
import type { Beat, Method, Show } from "./methods";
import type { Group } from "./tour";
import { el, int } from "./util";

export type CardKind = "hook" | "dip" | "fold" | "odds" | "casefile" | "predict" | "end";
export interface Ctx { p: Planets; s: Summary; star: StarPage | null; host: string }

const css = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const HACKATHON = Date.UTC(2026, 9, 26);  // building opened Oct 26
const words = (k: number) => ["No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"][k] ?? int(k);
const sig2 = (n: number) => int(Number(n.toPrecision(2)));

// plain definitions for the words a newcomer may not know; [[term]] or [[term|shown words]] in a sentence marks one,
// and hovering or tapping it shows the definition
const GLOSSARY: Record<string, string> = {
  ppm: "Parts per million. 326 ppm means the star dims by 0.0326%, about a third of a thousandth.",
  "light curve": "A star's brightness, measured again and again over time.",
  sector: "TESS watches one long strip of sky for about 27 days at a time. Each strip is a sector.",
  candidate: "A signal that passed every check but hasn't been confirmed as a planet yet.",
  "case file": "Muse Spark's written report on one dip: what each check found, and its verdict.",
  "Muse Spark": "Meta's AI model. Tessa uses it to read the evidence on each dip and explain it.",
  GPU: "A graphics chip. It runs thousands of small calculations at once, which suits a planet search.",
  score: "How far the dip stands out from the star's normal flicker. Higher is clearer.",
  TESS: "NASA's Transiting Exoplanet Survey Satellite. Since 2018 it has watched nearly the whole sky for dips.",
  orbit: "One trip of a planet around its star. Here, one orbit is also the planet's year.",
  "ground window": "A night when the star is up, the sky is dark, and a dip is due, at one observatory.",
};
export function rich(text: string): Node[] {
  const out: Node[] = [];
  const re = /\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g;
  let at = 0;
  for (let m = re.exec(text); m; m = re.exec(text)) {
    out.push(document.createTextNode(text.slice(at, m.index)));
    out.push(el("span", { class: "term", tabindex: 0, "data-def": GLOSSARY[m[1]] ?? "" }, m[2] ?? m[1]));
    at = re.lastIndex;
  }
  out.push(document.createTextNode(text.slice(at)));
  return out;
}

// the numbers the cards quote
function facts({ p, s, star }: Ctx) {
  const searched = Number(s.searched_total ?? s.survey_searched ?? 0);
  const g1 = (s.gate1_total ?? s.survey_gate1 ?? {}) as { pass?: number; flipped_pass?: number };
  const gate = (star && (s[`${star.batch}_gate1`] as { floor: number; false_pass_rate: number } | undefined)) || { floor: 0, false_pass_rate: 0.01 };
  const found = s.by_status.candidate;
  const inv = s.invented ?? { dossiers: 0, with_invented_numbers: 0 };
  const eph = star?.when?.ephemeris;
  return {
    searched, found, pass: g1.pass ?? 0, luck: g1.flipped_pass ?? 0, model: s.dossiers, floor: gate.floor,
    byLuck: Math.round(1 / (gate.false_pass_rate || 0.01)),
    days: Math.min(Math.max(Math.ceil((Date.now() - HACKATHON) / 864e5), 1), 10),
    waiting: p.counts.waiting, exact: inv.dossiers - inv.with_invented_numbers, files: inv.dossiers,
    perFile: s.dossiers ? s.costs_usd.vetting / s.dossiers : 0,
    bill: Object.values(s.costs_usd).reduce((a, b) => a + b, 0),
    depth: Math.round(star?.checks?.signal?.depth_ppm ?? 0), hours: eph?.t14_h ?? star?.checks?.signal?.dur_h ?? 0,
    period: eph?.period ?? star?.ephem.period ?? 0, rp: star?.checks?.size?.rp_rearth ?? null,
  };
}

const show = (node: HTMLElement, beats: Beat[], draw: Show["draw"] = () => {}, dispose = () => {}): Show => ({ node, beats, draw, dispose });
const later = <T extends HTMLElement>(node: T, from: number) => { node.classList.add("later"); node.dataset.from = String(from); return node; };

// ---- 1. what Tessa found
function hook(ctx: Ctx): Show {
  const f = facts(ctx);
  const one = f.found === 1;
  return show(el("div", {}), [
    { at: 0, text: `In ${f.days} day${f.days > 1 ? "s" : ""}, Tessa searched ${int(f.searched)} nearby stars.` },
    { at: 3, text: `It found ${one ? "one [[candidate|possible new world]]" : `${words(f.found).toLowerCase()} [[candidate|possible new worlds]]`}: ${ctx.host}.` },
    { at: 6.5, text: `${f.rp ? `About ${f.rp.toFixed(1)} times Earth's width, with` : "With"} a year just ${Math.round(f.period * 24)} hours long, `
      + `${int(Math.round((ctx.star?.checks?.star?.distance ?? 0) * 3.26156))} light-years from us.` },
  ]);
}

// ---- 3. the fold: years of noisy light, folded on the right period, and the dip rises out of the noise
function fold(ctx: Ctx): Show {
  const f = facts(ctx), st = ctx.star!;
  const { t, y_ppm: y, sector } = st.series;
  const P = f.period, t0 = st.when?.ephemeris.t0_btjd ?? st.ephem.t0, half = Math.max(...st.fold.x_h.map(Math.abs));
  const sectors = [...new Set(sector)].sort((a, b) => a - b);
  const span = sectors.map((s) => { const ts = t.filter((_, i) => sector[i] === s); return [Math.min(...ts), Math.max(...ts)]; });
  const total = span.reduce((a, [lo, hi]) => a + hi - lo, 0);
  const sd = Math.sqrt(y.reduce((a, v) => a + v * v, 0) / y.length);
  const ph = t.map((ti) => ((((ti - t0) / P + 0.5) % 1 + 1) % 1 - 0.5) * P * 24);
  const years = Math.round((t[t.length - 1] - t[0]) / 365.25);
  const c = el("canvas", { class: "m-plot" });
  const one = sectors[Math.min(3, sectors.length - 1)];
  return show(el("div", { class: "m-show" }, c), [
    { at: 0, text: `This is ${ctx.host}'s [[light curve|light]]: ${int(t.length)} measurements from ${sectors.length} [[TESS]] [[sector|sectors]], spread over ${years} years.` },
    { at: 4.5, text: "In any one stretch, the dips are lost in the noise." },
    { at: 8.5, text: `Tessa tries thousands of possible [[orbit|orbits]]. At ${P.toFixed(2)} days, every dip lines up.` },
    { at: 13, text: `Stacked together, the dip rises out of the noise: ${f.depth} [[ppm|parts per million]], ${f.hours.toFixed(1)} hours long.` },
    { at: 18, text: `Tessa ran this search on ${int(f.searched)} nearby stars, each in under a second on a [[GPU]].` },
  ], (tt, k) => {
    const m = smooth(9, 11.5, tt);  // 0 = laid out in time, 1 = folded on the period
    const { g, w, B, T, L } = axes(c, "brightness", m < 0.5 ? `time: ${sectors.length} TESS sectors →` : "hours from the middle of the dip →");
    const Y = (v: number) => T + 6 + ((3.2 * sd - v) / (6.4 * sd)) * (B - T - 12);
    const gap = 6, room = w - L - 4 - gap * (sectors.length - 1);
    const xs: number[] = [];  // each sector's left edge in time order, side by side without the gaps between them
    let at = L + 2;
    for (const [lo, hi] of span) { xs.push(at); at += ((hi - lo) / total) * room + gap; }
    const xRaw = (i: number) => { const k2 = sectors.indexOf(sector[i]); return xs[k2] + ((t[i] - span[k2][0]) / total) * room; };
    const xFold = (h: number) => L + 2 + ((h + half) / (2 * half)) * (w - L - 4);
    for (let i = 0; i < t.length; i++) {
      const inside = Math.abs(ph[i]) <= half;
      const a = k === 1 && m === 0 ? (sector[i] === one ? 0.85 : 0.12) : inside ? 0.55 : 0.55 * (1 - m);
      if (a < 0.02) continue;
      g.fillStyle = `rgba(236,235,230,${a})`;
      const x = m === 0 ? xRaw(i) : xRaw(i) + (xFold(inside ? ph[i] : ph[i] > 0 ? half * 1.3 : -half * 1.3) - xRaw(i)) * m;
      g.fillRect(x - 0.75, Y(Math.max(Math.min(y[i], 3.2 * sd), -3.2 * sd)) - 0.75, 1.5, 1.5);
    }
    if (m === 0) {
      g.fillStyle = css("--muted");
      sectors.forEach((s, i) => g.fillText(`S${s}`, xs[i], B - 4));
    }
    const line = smooth(13, 14.5, tt);
    if (line > 0) {
      g.globalAlpha = line; g.strokeStyle = "#5eead4"; g.lineWidth = 2.5; g.beginPath();
      st.fold.x_h.forEach((h, i) => { const x = xFold(h), yy = Y(st.fold.y_ppm[i]); if (i) g.lineTo(x, yy); else g.moveTo(x, yy); });
      g.stroke();
      g.fillStyle = css("--ink"); g.fillText(`the dip: ${f.depth} ppm`, xFold(0) + 14, Y(-f.depth) + 22);
      g.globalAlpha = 1;
    }
  });
}

// ---- 4. the odds: the upside-down test, the bar it sets, and the funnel
function odds(ctx: Ctx): Show {
  const f = facts(ctx), st = ctx.star!;
  const c = el("canvas", { class: "m-plot" });
  const sd = Math.max(...st.fold.y_ppm.map(Math.abs));
  // one dot per star searched, in a grid; which ones pass each step is just a count, the order is shuffled
  let seed = 11;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const order = Array.from({ length: f.searched }, (_, i) => i).sort(() => rnd() - 0.5);
  const rank = new Int32Array(f.searched);
  order.forEach((s, i) => { rank[s] = i; });
  // the survivor sits in the middle of the grid, so it reads as the point and not a stray mark
  let mid = 0;
  const centre = (cols: number) => {
    const m = Math.floor(Math.ceil(f.searched / cols) / 2) * cols + Math.floor(cols / 2);
    if (m === mid) return;
    const top = rank.indexOf(0);  // whichever star holds rank 0 swaps places with the centre one
    [rank[top], rank[m]] = [rank[m], 0];
    mid = m;
  };
  const one = `1 in ${sig2(f.searched / Math.max(f.found, 1))}`;
  return show(el("div", { class: "m-show" }, c), [
    { at: 0, text: "Noise can fake a dip. So Tessa also searches every light curve upside down." },
    { at: 4.5, text: `A real dip can't show up as a bump. Upside down, this star's best signal [[score|scores]] ${ctx.star!.gate1.flipped_snr.toFixed(1)}; the right way up, ${ctx.star!.gate1.dip_snr.toFixed(1)}.` },
    { at: 9.5, text: `Doing that on every star sets the bar at ${f.floor}: only 1 star in ${f.byLuck} clears it by luck.` },
    { at: 14, text: `Of ${int(f.searched)} stars searched, ${f.pass} cleared the bar.` },
    { at: 19, text: `Code checks ruled out ${f.pass - f.model} of those: stars that eclipse each other, or flicker on their own. ${f.model} went to [[Muse Spark]], and ${f.found === 1 ? "one" : int(f.found)} survived: about ${one} stars.` },
  ], (tt, k) => {
    if (k <= 2) {
      const flip = k === 0 ? smooth(1.5, 3, tt) : 1;
      const { g, w, h, B, T, L } = axes(c, "brightness", "hours from the middle of the dip →");
      const plotW = k === 0 ? w - L : (w - L) * 0.45, Y = (v: number) => (T + B) / 2 - (v / (sd * 2.4)) * (B - T);
      const xs = st.fold.x_h, half = Math.max(...xs.map(Math.abs)), X = (x: number) => L + 4 + ((x + half) / (2 * half)) * (plotW - 8);
      g.strokeStyle = "#5eead4"; g.lineWidth = 2; g.globalAlpha = 0.35; g.beginPath();
      xs.forEach((x, i) => { const yy = Y(st.fold.y_ppm[i]); if (i) g.lineTo(X(x), yy); else g.moveTo(X(x), yy); }); g.stroke();
      g.globalAlpha = 1; g.strokeStyle = css("--ink"); g.beginPath();
      xs.forEach((x, i) => { const yy = Y(st.fold.y_ppm[i] * (1 - 2 * flip)); if (i) g.lineTo(X(x), yy); else g.moveTo(X(x), yy); }); g.stroke();
      g.fillStyle = css("--muted");
      g.fillText("the right way up", X(0) + 30, Y(-sd) + 20);
      if (flip > 0.5) g.fillText("upside down: the dip becomes a bump", X(0) - 40, Y(sd) - 14);
      if (k >= 1) {  // the two scores as bars, and the bar Tessa sets
        const bx = L + (w - L) * 0.52, bw = w - bx - 8, max = Math.max(st.gate1.dip_snr, f.floor) * 1.25, W2 = (v: number) => (v / max) * bw;
        const row = (yy: number, v: number, name: string, color: string) => {
          g.fillStyle = color; g.fillRect(bx, yy, W2(v), 14);
          g.fillStyle = css("--ink-2"); g.fillText(`${name}: ${v.toFixed(1)}`, bx, yy - 5);
        };
        row(h * 0.28, st.gate1.dip_snr, "right way up", "#5eead4");
        row(h * 0.58, st.gate1.flipped_snr, "upside down", css("--muted"));
        if (k >= 2) {
          const fx = bx + W2(f.floor);
          g.strokeStyle = css("--ink"); g.setLineDash([3, 3]); g.beginPath(); g.moveTo(fx, h * 0.14); g.lineTo(fx, h * 0.72); g.stroke(); g.setLineDash([]);
          g.fillStyle = css("--ink"); g.fillText(`the bar: ${f.floor}`, fx - 30, h * 0.82);
        }
      }
      return;
    }
    // the funnel: every star searched is a dot
    const { g, w, h } = fit(c);
    const cols = Math.ceil(Math.sqrt(f.searched * (w / h))), size = Math.min(w / cols, h / Math.ceil(f.searched / cols));
    centre(cols);
    const passIn = smooth(14.5, 16, tt), modelIn = smooth(19.5, 21, tt);
    for (let i = 0; i < f.searched; i++) {
      const r = rank[i];
      const passed = r < f.pass, toModel = r < f.model, found = r < f.found;
      let color = `rgba(236,235,230,${0.5 - 0.4 * passIn})`;
      if (passed) color = `rgba(236,235,230,${0.5 + 0.5 * passIn - (toModel ? 0 : 0.75 * modelIn)})`;
      if (found && modelIn > 0.5) color = "#5eead4";
      g.fillStyle = color;
      const x = (i % cols) * size, y = Math.floor(i / cols) * size;
      const d = found && modelIn > 0.5 ? size * 2.2 : Math.max(size * 0.62, 1);
      g.fillRect(x + (size - d) / 2, y + (size - d) / 2, d, d);
    }
    g.fillStyle = css("--muted"); g.fillText("each dot is a star searched", 0, h - 4);
    if (modelIn > 0.5) {
      const x = (mid % cols) * size, y = Math.floor(mid / cols) * size;
      g.fillStyle = "#5eead4"; g.fillText(ctx.host, x + size * 2.5, y + size * 1.2);
    }
  });
}

// ---- 5. the case file: the bottleneck, then Muse Spark's checks and verdict
const PLAIN: [string, string][] = [["odd_even", "odd and even dips match"], ["secondary", "no second, fainter eclipse"],
  ["sectors", "same depth in every sector"], ["size", "planet-sized"], ["duration", "lasts as long as it should"],
  ["known", "not a known star or planet"], ["shape", "flat-bottomed, not V-shaped"], ["neighbours", "on this star, not a neighbour"],
  ["centroid", "on this star, not a neighbour"], ["phase_curve", "the star doesn't vary at the period"]];
function casefile(ctx: Ctx): Show {
  const f = facts(ctx), d = ctx.star?.dossier;
  // the checks in plain words, once each; the model's own reason for each is on hover
  const seen = new Set<string>();
  const rows = (d?.reasons ?? []).flatMap((r) => {
    const name = PLAIN.find(([k]) => r.check.startsWith(k))?.[1];
    if (!name || seen.has(name)) return [];
    seen.add(name);
    return [el("li", { class: "t-check", title: r.why }, el("span", { class: "t-tick", "aria-hidden": "true" }, "✓"), el("b", {}, name))];
  }).slice(0, 8);
  const list = later(el("ul", { class: "t-checks" }, ...rows), 1);
  const quote = later(el("blockquote", { class: "t-quote" }, d?.public_summary ?? ""), 2);
  return show(el("div", { class: "m-show" }, list, quote), [
    { at: 0, text: `Finding dips is fast. Checking them is the slow part: ${int(f.waiting)} [[TESS]] [[candidate|candidates]] are still waiting for someone to look.` },
    { at: 5.5, text: "For every dip that survives the code checks, [[Muse Spark]] reads the evidence and writes a [[case file]]." },
    { at: 12, text: `Its verdict on ${ctx.host}:` },
    { at: 17, text: `Every number it writes is checked against the data: ${f.exact} of ${f.files} case files were exact, and the slip was caught. Code makes the final call.` },
    { at: 22.5, text: `A case file costs about $${f.perFile.toFixed(4)}. The model work for the whole survey: $${f.bill.toFixed(2)}.` },
  ], (tt, k) => {
    rows.forEach((r, i) => r.classList.toggle("on", k >= 1 && tt >= 5.9 + i * 0.55));
  });
}

// ---- 6. the prediction: what the next data must show for Tessa to be right
function predict(ctx: Ctx): Show {
  const f = facts(ctx), st = ctx.star!, w2 = st.when;
  const c = el("canvas", { class: "m-plot short" });
  const now = Date.now();
  const nights = (w2?.ground ?? []).filter((x) => Date.parse(`${x.mid_utc.replace(" ", "T")}Z`) > now);
  const next = nights[0];
  const err = Math.round(w2?.timing_err_now_min ?? 5);
  const when = next ? new Date(`${next.mid_utc.replace(" ", "T")}Z`) : null;
  const day = (d: Date) => d.toLocaleString("en-GB", { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "UTC" });
  const tess = w2?.next_tess;
  const pub = tess ? new Date(`${tess.public_from}T00:00:00Z`).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" }) : "";
  const countdown = el("p", { class: "t-countdown later", "data-from": "1" });
  return show(el("div", { class: "m-show" }, c, countdown), [
    { at: 0, text: `For Tessa to be right, the next data needs to show the same dip: ${f.depth} [[ppm]], ${f.hours.toFixed(1)} hours long, every ${(f.period * 24).toFixed(1)} hours.` },
    ...(when ? [{ at: 5.5, text: `The next dip a telescope on the ground can watch is due ${day(when)} UTC, give or take ${err} minutes, at ${next.site}.` }] : []),
    { at: when ? 11 : 5.5, text: `Telescopes on the ground get ${nights.length} [[ground window|chances]] in the next 90 days${tess ? `, and [[TESS]] looks again in [[sector]] ${tess.sector}, with its data public on ${pub}` : ""}.` },
    ...(st.observing ? [{ at: when ? 16.5 : 11, text: `Muse Spark's note for observers: ${st.observing.confirmed_by}` }] : []),
  ], (tt, k) => {
    const { g, w, B, T, L } = axes(c, "brightness", "hours from the predicted time →");
    const xs = st.fold.x_h, half = Math.max(...xs.map(Math.abs)), sd = Math.max(...st.fold.y_ppm.map(Math.abs)) * 1.6;
    const X = (x: number) => L + 4 + ((x + half) / (2 * half)) * (w - L - 8), Y = (v: number) => T + 8 + ((sd * 0.4 - v) / sd) * (B - T - 16);
    // the expected dip: a box of the predicted depth and length, with the timing error as a shaded band
    const d = f.hours / 2, e = err / 60;
    g.fillStyle = "rgba(94,234,212,0.12)"; g.fillRect(X(-d - e), T, X(-d + e) - X(-d - e), B - T); g.fillRect(X(d - e), T, X(d + e) - X(d - e), B - T);
    g.strokeStyle = "#5eead4"; g.lineWidth = 2; g.setLineDash([5, 4]); g.beginPath();
    g.moveTo(X(-half), Y(0)); g.lineTo(X(-d), Y(0)); g.lineTo(X(-d), Y(-f.depth)); g.lineTo(X(d), Y(-f.depth)); g.lineTo(X(d), Y(0)); g.lineTo(X(half), Y(0));
    g.stroke(); g.setLineDash([]);
    g.fillStyle = css("--ink"); g.fillText(`what Tessa expects: ${f.depth} ppm for ${f.hours.toFixed(1)} h`, X(-half) + 4, Y(0) - 10);
    g.fillStyle = css("--muted"); g.fillText(`shaded: ± ${err} min`, X(d + e) + 8, Y(-f.depth) + 4);
    if (when) {
      const ms = when.getTime() - Date.now();
      countdown.textContent = ms > 0 ? `Next dip due in ${Math.floor(ms / 36e5)} h ${Math.floor((ms % 36e5) / 6e4)} min, watchable from ${next.site}` : "Due now";
    }
    void tt; void k;
  });
}

// ---- 7. your turn
function end(ctx: Ctx, onSearch: () => void): Show {
  const f = facts(ctx);
  const share = `${location.origin}/?at=${encodeURIComponent(ctx.host)}`;
  const text = `Tessa searched ${int(f.searched)} nearby stars in public NASA TESS data and found a possible new world: ${ctx.host}.`;
  const actions = later(el("div", { class: "u-actions" },
    el("button", { type: "button", class: "u-btn", onclick: () => onSearch() }, "Search a star"),
    el("button", { type: "button", class: "u-btn", onclick: () => openAsk() }, "Ask Tessa"),
    el("a", { class: "u-btn", href: `https://x.com/intent/post?text=${encodeURIComponent(text)}&url=${encodeURIComponent(share)}`, target: "_blank", rel: "noopener" }, "Share this world")), 1);
  return show(el("div", { class: "m-show" }, actions), [
    { at: 0, text: `That's Tessa so far: ${int(f.searched)} stars searched, ${f.found === 1 ? "one possible new world" : `${int(f.found)} possible new worlds`}, every step in the open.` },
    { at: 3.5, text: "Now search a star yourself: your computer runs the same search on a star nobody has searched. Or ask Tessa about any star or planet here." },
  ]);
}

// the method cards, for the dip stop and its "other ways" chips
const METHODS: [Method, string, Group][] = [["dip", "The dip", "transit"], ["wobble", "The wobble", "rv"], ["bent", "Bent light", "lens"], ["photo", "Photographs", "other"]];

export interface CardOpts {
  kind: CardKind; ctx: Ctx; index: number; total: number;
  onNext(): void; onBack(): void; onLeave(): void; onFocus(g: Group | null): void; onSearch(): void;
}
// One stop of the tour: a bar, the stop's progress, a headline, the picture and the sentence, with back and next.
// The sentences play on their own clock; tapping the picture or the arrows moves a step, past the ends moves a stop.
export function tourCard(o: CardOpts): { node: HTMLElement; dispose(): void } {
  const f = facts(o.ctx);
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const head: Record<CardKind, [string, string, boolean]> = {
    hook: [String(f.found), f.found === 1 ? "possible new world" : "possible new worlds", false],
    dip: [int(o.ctx.p.counts.by_method.Transit?.planets ?? 0), "planets found this way", false],
    fold: [o.ctx.host, "", false],
    odds: [`1 in ${sig2(f.searched / Math.max(f.found, 1))}`, "stars searched gave a candidate", false],
    casefile: [o.ctx.host, "case file", true],
    predict: [o.ctx.host, "what to look for next", true],
    end: ["Your turn", "", false],
  };
  const [big, sub, spark] = head[o.kind];
  // what each stop is about, named in the header so the reader always knows where they are
  const NAME: Record<CardKind, string> = { hook: "Tessa's find", dip: "How a planet shows itself", fold: "The fold",
    odds: "The odds", casefile: "The case file", predict: "What comes next", end: "Your turn" };

  const leave = el("button", { type: "button", class: "u-bar-btn" }, "Leave");
  leave.addEventListener("click", () => o.onLeave());
  const back = el("button", { type: "button", class: "t-arrow", "aria-label": "Back" }, "‹");
  const fwd = el("button", { type: "button", class: "t-arrow", "aria-label": "Next" }, "›");
  // a thin ring around › fills while the tour waits to move on to the next stop by itself
  const ringEl = el("span", { class: "t-ring", "aria-hidden": "true" });
  ringEl.innerHTML = '<svg viewBox="0 0 44 40"><rect x="1" y="1" width="42" height="38" pathLength="100"/></svg>';
  const ringRect = ringEl.querySelector("rect")!;
  const fwdBox = el("span", { class: "t-fwd" }, fwd, ringEl);
  // one pip per step; the current one fills as the step plays, so it is clear the tour moves on by itself
  const pips = el("span", { class: "t-pips", role: "img" });
  const pause = el("button", { type: "button", class: "t-pause", "aria-pressed": "false" }, "Pause");
  const text = el("p", { class: "t-text", "aria-live": "polite" });
  const box = el("div", { class: "t-show" });
  const chips = o.kind === "dip" ? el("div", { class: "u-chips t-chips" }, el("span", { class: "u-k" }, "Other ways planets are found")) : null;
  // reading order: where you are (the header), the sentence (the one thing to read), then the stop's number for
  // context, then the controls
  const node = el("section", { class: "t-card", "aria-label": "Tour" },
    el("div", { class: "t-side" },
      el("div", { class: "t-head" },
        el("div", { class: "t-top" }, el("span", { class: "t-where" }, el("span", { class: "u-dot", style: "background: var(--ok)" }),
          el("span", { class: "u-of" }, `${o.index + 1} of ${o.total}`), el("b", {}, NAME[o.kind])), leave),
        el("div", { class: "t-prog", "aria-hidden": "true" }, ...Array.from({ length: o.total }, (_, i) => el("span", { class: i <= o.index ? "on" : "" })))),
      el("div", { class: "t-say" }, spark ? el("span", { class: "t-tag", title: "This step is done by Meta's Muse Spark model" }, "Muse Spark") : null, text),
      el("div", { class: "t-stat" }, el("p", { class: "t-big" }, big), sub ? el("p", { class: "t-sub" }, sub) : null, chips),
      el("div", { class: "t-ctrl" }, el("div", { class: "t-nav" }, back, pips, fwdBox), pause)),
    el("div", { class: "t-stage" }, box));

  // Pacing. Each step stays up long enough to read (about 180 words a minute, plus a beat to look at the picture),
  // and never shorter than its animation needs. The picture's clock is stretched to match, so words and pictures
  // stay in step. After the last step the ring fills and the tour moves on, unless it is paused.
  const RING = 6;
  const readTime = (b: Beat) => 2.5 + b.text.replace(/\[\[|\]\]|\|[^\]]*/g, "").split(/\s+/).length / 3;
  let cur: Show, starts: number[] = [], dwell: number[] = [];
  let r0 = 0, k = -1, raf = 0, paused = still, last = 0, moved = false;
  const plan = (beats: Beat[]) => {
    dwell = beats.map((b, i) => (i < beats.length - 1 ? Math.max((beats[i + 1].at - b.at) * 1.25, readTime(b)) : readTime(b) + 1));
    starts = dwell.reduce<number[]>((acc, d, i) => [...acc, i ? acc[i - 1] + dwell[i - 1] : 0], []);
  };
  const showTime = (rt: number) => {  // real seconds into the stop -> the picture's own clock
    const b = cur.beats;
    let i = 0;
    while (i < b.length - 1 && rt >= starts[i + 1]) i++;
    const gap = i < b.length - 1 ? b[i + 1].at - b[i].at : dwell[i];
    return { i, t: b[i].at + (rt - starts[i]) * (gap / dwell[i]) };
  };
  const load = (sh: Show) => {
    cur?.dispose();
    cur = sh;
    box.replaceChildren(sh.node);
    node.classList.toggle("bare", !sh.node.childElementCount);  // no picture: the universe fills the screen
    plan(sh.beats);
    pips.replaceChildren(...sh.beats.map(() => el("span", {}, el("i", {}))));
    k = -1;
    moved = false;
    r0 = performance.now() - (still ? (starts[starts.length - 1] + 30) * 1000 : 0);
  };
  const make = (m?: Method) => {
    switch (o.kind) {
      case "hook": return hook(o.ctx);
      case "dip": return method(m ?? "dip");
      case "fold": return fold(o.ctx);
      case "odds": return odds(o.ctx);
      case "casefile": return casefile(o.ctx);
      case "predict": return predict(o.ctx);
      default: return end(o.ctx, o.onSearch);
    }
  };
  if (chips) {
    for (const [m, label, group] of METHODS) {
      const b = el("button", { type: "button", "aria-pressed": String(m === "dip") }, label);
      b.addEventListener("click", () => {
        chips.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
        o.onFocus(group);
        load(make(m));
      });
      chips.append(b);
    }
  }
  load(make());
  const go = (n: number) => {
    if (n < 0) { o.onBack(); return; }
    if (n >= cur.beats.length) { o.onNext(); return; }
    r0 = performance.now() - starts[n] * 1000;
    k = -1;
  };
  const setPaused = (v: boolean) => { paused = v; pause.setAttribute("aria-pressed", String(v)); pause.textContent = v ? "Play" : "Pause"; };
  setPaused(paused);
  back.addEventListener("click", () => go(k - 1));
  fwd.addEventListener("click", () => go(k + 1));
  pause.addEventListener("click", () => setPaused(!paused));
  box.addEventListener("click", (ev) => { if (!(ev.target as HTMLElement).closest("a,button,.term")) go(k + 1); });
  const key = (ev: KeyboardEvent) => {
    if ((ev.target as HTMLElement).closest("input,textarea,canvas.universe")) return;
    if (ev.key === "ArrowRight") go(k + 1);
    else if (ev.key === "ArrowLeft") go(k - 1);
    else if (ev.key === " " && !(ev.target as HTMLElement).closest("button,a")) { ev.preventDefault(); setPaused(!paused); }
  };
  addEventListener("keydown", key);

  function frame(now: number) {
    raf = requestAnimationFrame(frame);
    if (paused) r0 += now - (last || now);  // a held clock: real time passes, the stop's clock does not
    last = now;
    const rt = (now - r0) / 1000;
    const { i: kk, t } = showTime(Math.max(rt, 0));
    if (kk !== k) {
      k = kk;
      text.replaceChildren(...rich(cur.beats[k].text));
      pips.setAttribute("aria-label", `Step ${k + 1} of ${cur.beats.length}`);
      back.disabled = k === 0 && o.index === 0;
      fwd.classList.toggle("end", k === cur.beats.length - 1);
      fwd.setAttribute("aria-label", k === cur.beats.length - 1 ? (o.index === o.total - 1 ? "Finish the tour" : "Next stop") : "Next");
      box.querySelectorAll<HTMLElement>(".later").forEach((e) => e.classList.toggle("show", k >= Number(e.dataset.from)));
    }
    [...pips.children].forEach((pip, i) => {
      const f = i < k ? 1 : i > k ? 0 : Math.min((rt - starts[i]) / dwell[i], 1);
      (pip.firstElementChild as HTMLElement).style.transform = `scaleX(${f})`;
    });
    // after the last step, the ring fills and the tour moves on (the last stop stays put)
    const wait = k === cur.beats.length - 1 && o.index < o.total - 1 ? (rt - starts[k] - dwell[k]) / RING : -1;
    ringEl.classList.toggle("on", wait > 0);
    ringRect.style.strokeDashoffset = String(100 - Math.min(Math.max(wait, 0), 1) * 100);
    if (wait >= 1 && !moved) { moved = true; o.onNext(); }
    cur.draw(t, k);
  }
  raf = requestAnimationFrame(frame);
  return {
    node,
    dispose() { cancelAnimationFrame(raf); removeEventListener("keydown", key); cur.dispose(); node.remove(); },
  };
}
