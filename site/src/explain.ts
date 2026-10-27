// Ask Tessa's pictures: small, animated, in the site's own style, for the questions people ask most. Three reuse the
// tour's scenes (the four ways to find a planet, the fold, the upside-down test); the funnel, a planet's size and
// the steps are drawn here. Numbers come from the same files as the tour.
import { getPlanets, getStar, getSummary } from "./data";
import { method } from "./methods";
import type { Method, Show } from "./methods";
import { facts, scene } from "./tourcards";
import type { Ctx } from "./tourcards";
import { el, int } from "./util";

export type Topic = "ways_to_find" | "upside_down" | "funnel" | "fold" | "size" | "steps";
const CAPTION: Record<Topic, string> = {
  ways_to_find: "Four ways to find a planet", upside_down: "The upside-down test", funnel: "From stars to candidates",
  fold: "The fold", size: "How big", steps: "How I search",
};
const WAYS: [Method, string][] = [["dip", "The dip"], ["wobble", "The wobble"], ["bent", "Bent light"], ["photo", "Photographs"]];

// the tour's context: the map, the summary, and the case file of my strongest find
let ctx: Promise<Ctx | null> | null = null;
function tourCtx(): Promise<Ctx | null> {
  ctx ??= Promise.all([getPlanets(), getSummary()]).then(async ([p, s]) => {
    const i = p.kind.indexOf(2);
    const star = i >= 0 ? await getStar(Number(p.host[i].replace(/^TIC /, ""))).catch(() => null) : null;
    return star ? { p, s, star, host: `TIC ${star.tic}` } : null;
  }).catch(() => null);
  return ctx;
}

// plays a scene through its steps on its own clock, then holds the frame at `until` (by default its last step);
// stops once it leaves the page
function play(sh: Show, until?: number): HTMLElement {
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const end = until ?? sh.beats[sh.beats.length - 1].at + 2;
  const t0 = performance.now() - (still ? end * 1000 : 0);
  let seen = false;
  const frame = (now: number) => {
    if (!sh.node.isConnected) {
      if (seen) { sh.dispose(); return; }
    } else seen = true;
    const t = Math.min((now - t0) / 1000, end);
    let k = 0;
    while (k < sh.beats.length - 1 && t >= sh.beats[k + 1].at) k++;
    sh.node.querySelectorAll<HTMLElement>(".later").forEach((e) => e.classList.toggle("show", k >= Number(e.dataset.from)));
    sh.draw(t, k);
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
  return sh.node;
}

const svgNode = (w: number, h: number, body: string) => {
  const box = el("div", { class: "x-svg" });
  box.innerHTML = `<svg viewBox="0 0 ${w} ${h}" role="img">${body}</svg>`;
  return box;
};
const mono = (x: number, y: number, s: string, fill = "#9C9B95", anchor = "start", size = 11) =>
  `<text x="${x}" y="${y}" fill="${fill}" font-family="Courier Prime, monospace" font-size="${size}" text-anchor="${anchor}">${s}</text>`;

// stars searched, down to candidates: each step's bar on a log scale, or the last would vanish
function funnel(c: Ctx): HTMLElement {
  const f = facts(c);
  const rows: [number, string][] = [[f.searched, "stars searched"], [f.pass, "dips that cleared the bar"],
    [f.model, "read by Muse Spark"], [f.found, f.found === 1 ? "candidate" : "candidates"]];
  const top = Math.log10(Math.max(f.searched, 10));
  return svgNode(400, 150, rows.map(([n, label], i) => {
    const w = Math.max((Math.log10(Math.max(n, 1)) / top) * 230, 4), y = 14 + i * 34;
    const last = i === rows.length - 1;
    return `<rect x="0" y="${y}" width="${w.toFixed(1)}" height="14" fill="${last ? "#5eead4" : "#3f4775"}"/>`
      + mono(w + 8, y + 11, int(n), last ? "#5eead4" : "#ECEBE6", "start", 12)
      + mono(w + 8 + int(n).length * 7.4 + 8, y + 11, label);
  }).join(""));
}

// a planet next to Earth (and Neptune or Jupiter when it is bigger), against the edge of its star
async function size(tic: number): Promise<HTMLElement | null> {
  const page = await getStar(tic).catch(() => null);
  const rp = page?.checks?.size?.rp_rearth, rs = page?.checks?.star?.rad;
  if (!rp) return null;
  const ref: [number, string][] = [[1, "Earth"], ...(rp > 2 ? [[rp > 6 ? 11.2 : 3.9, rp > 6 ? "Jupiter" : "Neptune"] as [number, string]] : [])];
  const big = Math.max(rp, ...ref.map((r) => r[0]));
  const k = 34 / big;  // the largest disc gets a 34 px radius
  const discs: [number, string, string][] = [[rp, `TIC ${tic}'s planet`, "#5eead4"], ...ref.map(([r, n]) => [r, n, "#c4c3bd"] as [number, string, string])];
  let x = 40;
  const body = discs.map(([r, name, col]) => {
    const cx = x + r * k;
    x = cx + r * k + 40;
    return `<circle cx="${cx.toFixed(1)}" cy="78" r="${Math.max(r * k, 1.5).toFixed(1)}" fill="${col}" opacity="0.9"/>`
      + mono(cx, 136, name, col === "#5eead4" ? "#5eead4" : "#9C9B95", "middle") + mono(cx, 150, `${r < 10 ? r.toFixed(1) : int(r)} × Earth`, "#9C9B95", "middle", 10);
  }).join("");
  // the star's edge, at the same scale (a star is about 109 Earths wide per solar radius)
  const R = (rs ?? 0.5) * 109 * k;
  const star = `<circle cx="${(x + R - 12).toFixed(0)}" cy="78" r="${R.toFixed(0)}" fill="#ffd9a8" opacity="0.16"/>`
    + mono(Math.min(x + 4, 392), 20, "its star's edge", "#9C9B95", "start", 10);
  return svgNode(400, 158, star + body);
}

// the whole search, as the steps it really runs in
function steps(): HTMLElement {
  const S: [string, string][] = [["Pick the stars", "small, cool, nearby, never searched"], ["Search for dips", "hundreds of thousands of orbits a star"],
    ["Beat the bar", "set by searching upside down"], ["Rule out lookalikes", "eclipsing stars, spots, neighbours"],
    ["Write the case file", "Muse Spark reads every check"], ["Code decides", "a candidate needs a prediction to test"]];
  const row = 38;
  return svgNode(400, 12 + S.length * row, `<line x1="7" y1="14" x2="7" y2="${14 + (S.length - 1) * row}" stroke="#343b5c"/>`
    + S.map(([a, b], i) => `<circle cx="7" cy="${14 + i * row}" r="4" fill="${i === S.length - 1 ? "#5eead4" : "#3f4775"}"/>`
      + `<text x="22" y="${18 + i * row}" fill="#ECEBE6" font-family="Archivo, sans-serif" font-size="14">${a}</text>`
      + mono(22, 33 + i * row, b)).join(""));
}

// the four ways, one scene at a time, switched by plain links under it
function ways(): HTMLElement {
  const box = el("div", { class: "x-ways" });
  const stage = el("div");
  const links = el("div", { class: "x-links" });
  const pick = (m: Method) => {
    links.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.m === m)));
    stage.replaceChildren(play(method(m)));
  };
  for (const [m, label] of WAYS) {
    const b = el("button", { type: "button", "data-m": m }, label);
    b.addEventListener("click", () => pick(m));
    links.append(b);
  }
  box.append(stage, links);
  pick("dip");
  return box;
}

export async function explainFigure(topic: Topic, tic: number | null): Promise<HTMLElement | null> {
  let body: HTMLElement | null = null;
  if (topic === "ways_to_find") body = ways();
  else if (topic === "steps") body = steps();
  else if (topic === "size") body = tic ? await size(tic) : null;
  else {
    const c = await tourCtx();
    if (!c) return null;
    // the tour's odds stop goes on to the funnel; for the upside-down test it stops at the bar
    body = topic === "funnel" ? funnel(c) : topic === "fold" ? play(scene("fold", c)) : play(scene("odds", c), 13);
  }
  return body && el("figure", { class: "x-fig" }, el("figcaption", {}, CAPTION[topic]), body);
}
