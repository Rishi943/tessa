// The star page's charts, drawn by hand: one series each, hairline axes, a tooltip on hover and the
// numbers behind every chart in a table underneath.
import type { StarPage } from "./data";
import { css, el, fix, int, pct, svg } from "./util";

const NS = "http://www.w3.org/2000/svg";

function niceTicks(lo: number, hi: number, count = 5): number[] {
  const span = hi - lo;
  const raw = span / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) ?? raw;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

const compact = (v: number) => (Math.abs(v) >= 1000 ? `${+(v / 1000).toFixed(1)}K` : `${+v.toFixed(0)}`);

function dataTable(head: string[], rows: (string | number)[][], label: string): HTMLElement {
  return el("details", { class: "data-toggle" }, el("summary", {}, label),
    el("table", {}, el("thead", {}, el("tr", {}, ...head.map((h) => el("th", {}, h)))),
      el("tbody", {}, ...rows.map((r) => el("tr", {}, ...r.map((c) => el("td", {}, c)))))));
}

function watchTheme(draw: () => void) {
  new MutationObserver(draw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
}

// Every sector side by side (the gaps between them are months, so they are cut out), binned to 30 min,
// with a tick under each predicted transit.
export function lightCurve(page: StarPage): HTMLElement {
  const { t, y_ppm, sector } = page.series;
  const { period, t0 } = page.ephem;
  const depth = page.checks.signal.depth_ppm as number;
  const blocks: { s: number; i0: number; i1: number; t0: number; t1: number }[] = [];
  for (let i = 0; i < t.length; i++) {
    const b = blocks[blocks.length - 1];
    if (b && b.s === sector[i]) { b.i1 = i; b.t1 = t[i]; } else blocks.push({ s: sector[i], i0: i, i1: i, t0: t[i], t1: t[i] });
  }
  const sorted = [...y_ppm].sort((a, b) => a - b);
  const q = (p: number) => sorted[Math.floor(p * (sorted.length - 1))];
  const lo = Math.min(q(0.003), -1.6 * depth), hi = Math.max(q(0.997), 0.6 * depth);

  const box = el("div", { class: "chart" });
  const canvas = el("canvas", { role: "img", "aria-label": `Brightness of TIC ${page.tic} over ${blocks.length} TESS sectors, in parts per million` });
  const tip = el("div", { class: "tip", hidden: true });
  box.append(canvas, tip, el("p", { class: "cap" },
    `Brightness in parts per million, 30-minute bins, ${blocks.length} sectors of about 27 days each. Ticks mark where the dip should fall every ${fix(period, 4)} days.`));

  const H = 260, padL = 44, padR = 8, padT = 10, padB = 34, GAP = 10;
  let xs: Float32Array = new Float32Array(0);
  let ys: Float32Array = new Float32Array(0);
  function draw() {
    const W = canvas.clientWidth || 800;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = W * dpr; canvas.height = H * dpr; canvas.style.height = `${H}px`;
    const ctx = canvas.getContext("2d")!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);
    const plotW = W - padL - padR - GAP * (blocks.length - 1);
    const span = blocks.reduce((a, b) => a + (b.t1 - b.t0), 0);
    const yOf = (v: number) => padT + ((hi - v) / (hi - lo)) * (H - padT - padB);
    const font = `11px ${css("--sans")}`;
    ctx.font = font;
    ctx.textBaseline = "middle";
    // grid and axis
    ctx.strokeStyle = css("--hair");
    ctx.fillStyle = css("--muted");
    ctx.lineWidth = 1;
    ctx.textAlign = "right";
    for (const v of niceTicks(lo, hi, 5)) {
      const y = Math.round(yOf(v)) + 0.5;
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
      ctx.fillText(compact(v), padL - 6, y);
    }
    ctx.strokeStyle = css("--axis");
    ctx.beginPath(); ctx.moveTo(padL, Math.round(yOf(0)) + 0.5); ctx.lineTo(W - padR, Math.round(yOf(0)) + 0.5); ctx.stroke();
    // points
    xs = new Float32Array(t.length); ys = new Float32Array(t.length);
    let x0 = padL;
    ctx.fillStyle = css("--searched");
    ctx.globalAlpha = 0.75;
    ctx.textAlign = "center";
    for (const b of blocks) {
      const bw = ((b.t1 - b.t0) / span) * plotW;
      for (let i = b.i0; i <= b.i1; i++) {
        xs[i] = x0 + ((t[i] - b.t0) / (b.t1 - b.t0 || 1)) * bw;
        ys[i] = yOf(Math.max(lo, Math.min(hi, y_ppm[i])));
        ctx.fillRect(xs[i] - 1, ys[i] - 1, 2, 2);
      }
      // predicted transits under the block
      ctx.globalAlpha = 1;
      ctx.strokeStyle = css("--candidate");
      for (let k = Math.ceil((b.t0 - t0) / period); t0 + k * period <= b.t1; k++) {
        const x = Math.round(x0 + ((t0 + k * period - b.t0) / (b.t1 - b.t0 || 1)) * bw) + 0.5;
        ctx.beginPath(); ctx.moveTo(x, H - padB + 3); ctx.lineTo(x, H - padB + 9); ctx.stroke();
      }
      ctx.fillStyle = css("--muted");
      if (bw > 22) ctx.fillText(`S${b.s}`, x0 + bw / 2, H - padB + 21);
      ctx.fillStyle = css("--searched");
      ctx.globalAlpha = 0.75;
      x0 += bw + GAP;
    }
    ctx.globalAlpha = 1;
  }
  canvas.addEventListener("pointermove", (ev) => {
    const r = canvas.getBoundingClientRect();
    const mx = ev.clientX - r.left;
    let best = -1, bd = 8;
    for (let i = 0; i < xs.length; i++) {
      const d = Math.abs(xs[i] - mx);
      if (d < bd) { bd = d; best = i; }
    }
    if (best < 0) { tip.hidden = true; return; }
    tip.hidden = false;
    tip.replaceChildren(`Sector ${sector[best]}, day ${fix(t[best], 2)}`, el("br"), `${int(y_ppm[best])} ppm`);
    tip.style.left = `${Math.min(xs[best] + 20, r.width - 150)}px`;
    tip.style.top = `${ys[best] - 10}px`;
  });
  canvas.addEventListener("pointerleave", () => { tip.hidden = true; });
  new ResizeObserver(draw).observe(canvas);
  watchTheme(draw);
  return box;
}

// The dip folded on the period: 60 bins with their error bars
export function foldChart(page: StarPage): HTMLElement {
  const { x_h, y_ppm, err_ppm } = page.fold;
  const depth = page.checks.signal.depth_ppm as number;
  const W = 560, H = 300, padL = 48, padR = 12, padT = 14, padB = 40;
  const xlo = Math.min(...x_h), xhi = Math.max(...x_h);
  const ylo = Math.min(...y_ppm.map((v, i) => v - err_ppm[i]), -1.2 * depth);
  const yhi = Math.max(...y_ppm.map((v, i) => v + err_ppm[i]), 0.4 * depth);
  const X = (v: number) => padL + ((v - xlo) / (xhi - xlo)) * (W - padL - padR);
  const Y = (v: number) => padT + ((yhi - v) / (yhi - ylo)) * (H - padT - padB);
  const yt = niceTicks(ylo, yhi, 5), xt = niceTicks(xlo, xhi, 6);
  const s = svg(`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="The dip folded on its ${fix(page.checks.signal.period, 4)} day period">
    <g class="grid">${yt.map((v) => `<line x1="${padL}" x2="${W - padR}" y1="${Y(v)}" y2="${Y(v)}"/>`).join("")}</g>
    ${yt.map((v) => `<text x="${padL - 6}" y="${Y(v)}" text-anchor="end" dominant-baseline="middle">${compact(v)}</text>`).join("")}
    ${xt.map((v) => `<text x="${X(v)}" y="${H - padB + 16}" text-anchor="middle">${+v.toFixed(1)}</text>`).join("")}
    <text x="${(padL + W - padR) / 2}" y="${H - 6}" text-anchor="middle">hours from the middle of the dip</text>
    <line class="base" x1="${padL}" x2="${W - padR}" y1="${Y(0)}" y2="${Y(0)}"/>
  </svg>`);
  const g = document.createElementNS(NS, "g");
  const color = "var(--candidate)";
  x_h.forEach((x, i) => {
    const line = document.createElementNS(NS, "line");
    line.setAttribute("x1", `${X(x)}`); line.setAttribute("x2", `${X(x)}`);
    line.setAttribute("y1", `${Y(y_ppm[i] - err_ppm[i])}`); line.setAttribute("y2", `${Y(y_ppm[i] + err_ppm[i])}`);
    line.setAttribute("stroke", "var(--axis)"); line.setAttribute("stroke-width", "1.5");
    const dot = document.createElementNS(NS, "circle");
    dot.setAttribute("cx", `${X(x)}`); dot.setAttribute("cy", `${Y(y_ppm[i])}`); dot.setAttribute("r", "4");
    dot.setAttribute("fill", color); dot.setAttribute("stroke", "var(--surface)"); dot.setAttribute("stroke-width", "2");
    g.append(line, dot);
  });
  s.append(g);
  const box = el("div", { class: "chart" }, s);
  const tip = el("div", { class: "tip", hidden: true });
  box.append(tip, el("p", { class: "cap" }, `Every transit stacked on the ${fix(page.checks.signal.period, 4)}-day period. The dip is ${int(depth)} ppm deep.`));
  s.addEventListener("pointermove", (ev) => {
    const r = s.getBoundingClientRect();
    const vx = xlo + (((ev.clientX - r.left) / r.width) * W - padL) / (W - padL - padR) * (xhi - xlo);
    let i = 0;
    x_h.forEach((x, k) => { if (Math.abs(x - vx) < Math.abs(x_h[i] - vx)) i = k; });
    tip.hidden = false;
    tip.replaceChildren(`${fix(x_h[i], 2)} h`, el("br"), `${int(y_ppm[i])} ± ${int(err_ppm[i])} ppm`);
    tip.style.left = `${(X(x_h[i]) / W) * r.width + 22}px`;
    tip.style.top = `${(Y(y_ppm[i]) / H) * r.height - 10}px`;
  });
  s.addEventListener("pointerleave", () => { tip.hidden = true; });
  box.append(dataTable(["hours", "ppm", "error"], x_h.map((x, i) => [fix(x, 2), fix(y_ppm[i], 0), fix(err_ppm[i], 0)]), "Show the numbers"));
  return box;
}

// Dip depth sector by sector: a planet's stays put; a blend's moves with the aperture
export function sectorDepths(page: StarPage): HTMLElement | null {
  const per = page.checks.sectors?.per_sector as { sector: number; depth_ppm: number; err_ppm: number }[] | undefined;
  if (!per || per.length < 2) return null;
  const W = 560, H = 220, padL = 48, padR = 12, padT = 12, padB = 36;
  const w = per.map((p) => 1 / p.err_ppm ** 2);
  const mean = per.reduce((a, p, i) => a + p.depth_ppm * w[i], 0) / w.reduce((a, b) => a + b, 0);
  const ylo = Math.min(0, ...per.map((p) => p.depth_ppm - p.err_ppm)), yhi = Math.max(...per.map((p) => p.depth_ppm + p.err_ppm)) * 1.08;
  const X = (i: number) => padL + ((i + 0.5) / per.length) * (W - padL - padR);
  const Y = (v: number) => padT + ((yhi - v) / (yhi - ylo)) * (H - padT - padB);
  const yt = niceTicks(ylo, yhi, 4);
  const s = svg(`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Dip depth in each sector">
    <g class="grid">${yt.map((v) => `<line x1="${padL}" x2="${W - padR}" y1="${Y(v)}" y2="${Y(v)}"/>`).join("")}</g>
    ${yt.map((v) => `<text x="${padL - 6}" y="${Y(v)}" text-anchor="end" dominant-baseline="middle">${compact(v)}</text>`).join("")}
    <line x1="${padL}" x2="${W - padR}" y1="${Y(mean)}" y2="${Y(mean)}" stroke="var(--ink-2)" stroke-width="1"/>
    <text x="${W - padR}" y="${Y(mean) - 7}" text-anchor="end">average ${int(mean)} ppm</text>
    ${per.map((p, i) => `<line x1="${X(i)}" x2="${X(i)}" y1="${Y(p.depth_ppm - p.err_ppm)}" y2="${Y(p.depth_ppm + p.err_ppm)}" stroke="var(--axis)" stroke-width="1.5"/>
      <circle cx="${X(i)}" cy="${Y(p.depth_ppm)}" r="4.5" fill="var(--searched)" stroke="var(--surface)" stroke-width="2"><title>Sector ${p.sector}: ${p.depth_ppm} ± ${p.err_ppm} ppm</title></circle>
      <text x="${X(i)}" y="${H - padB + 16}" text-anchor="middle">S${p.sector}</text>`).join("")}
  </svg>`);
  const sc = page.checks.sectors;
  const box = el("div", { class: "chart" }, s, el("p", { class: "cap" },
    `Depth in each sector. Chance this spread comes from noise alone: ${pct(sc.p_constant)}.`));
  box.append(dataTable(["sector", "depth ppm", "error"], per.map((p) => [p.sector, p.depth_ppm, p.err_ppm]), "Show the numbers"));
  return box;
}

// How clearly one transit shows on a 0.4, 1 and 2 m telescope (signal-to-noise per transit)
export function snrBars(snr: Record<string, number>): HTMLElement {
  const rows = Object.entries(snr);
  const W = 560, rowH = 34, padL = 54, padR = 60, padT = 8, H = padT + rows.length * rowH + 30;
  const max = Math.max(6, ...rows.map(([, v]) => v)) * 1.1;
  const X = (v: number) => padL + (v / max) * (W - padL - padR);
  const s = svg(`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Signal-to-noise of one transit by telescope size">
    ${[3, 5].map((v) => `<line x1="${X(v)}" x2="${X(v)}" y1="${padT - 4}" y2="${padT + rows.length * rowH}" stroke="var(--axis)" stroke-width="1"/>
      <text x="${X(v)}" y="${padT + rows.length * rowH + 16}" text-anchor="middle">${v === 3 ? "3: stack a few" : "5: one is enough"}</text>`).join("")}
    ${rows.map(([k, v], i) => {
      const y = padT + i * rowH + 7, w = Math.max(X(v) - padL, 2), h = 18;
      return `<text x="${padL - 8}" y="${y + h / 2}" text-anchor="end" dominant-baseline="middle">${k}</text>
        <path d="M${padL} ${y} h${w - 4} a4 4 0 0 1 4 4 v${h - 8} a4 4 0 0 1 -4 4 h${-(w - 4)} z" fill="var(--candidate)"><title>${k}: ${v}</title></path>
        <text x="${padL + w + 8}" y="${y + h / 2}" dominant-baseline="middle" style="fill:var(--ink)">${v}</text>`;
    }).join("")}
  </svg>`);
  return el("div", { class: "chart" }, s, el("p", { class: "cap" },
    "Signal-to-noise of a single transit for telescope apertures of 0.4, 1 and 2 metres, from photon and atmospheric noise with a factor of 2 for real-world losses."));
}
