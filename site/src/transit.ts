// "If it is a planet, this is the dip": the star face-on as TESS sees it, the planet to scale crossing
// it on the chord the duration implies, and the real folded data with a cursor at the same moment.
// Plays on its own; pointer or arrow keys scrub; reduced motion starts paused.
import type { StarPage } from "./data";
import { css, el, fix, int } from "./util";

// Star colours by temperature (blackbody, white-balanced to the Sun), for the disc only
const TEFF_RGB: [number, string][] = [[2800, "#ffad5e"], [3200, "#ffbb78"], [3600, "#ffc68f"], [4000, "#ffd1a3"], [4500, "#ffdbb8"], [5000, "#ffe4cc"]];

function starColor(teff: number): string {
  return TEFF_RGB.reduce((best, row) => (Math.abs(row[0] - teff) < Math.abs(best[0] - teff) ? row : best))[1];
}

export function transitView(page: StarPage): HTMLElement | null {
  const c = page.checks, w = page.when;
  const k = c.size?.rp_over_rstar as number | undefined;
  const t14 = (w?.ephemeris.t14_h ?? c.shape?.t14_h ?? c.signal.dur_h) as number;
  if (!k || !t14) return null;
  const kr: number = k;
  const ratio = Math.min(c.duration?.ratio ?? 1, 1);
  const b = Math.sqrt(Math.max(0, 1 - ratio * ratio));  // impact parameter from how short the dip is
  const half = Math.sqrt(Math.max((1 + k) ** 2 - b * b, 0.01));  // half the chord, in star radii
  const { x_h, y_ppm, err_ppm } = page.fold;
  const xlo = Math.min(...x_h), xhi = Math.max(...x_h);
  const depth = c.signal.depth_ppm as number;
  const ylo = Math.min(...y_ppm.map((v, i) => v - err_ppm[i]), -1.25 * depth), yhi = Math.max(...y_ppm.map((v, i) => v + err_ppm[i]), 0.35 * depth);

  const canvas = el("canvas", { tabindex: 0, role: "img",
    "aria-label": `A planet ${fix(k * 100, 1)}% of the star's width crossing it, with the folded dip below. Arrow keys move through the transit.` });
  const play = el("button", { type: "button", class: "chip" }, "Pause");
  const box = el("div", { class: "chart transit" }, canvas,
    el("div", { class: "transit-bar" }, play, el("span", { class: "cap" },
      `Drawn to scale: the planet is ${fix(k * 100, 1)}% of the star's width and crosses in ${fix(t14, 1)} hours. `
      + `The dots are the real data, ${int(depth)} ppm deep.`)));

  let t = xlo, running = !matchMedia("(prefers-reduced-motion: reduce)").matches, last = performance.now(), raf = 0;
  const H = 360;
  function draw() {
    const W = canvas.clientWidth || 600;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    if (canvas.width !== W * dpr) { canvas.width = W * dpr; canvas.height = H * dpr; canvas.style.height = `${H}px`; }
    const g = canvas.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    // the star, limb-darkened
    const R = Math.min(78, W * 0.14), cx = W / 2, cy = 96;
    const col = starColor(c.star.teff ?? 3500);
    const grad = g.createRadialGradient(cx, cy, 0, cx, cy, R);
    grad.addColorStop(0, col);
    grad.addColorStop(0.75, col);
    grad.addColorStop(1, "rgba(150, 70, 20, 0.85)");
    g.fillStyle = grad;
    g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.fill();
    // the chord, then the planet on it
    const px = cx + (t / (t14 / 2)) * half * R, py = cy + b * R;
    g.strokeStyle = css("--axis");
    g.lineWidth = 1;
    const span = (v: number) => cx + (v / (t14 / 2)) * half * R;
    g.beginPath(); g.moveTo(span(xlo), py); g.lineTo(span(xhi), py); g.stroke();
    g.fillStyle = css("--page");
    g.strokeStyle = css("--ink");
    g.beginPath(); g.arc(px, py, Math.max(kr * R, 1.5), 0, Math.PI * 2); g.fill(); g.stroke();
    // the fold below, with a cursor at time t
    const top = 205, bottom = H - 26, left = 44, right = W - 10;
    const X = (v: number) => left + ((v - xlo) / (xhi - xlo)) * (right - left);
    const Y = (v: number) => top + ((yhi - v) / (yhi - ylo)) * (bottom - top);
    g.font = `11px ${css("--sans")}`;
    g.fillStyle = css("--muted");
    g.textAlign = "right"; g.textBaseline = "middle";
    for (const v of [0, -Math.round(depth / 100) * 100]) {
      g.strokeStyle = css("--hair");
      g.beginPath(); g.moveTo(left, Math.round(Y(v)) + 0.5); g.lineTo(right, Math.round(Y(v)) + 0.5); g.stroke();
      g.fillText(`${v}`, left - 6, Y(v));
    }
    g.textAlign = "center";
    g.fillText("hours from mid-transit", (left + right) / 2, H - 8);
    g.fillStyle = css("--candidate");
    x_h.forEach((x, i) => {
      g.globalAlpha = x <= t ? 1 : 0.3;
      g.beginPath(); g.arc(X(x), Y(y_ppm[i]), 3, 0, Math.PI * 2); g.fill();
    });
    g.globalAlpha = 1;
    g.strokeStyle = css("--ink-2");
    g.beginPath(); g.moveTo(Math.round(X(t)) + 0.5, top - 6); g.lineTo(Math.round(X(t)) + 0.5, bottom); g.stroke();
    g.fillStyle = css("--ink-2");
    g.textAlign = t > (xlo + xhi) / 2 ? "right" : "left";
    g.fillText(`${t >= 0 ? "+" : ""}${fix(t, 1)} h`, X(t) + (t > (xlo + xhi) / 2 ? -6 : 6), top - 10);
  }
  function tick(now: number) {
    raf = 0;
    if (!box.isConnected) return;
    if (running) {
      t += ((now - last) / 1000) * ((xhi - xlo) / 9);  // one pass in 9 seconds
      if (t > xhi) t = xlo;
      raf = requestAnimationFrame(tick);
    }
    last = now;
    draw();
  }
  const go = () => { if (!raf) { last = performance.now(); raf = requestAnimationFrame(tick); } };
  play.addEventListener("click", () => {
    running = !running;
    play.textContent = running ? "Pause" : "Play";
    go();
  });
  const scrub = (ev: PointerEvent) => {
    const r = canvas.getBoundingClientRect();
    t = Math.max(xlo, Math.min(xhi, xlo + ((ev.clientX - r.left - 44) / (r.width - 54)) * (xhi - xlo)));
    running = false;
    play.textContent = "Play";
    go();
  };
  canvas.addEventListener("pointerdown", scrub);
  canvas.addEventListener("pointermove", (ev) => { if (ev.buttons & 1) scrub(ev); });
  canvas.addEventListener("keydown", (ev) => {
    if (ev.key !== "ArrowLeft" && ev.key !== "ArrowRight") return;
    ev.preventDefault();
    t = Math.max(xlo, Math.min(xhi, t + (ev.key === "ArrowRight" ? 0.1 : -0.1)));
    running = false;
    play.textContent = "Play";
    go();
  });
  if (!running) play.textContent = "Play";
  new ResizeObserver(go).observe(canvas);
  return box;
}
