// The hero: every red star in the pool at its true place within 100 parsecs of the Sun, coloured and
// shaped by what Tessa did with it. Drag to turn it; hover for a name; click a dip to open its page.
import * as THREE from "three";
import type { Sky, Status } from "./data";
import { STATUS, css, el, int, navigate, spectral } from "./util";

const SHAPE: Record<string, number> = { hollow: 0, dot: 1, diamond: 2, triangle: 3, cross: 4, square: 5, ring: 6 };
const SIZE: Record<Status, number> = {
  pool: 2.2, "held back": 2.2, searched: 3, "awaiting agent": 7, rejected: 7, flagged: 9, known: 8, candidate: 15,
};
const PC = 1 / 100;  // scene units per parsec

const vertex = /* glsl */ `
  attribute vec3 color;
  attribute float size;
  attribute float shape;
  attribute float alpha;
  varying vec3 vColor;
  varying float vShape;
  varying float vAlpha;
  uniform float dpr;
  void main() {
    vColor = color; vShape = shape; vAlpha = alpha;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = size * dpr;
  }`;

// Shapes match the legend marks in util.ts
const fragment = /* glsl */ `
  varying vec3 vColor;
  varying float vShape;
  varying float vAlpha;
  void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    p.y = -p.y;
    float r = length(p);
    float a = 0.0;
    if (vShape < 0.5) a = 1.0 - smoothstep(0.75, 1.0, r);                              // small soft dot
    else if (vShape < 1.5) a = 1.0 - smoothstep(0.7, 1.0, r);
    else if (vShape < 2.5) a = 1.0 - smoothstep(0.8, 0.95, abs(p.x) + abs(p.y));       // diamond
    else if (vShape < 3.5) {                                                            // triangle
      float edge = max(-p.y - 0.62, max(p.y * 0.55 + abs(p.x) * 1.1 - 0.58, -1.0));
      a = 1.0 - smoothstep(-0.06, 0.04, edge);
    } else if (vShape < 4.5) {                                                          // cross
      float d = min(abs(p.x - p.y), abs(p.x + p.y)) * 0.7071;
      a = (1.0 - smoothstep(0.15, 0.24, d)) * (1.0 - smoothstep(0.8, 0.9, r));
    } else if (vShape < 5.5) a = 1.0 - smoothstep(0.66, 0.76, max(abs(p.x), abs(p.y))); // square
    else a = (1.0 - smoothstep(0.08, 0.16, abs(r - 0.62)));                             // ring
    if (a < 0.02) discard;
    gl_FragColor = vec4(vColor, a * vAlpha);
  }`;

export interface SkyView {
  focus(status: Status | null): void;
  dispose(): void;
}

export function mountSky(host: HTMLElement, sky: Sky, onFocusChange?: (s: Status | null) => void): SkyView | null {
  const canvas = el("canvas", { class: "sky", role: "img",
    "aria-label": `${int(sky.tic.length)} red stars within 100 parsecs of the Sun, coloured by what Tessa did with each` });
  let renderer: THREE.WebGLRenderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  } catch {
    return null;
  }
  host.prepend(canvas);
  const tip = el("div", { class: "sky-tip", hidden: true });
  host.append(tip);

  const codeToStatus = Object.fromEntries(Object.entries(sky.codes).map(([k, v]) => [v, k])) as Record<number, Status>;
  const n = sky.tic.length;
  const status = sky.status.map((c) => codeToStatus[c]);

  // Draw order: the pool first, then searched, then dips, candidates last and on top
  const rank = (s: Status) => ["pool", "held back", "searched", "awaiting agent", "rejected", "known", "flagged", "candidate"].indexOf(s);
  const order = [...Array(n).keys()].sort((a, b) => rank(status[a]) - rank(status[b]));

  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);
  const size = new Float32Array(n);
  const shape = new Float32Array(n);
  const alpha = new Float32Array(n);
  const idx = new Int32Array(n);  // buffer slot -> sky row
  order.forEach((row, i) => {
    const ra = (sky.ra[row] * Math.PI) / 180, dec = (sky.dec[row] * Math.PI) / 180, d = sky.dist[row] * PC;
    pos[i * 3] = d * Math.cos(dec) * Math.cos(ra);
    pos[i * 3 + 1] = d * Math.sin(dec);
    pos[i * 3 + 2] = -d * Math.cos(dec) * Math.sin(ra);
    size[i] = SIZE[status[row]];
    shape[i] = SHAPE[STATUS[status[row]].shape];
    idx[i] = row;
  });

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geom.setAttribute("color", new THREE.BufferAttribute(col, 3));
  geom.setAttribute("size", new THREE.BufferAttribute(size, 1));
  geom.setAttribute("shape", new THREE.BufferAttribute(shape, 1));
  geom.setAttribute("alpha", new THREE.BufferAttribute(alpha, 1));
  const material = new THREE.ShaderMaterial({
    vertexShader: vertex, fragmentShader: fragment, transparent: true, depthWrite: false, depthTest: false,
    uniforms: { dpr: { value: 1 } },
  });
  const points = new THREE.Points(geom, material);

  const scene = new THREE.Scene();
  const world = new THREE.Group();
  world.add(points);
  scene.add(world);

  // The Sun, and a ring for the 100 parsec edge that always faces the viewer
  const sunMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const sun = new THREE.Mesh(new THREE.CircleGeometry(0.012, 24), sunMat);
  scene.add(sun);
  const ringMat = new THREE.LineBasicMaterial({ transparent: true, opacity: 0.6 });
  const ringPts = Array.from({ length: 129 }, (_, i) => {
    const a = (i / 128) * Math.PI * 2;
    return new THREE.Vector3(Math.cos(a), Math.sin(a), 0);
  });
  const ring = new THREE.Line(new THREE.BufferGeometry().setFromPoints(ringPts), ringMat);
  scene.add(ring);

  const camera = new THREE.PerspectiveCamera(36, 1, 0.01, 20);
  camera.position.set(0, 0, 3.3);
  world.rotation.set(0.42, -0.6, 0);

  // Labels: the candidates by name, plus the Sun and the scale
  const labels: { el: HTMLElement; at: THREE.Vector3; local: boolean }[] = [];
  const addLabel = (text: string, at: THREE.Vector3, local: boolean, href?: string) => {
    const node = href ? el("a", { class: "sky-label", href, "data-link": true }, text) : el("span", { class: "sky-label" }, text);
    host.append(node);
    labels.push({ el: node, at, local });
  };
  addLabel("Sun", new THREE.Vector3(0, 0, 0), true);
  addLabel("100 parsecs", new THREE.Vector3(Math.cos(-0.5), Math.sin(-0.5), 0), false);
  for (let i = 0; i < n; i++) {
    const row = idx[i];
    if (status[row] === "candidate") {
      addLabel(`TIC ${sky.tic[row]}`, new THREE.Vector3(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]), true, `/star/${sky.tic[row]}`);
    }
  }

  let focused: Status | null = null;
  function paint() {
    const rgb: Record<string, THREE.Color> = {};
    for (const s of Object.keys(STATUS) as Status[]) {
      rgb[s] = new THREE.Color(css(STATUS[s].color.slice(4, -1)));
    }
    for (let i = 0; i < n; i++) {
      const s = status[idx[i]];
      const c = rgb[s];
      col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
      const base = s === "pool" || s === "held back" ? 0.75 : 1;
      alpha[i] = focused && s !== focused && !(focused === "pool" && s === "held back") ? base * 0.12 : base;
    }
    geom.attributes.color.needsUpdate = true;
    geom.attributes.alpha.needsUpdate = true;
    sunMat.color.set(css("--ink"));
    ringMat.color.set(css("--axis"));
    request();
  }

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    renderer.setPixelRatio(dpr);
    renderer.setSize(w, h, false);
    material.uniforms.dpr.value = dpr;
    camera.aspect = w / h;
    // Keep the whole 100 pc sphere in view on narrow screens, and push it right on wide ones
    const fit = w / h < 1 ? 3.3 / Math.max(w / h, 0.55) : 3.3;
    camera.position.z = fit;
    camera.setViewOffset(w, h, w > 840 ? -w * 0.17 : 0, 0, w, h);
    camera.updateProjectionMatrix();
    request();
  }

  // Motion: a slow turn unless the reader prefers none; dragging takes over
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let spin = still ? 0 : 0.035;
  let lastTouch = 0;
  let raf = 0;
  let visible = true;
  let prev = performance.now();
  const v = new THREE.Vector3();

  function frame(now: number) {
    raf = 0;
    const dt = Math.min((now - prev) / 1000, 0.1);
    prev = now;
    if (spin && now - lastTouch > 6000) world.rotation.y += spin * dt;
    world.updateMatrixWorld();
    ring.quaternion.copy(camera.quaternion);
    sun.quaternion.copy(camera.quaternion);
    renderer.render(scene, camera);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    const ox = canvas.offsetLeft, oy = canvas.offsetTop;
    for (const l of labels) {
      v.copy(l.at);
      if (l.local) v.applyMatrix4(world.matrixWorld);
      v.project(camera);
      l.el.style.left = `${ox + ((v.x + 1) / 2) * w}px`;
      l.el.style.top = `${oy + ((1 - v.y) / 2) * h}px`;
    }
    if (spin && visible) request();
  }
  function request() {
    if (!raf) raf = requestAnimationFrame(frame);
  }

  // Picking in screen space: the nearest dip within reach wins over the nearest pool star
  function pick(mx: number, my: number): number | null {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    const m = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse).multiply(world.matrixWorld);
    const e = m.elements;
    let best = -1, bestD = Infinity;
    for (let i = n - 1; i >= 0; i--) {
      if (alpha[i] < 0.5) continue;
      const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
      const cw = e[3] * x + e[7] * y + e[11] * z + e[15];
      const sx = (((e[0] * x + e[4] * y + e[8] * z + e[12]) / cw + 1) / 2) * w;
      const sy = ((1 - (e[1] * x + e[5] * y + e[9] * z + e[13]) / cw) / 2) * h;
      const reach = Math.max(size[i] * 0.8, 6);
      const d = Math.hypot(sx - mx, sy - my);
      if (d < reach && d - size[i] < bestD) {
        bestD = d - size[i];
        best = i;
      }
    }
    return best >= 0 ? idx[best] : null;
  }

  const hasPage = (s: Status) => ["rejected", "flagged", "candidate", "known", "awaiting agent"].includes(s);
  function describe(row: number) {
    const s = status[row];
    tip.replaceChildren(
      el("b", {}, `TIC ${sky.tic[row]}`), el("br"),
      `${spectral(sky.teff[row])}, ${sky.dist[row].toFixed(1)} pc, TESS mag ${sky.tmag[row].toFixed(1)}`, el("br"),
      STATUS[s].label + (hasPage(s) ? ", click to open" : ""),
    );
  }

  let drag: { x: number; y: number; moved: boolean } | null = null;
  let hoverRow: number | null = null;
  canvas.addEventListener("pointerdown", (ev) => {
    drag = { x: ev.clientX, y: ev.clientY, moved: false };
    canvas.setPointerCapture(ev.pointerId);
  });
  canvas.addEventListener("pointermove", (ev) => {
    const r = canvas.getBoundingClientRect();
    if (drag && (ev.buttons & 1)) {
      const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      world.rotation.y += dx * 0.006;
      world.rotation.x = Math.max(-1.4, Math.min(1.4, world.rotation.x + dy * 0.006));
      drag.x = ev.clientX; drag.y = ev.clientY;
      lastTouch = performance.now();
      tip.hidden = true;
      request();
      return;
    }
    hoverRow = pick(ev.clientX - r.left, ev.clientY - r.top);
    if (hoverRow == null) {
      tip.hidden = true;
      canvas.style.cursor = "";
      return;
    }
    describe(hoverRow);
    tip.hidden = false;
    tip.style.left = `${ev.clientX - host.getBoundingClientRect().left}px`;
    tip.style.top = `${ev.clientY - host.getBoundingClientRect().top}px`;
    canvas.style.cursor = hasPage(status[hoverRow]) ? "pointer" : "";
  });
  canvas.addEventListener("pointerup", (ev) => {
    const wasDrag = drag?.moved;
    drag = null;
    if (wasDrag) return;
    const r = canvas.getBoundingClientRect();
    const row = pick(ev.clientX - r.left, ev.clientY - r.top);
    if (row != null && hasPage(status[row])) navigate(`/star/${sky.tic[row]}`);
  });
  canvas.addEventListener("pointerleave", () => { tip.hidden = true; });

  const ro = new ResizeObserver(resize);
  ro.observe(canvas);
  const io = new IntersectionObserver(([e]) => {
    visible = e.isIntersecting;
    if (visible) request();
  });
  io.observe(canvas);
  const themeWatch = new MutationObserver(paint);
  themeWatch.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  const mq = matchMedia("(prefers-color-scheme: dark)");
  mq.addEventListener("change", paint);

  paint();
  resize();

  return {
    focus(s) {
      focused = s;
      paint();
      onFocusChange?.(s);
    },
    dispose() {
      cancelAnimationFrame(raf);
      spin = 0;
      ro.disconnect();
      io.disconnect();
      themeWatch.disconnect();
      mq.removeEventListener("change", paint);
      geom.dispose();
      material.dispose();
      renderer.dispose();
    },
  };
}
