// The window everything happens in: every planet found around another star and every TESS candidate still
// waiting, at its true place around the Sun (NASA Exoplanet Archive), inside a sketch of the Milky Way drawn to the
// same scale. On the first visit the planets arrive in the order they were found. Drag to turn, scroll or pinch to
// zoom, click a point to fly there.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { getPlanets } from "./data";
import type { Planets } from "./data";
import { css, el, fix, int, svg } from "./util";

const D2R = Math.PI / 180;
const LY = 3.26156;  // light-years per parsec
const GC = 8178;     // parsecs from the Sun to the centre of the galaxy (GRAVITY Collaboration 2019)
const NOW = new Date().getFullYear();
// ICRS to galactic unit vectors: x to the galactic centre, y to longitude 90, z to the north galactic pole
const T = [-0.0548755604, -0.8734370902, -0.4838350155, 0.4941094279, -0.44482963, 0.7469822445,
  -0.867666149, -0.1980763734, 0.4559837762];
const HOME = { dist: 3000, dir: new THREE.Vector3(-0.35, 0.62, 1).normalize() };
const INTRO_FROM = 1200;  // close enough to see 1992's pulsar planets, 600 pc out, flare

type Group = "transit" | "rv" | "lens" | "other" | "waiting" | "tessa";
// Three hues pass the palette validator on black (all pairs, with Tessa's amber); every other method is grey.
// Shapes keep the kinds apart without colour: dots confirmed, rings waiting, diamonds Tessa's.
const GROUPS: { key: Group; label: string; color: string; shape: number }[] = [
  { key: "transit", label: "Crossed in front of its star", color: "--m-transit", shape: 0 },
  { key: "rv", label: "Made its star wobble", color: "--m-rv", shape: 0 },
  { key: "lens", label: "Bent the light of a star behind it", color: "--m-lens", shape: 0 },
  { key: "other", label: "Other ways, photographs included", color: "--m-other", shape: 0 },
  { key: "waiting", label: "TESS candidates, not confirmed yet", color: "--waiting", shape: 1 },
  { key: "tessa", label: "Found by Tessa", color: "--candidate", shape: 2 },
];
const GROUP_OF: Record<string, Group> = { Transit: "transit", "Radial Velocity": "rv", Microlensing: "lens" };
const HOW: Record<string, string> = {
  Transit: "it crossed in front of its star and dimmed it (the transit method)",
  "Radial Velocity": "its pull made its star wobble (radial velocity)",
  Microlensing: "its gravity bent the light of a star far behind it (microlensing)",
  Imaging: "a photograph of the planet itself (direct imaging)",
  "Pulsar Timing": "it shifted the beat of a dead star's radio pulses (pulsar timing)",
  "Transit Timing Variations": "it tugged a neighbouring planet off its schedule",
  "Eclipse Timing Variations": "it tugged two stars' eclipses off their schedule",
  "Orbital Brightness Modulation": "its glow rose and fell as it went round",
  Astrometry: "its pull moved its star across the sky (astrometry)",
  "Pulsation Timing Variations": "it shifted the rhythm of a pulsating star",
  "Disk Kinematics": "it stirred the gas of the disk it is forming in",
};
// The opening clock: seconds into the intro -> the year the sky has reached. Waiting candidates arrive at 2027,
// Tessa's at 2028, after today's planets.
const CLOCK: [number, number][] = [[0, 1990.6], [1.2, 1992.4], [8.4, 2026.6], [9.6, 2027.6], [11, 2028.6]];
const MILESTONES: [number, string][] = [
  [1992, "The first planets ever confirmed, circling a dead star."],
  [1995, "The first around a star like the Sun: 51 Pegasi b."],
  [2009, "NASA launches Kepler to stare at one patch of sky."],
  [2014, "Kepler: 715 planets announced in one day."],
  [2016, "Kepler again: 1,284 in one day."],
  [2018, "NASA launches TESS to watch the whole sky."],
  [2026, "Today."],
];

interface System {
  host: string; kind: number; group: Group; rows: number[]; at: THREE.Vector3; dist: number; year: number;
}

function systems(p: Planets): System[] {
  const by = new Map<string, System>();
  p.kind.forEach((kind, i) => {
    const key = `${kind}|${p.host[i]}`;
    let s = by.get(key);
    if (!s) {
      const a = p.ra[i] * D2R, b = p.dec[i] * D2R, d = p.dist[i];
      const e0 = Math.cos(b) * Math.cos(a), e1 = Math.cos(b) * Math.sin(a), e2 = Math.sin(b);
      const gx = T[0] * e0 + T[1] * e1 + T[2] * e2, gy = T[3] * e0 + T[4] * e1 + T[5] * e2, gz = T[6] * e0 + T[7] * e1 + T[8] * e2;
      // scene axes: x to the galactic centre, y to the north galactic pole, z = -galactic y (still right-handed)
      s = { host: p.host[i], kind, group: "other", rows: [], at: new THREE.Vector3(d * gx, d * gz, -d * gy), dist: d, year: Infinity };
      by.set(key, s);
    }
    s.rows.push(i);
    if (p.year[i] < s.year) {
      s.year = p.year[i];
      s.group = kind === 2 ? "tessa" : kind === 1 ? "waiting" : GROUP_OF[p.methods[p.method[i]]] ?? "other";
    }
  });
  return [...by.values()];
}

// A sketch of the Milky Way at true scale: a bar, four arms (one brushing past the Sun) and a thin disk. It is
// drawn, not measured, unlike every planet in the view.
function galaxy(n: number): { pos: Float32Array; col: Float32Array } {
  let seed = 20260;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-9)) * Math.cos(2 * Math.PI * rnd());
  const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
  const pitch = Math.tan(12 * D2R), tilt = 27 * D2R;
  for (let i = 0; i < n; i++) {
    const u = rnd();
    let x: number, y: number, z: number, c: number[];
    if (u < 0.2) {
      const bx = gauss() * 1500, bz = gauss() * 480;
      x = GC + bx * Math.cos(tilt) - bz * Math.sin(tilt);
      z = bx * Math.sin(tilt) + bz * Math.cos(tilt);
      y = gauss() * 380;
      c = [1, 0.84, 0.62];
    } else {
      const arm = u < 0.8;
      const r = arm ? 2600 - Math.log(rnd() + 1e-9) * 3200 : 400 - Math.log(rnd() + 1e-9) * 3800;
      if (r > 17000) { i--; continue; }
      const k = Math.floor(rnd() * 4);
      const ang = arm ? Math.PI + 0.1 + (k * Math.PI) / 2 + Math.log(r / GC) / pitch + gauss() * 0.16 : rnd() * Math.PI * 2;
      const rr = r + gauss() * (arm ? 260 : 0);
      x = GC + rr * Math.cos(ang);
      z = rr * Math.sin(ang);
      y = gauss() * (arm ? 110 : 220);
      c = arm ? (rnd() < 0.05 ? [1, 0.55, 0.66] : [0.7, 0.79, 1]) : [0.8, 0.78, 0.74];
    }
    pos.set([x, y, z], i * 3);
    const b = u < 0.2 ? 0.2 : 0.13;
    col.set([c[0] * b, c[1] * b, c[2] * b], i * 3);
  }
  return { pos, col };
}

const pointVertex = /* glsl */ `
  attribute vec3 color;
  attribute float size;
  attribute float shape;
  attribute float alpha;
  attribute float year;
  uniform float dpr;
  uniform float uYear;
  varying vec3 vColor;
  varying float vShape;
  varying float vAlpha;
  void main() {
    float age = uYear - year;                 // years since it was found, on the intro's clock
    float on = clamp(age / 0.7, 0.0, 1.0);
    vColor = color; vShape = shape; vAlpha = alpha * on;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = age < 0.0 ? 0.0 : size * dpr * (1.0 + 1.6 * (1.0 - on));  // a new find flares, then settles
  }`;

const pointFragment = /* glsl */ `
  varying vec3 vColor;
  varying float vShape;
  varying float vAlpha;
  void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r = length(p);
    float glow = 0.4 * pow(max(1.0 - r, 0.0), 2.2);
    float a;
    if (vShape < 0.5) a = max(1.0 - smoothstep(0.26, 0.4, r), glow);                          // dot
    else if (vShape < 1.5) a = 1.0 - smoothstep(0.06, 0.14, abs(r - 0.6));                     // ring
    else a = max(1.0 - smoothstep(0.42, 0.52, abs(p.x) + abs(p.y)), glow * 1.4);              // diamond
    a *= vAlpha;
    if (a < 0.01) discard;
    gl_FragColor = vec4(vColor, a);
  }`;

const dustVertex = /* glsl */ `
  attribute vec3 color;
  uniform float dpr;
  uniform float uFade;
  varying vec3 vColor;
  void main() {
    vColor = color * uFade;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = 2.2 * dpr;
  }`;

const dustFragment = /* glsl */ `
  varying vec3 vColor;
  void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float a = max(1.0 - dot(p, p), 0.0);
    gl_FragColor = vec4(vColor * a, 1.0);
  }`;

// Words for numbers a newcomer reads
const lyNum = (pc: number) => { const ly = pc * LY; return ly < 1000 ? int(ly) : int(Number(ly.toPrecision(2))); };
const times = (x: number) => (x >= 10 ? int(x) : fix(x, 1));
const words = (k: number) => ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"][k] ?? int(k);

function lightLeft(pc: number): string {
  const ly = Math.round(pc * LY);
  return ly < NOW ? `The light we see from it tonight left in ${NOW - ly}.` : `The light we see from it tonight left ${lyNum(pc)} years ago.`;
}

function facts(p: Planets, i: number): string {
  const out: string[] = [];
  const r = p.radius[i], m = p.mass[i], per = p.period[i], t = p.teq[i];
  if (r != null) out.push(r >= 1.15 ? `${times(r)} times as wide as Earth` : r > 0.85 ? "About the size of Earth" : `${fix(r, 1)} of Earth's width`);
  if (m != null) out.push(`${times(m)} times Earth's mass`);
  if (per != null) out.push(`A year there lasts ${per < 2 ? `${int(per * 24)} hours` : per < 730 ? `${fix(per, per < 10 ? 1 : 0)} days` : `${fix(per / 365.25, 1)} Earth years`}`);
  if (t != null) out.push(`Roughly ${int(Math.round((t - 273.15) / 10) * 10)} °C`);
  return out.length ? `${out.join(". ")}.` : "";
}

function swatch(g: (typeof GROUPS)[number]): SVGElement {
  const c = `var(${g.color})`;
  const body = g.shape === 0 ? `<circle cx="7" cy="7" r="4" fill="${c}"/>`
    : g.shape === 1 ? `<circle cx="7" cy="7" r="4.2" fill="none" stroke="${c}" stroke-width="1.5"/>`
      : `<path d="M7 1.5 12.5 7 7 12.5 1.5 7z" fill="${c}"/>`;
  return svg(`<svg class="sw" viewBox="0 0 14 14" aria-hidden="true">${body}</svg>`);
}

// A system's planets going round, orbits to scale with each other (Kepler's third law: same star, so the
// orbit's size goes as the period to the 2/3), time sped up so the innermost goes round every 3 seconds.
function orrery(p: Planets, rows: number[], color: string): HTMLElement | null {
  const known = rows.filter((i) => p.period[i] != null).sort((a, b) => p.period[a]! - p.period[b]!);
  if (!known.length) return null;
  const canvas = el("canvas", { class: "orrery", role: "img",
    "aria-label": `${known.length === 1 ? "The planet's orbit" : `${known.length} orbits`}, drawn to scale with each other, sped up` });
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const pMin = p.period[known[0]]!, pMax = p.period[known[known.length - 1]]!;
  // planets start a golden angle apart, so neighbours named b, c, d don't line up
  const phase = known.map((_, k) => k * 2.39996);
  const t0 = performance.now();
  const H = 132;
  function draw(now: number) {
    if (!canvas.isConnected && now - t0 > 500) return;
    const W = canvas.clientWidth || 320, dpr = Math.min(devicePixelRatio || 1, 2);
    if (canvas.width !== W * dpr) { canvas.width = W * dpr; canvas.height = H * dpr; }
    const g = canvas.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const cx = W / 2, cy = H / 2, outer = Math.min(W / 2 - 10, (H / 2 - 8) * 2.6);
    const teff = p.teff[known[0]] ?? 5000;
    g.fillStyle = teff < 3900 ? "#ffbf80" : teff < 5300 ? "#ffd9a8" : teff < 6300 ? "#fff1dc" : "#dfe8ff";
    g.beginPath(); g.arc(cx, cy, 4.5, 0, Math.PI * 2); g.fill();
    known.forEach((i, k) => {
      const a = Math.max(outer * (p.period[i]! / pMax) ** (2 / 3), 10 + k * 3);
      g.strokeStyle = "rgba(236, 235, 230, 0.18)";
      g.lineWidth = 1;
      g.beginPath(); g.ellipse(cx, cy, a, a * 0.38, 0, 0, Math.PI * 2); g.stroke();
      const ang = phase[k] + (still ? 0 : ((now - t0) / 3000) * Math.PI * 2 * (pMin / p.period[i]!));
      const r = Math.min(Math.max(1.6 + Math.sqrt(p.radius[i] ?? 2) * 1.1, 2), 6.5);
      g.fillStyle = color;
      g.beginPath(); g.arc(cx + a * Math.cos(ang), cy + a * 0.38 * Math.sin(ang), r, 0, Math.PI * 2); g.fill();
    });
    if (!still) requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);
  return el("figure", { class: "orrery-box" }, canvas,
    el("figcaption", {}, known.length > 1 ? "Orbits to scale with each other. Time sped up: the inner one goes round every 3 seconds."
      : "Time sped up: one orbit every 3 seconds."));
}

interface World { showHost(host: string): boolean; dispose(): void }

let introDone = false;
let current: World | null = null;

export async function universe(main: HTMLElement) {
  const data = await getPlanets();
  current = mount(main, data);
}

export function leaveUniverse() {
  current?.dispose();
  current = null;
}

// Ask Tessa's "show it on the map", for a star in this view (TESS candidates and Tessa's own are named by TIC)
export function flyToTic(tic: number): boolean {
  return current?.showHost(`TIC ${tic}`) ?? false;
}

function mount(host: HTMLElement, p: Planets): World | null {
  const sys = systems(p);
  const section = el("section", { class: "world-view", "aria-label": "Every planet found around another star" });
  const canvas = el("canvas", { class: "universe", tabindex: 0, role: "img",
    "aria-label": `${int(p.counts.confirmed)} confirmed planets and ${int(p.counts.waiting)} TESS candidates at their true places around the Sun. Arrow keys turn the view, plus and minus zoom.` });
  let renderer: THREE.WebGLRenderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  } catch {
    host.append(el("p", { class: "loading" }, "This view needs WebGL, which this browser has turned off."));
    return null;
  }
  renderer.setClearColor(css("--page"));
  const tip = el("div", { class: "u-tip", hidden: true });
  const labelLayer = el("div", { class: "u-labels", "aria-hidden": "true" });
  const card = el("section", { class: "u-card", "aria-live": "polite" });
  const scaleBar = el("span", { class: "u-scale-bar" });
  const scaleText = el("span", {});
  const centred = el("span", { class: "u-centred" });
  const readout = el("div", { class: "u-readout", "aria-hidden": "true" }, el("div", {}, scaleBar, scaleText), centred);
  section.append(canvas, labelLayer, tip, card, readout);
  host.append(section);

  // ---- the planets
  const n = sys.length;
  const order = [...sys.keys()].sort((a, b) => ["waiting", "other", "lens", "rv", "transit", "tessa"].indexOf(sys[a].group)
    - ["waiting", "other", "lens", "rv", "transit", "tessa"].indexOf(sys[b].group));
  const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
  const size = new Float32Array(n), shape = new Float32Array(n), alpha = new Float32Array(n), year = new Float32Array(n);
  const slotOf = new Int32Array(n);
  order.forEach((s, i) => {
    const q = sys[s];
    slotOf[s] = i;
    pos.set([q.at.x, q.at.y, q.at.z], i * 3);
    const g = GROUPS.find((x) => x.key === q.group)!;
    size[i] = q.kind === 2 ? 26 : q.kind === 1 ? 6 : 9 + 2 * Math.min(q.rows.length - 1, 4);
    shape[i] = g.shape;
    year[i] = q.kind === 2 ? 2028 : q.kind === 1 ? 2027 : q.year;
  });
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  for (const [name, arr, k] of [["color", col, 3], ["size", size, 1], ["shape", shape, 1], ["alpha", alpha, 1], ["year", year, 1]] as const) {
    geom.setAttribute(name, new THREE.BufferAttribute(arr, k));
  }
  const uniforms = { dpr: { value: 1 }, uYear: { value: introDone ? 3000 : 1990 } };
  const material = new THREE.ShaderMaterial({ vertexShader: pointVertex, fragmentShader: pointFragment, uniforms,
    transparent: true, depthWrite: false, depthTest: false });
  const points = new THREE.Points(geom, material);
  points.frustumCulled = false;

  // the selected system's ring
  const selGeom = new THREE.BufferGeometry();
  selGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3), 3));
  for (const [name, v] of [["color", [0.93, 0.92, 0.9]], ["size", [30]], ["shape", [1]], ["alpha", [1]], ["year", [0]]] as const) {
    selGeom.setAttribute(name, new THREE.BufferAttribute(new Float32Array(v), v.length));
  }
  const selPoint = new THREE.Points(selGeom, new THREE.ShaderMaterial({ vertexShader: pointVertex, fragmentShader: pointFragment,
    uniforms, transparent: true, depthWrite: false, depthTest: false }));
  selPoint.visible = false;
  selPoint.frustumCulled = false;

  // ---- the galaxy sketch, the Sun, and rings of light-travel time
  const dust = galaxy(70000);
  const dustGeom = new THREE.BufferGeometry();
  dustGeom.setAttribute("position", new THREE.BufferAttribute(dust.pos, 3));
  dustGeom.setAttribute("color", new THREE.BufferAttribute(dust.col, 3));
  const dustMat = new THREE.ShaderMaterial({ vertexShader: dustVertex, fragmentShader: dustFragment,
    uniforms: { dpr: uniforms.dpr, uFade: { value: 1 } }, transparent: true, depthWrite: false, depthTest: false,
    blending: THREE.AdditiveBlending });
  const dustPoints = new THREE.Points(dustGeom, dustMat);
  dustPoints.frustumCulled = false;

  const scene = new THREE.Scene();
  scene.add(dustPoints, points, selPoint);
  const sun = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), new THREE.MeshBasicMaterial({ color: 0xfff1dc }));
  scene.add(sun);
  const RINGS = [100, 1000, 10000];
  const rings = RINGS.map((ly) => {
    const r = ly / LY;
    const pts = Array.from({ length: 257 }, (_, i) => new THREE.Vector3(r * Math.cos((i / 256) * Math.PI * 2), 0, r * Math.sin((i / 256) * Math.PI * 2)));
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: css("--axis"), transparent: true, opacity: 0 }));
    scene.add(line);
    return { r, line };
  });

  const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 400000);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.rotateSpeed = 0.5;
  controls.zoomSpeed = 1.1;
  controls.enablePan = false;
  controls.minDistance = 0.3;
  controls.maxDistance = 120000;
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  controls.autoRotate = !still;
  controls.autoRotateSpeed = 0.18;
  camera.position.copy(HOME.dir).multiplyScalar(introDone ? HOME.dist : INTRO_FROM);

  // ---- labels in the scene
  // rank decides who keeps the spot when two labels would overlap: the Sun, then Tessa's, then the rest
  type Label = { node: HTMLElement; at: THREE.Vector3; rank: number; show?: (d: number) => number; w?: number; h?: number };
  const labels: Label[] = [];
  const label = (node: HTMLElement, at: THREE.Vector3, rank: number, show?: (d: number) => number) => {
    labelLayer.append(node);
    labels.push({ node, at, rank, show });
    labels.sort((a, b) => a.rank - b.rank);
  };
  label(el("span", { class: "u-label sun" }, "Sun"), new THREE.Vector3(0, 0, 0), 0, (d) => (d < 15000 ? 1 : 0));
  label(el("span", { class: "u-label sun" }, "You are here", el("small", {}, "Nearly every planet we know is in this corner")),
    new THREE.Vector3(0, 0, 0), 0, (d) => (d >= 15000 ? 1 : 0));
  label(el("span", { class: "u-label" }, "Centre of the Milky Way", el("small", {}, `${lyNum(GC)} light-years`)),
    new THREE.Vector3(GC, 0, 0), 3, (d) => (d > 4000 ? 1 : 0));
  RINGS.forEach((ly, k) => {
    const left = NOW - ly;
    const r = rings[k].r, a = -0.6;
    label(el("span", { class: "u-label ring" }, `${int(ly)} light-years`,
      el("small", {}, left > 0 ? `Light from here left in ${left}` : `Light from here left ${int(ly)} years ago`)),
    new THREE.Vector3(r * Math.cos(a), 0, r * Math.sin(a)), 2, (d) => ringAlpha(r, d, 160));
  });
  sys.filter((s) => s.kind === 2).forEach((s) => {
    const b = el("button", { type: "button", class: "u-label tessa", tabindex: -1 }, s.host, el("small", {}, "found by Tessa"));
    b.addEventListener("click", () => select(s));
    label(b, s.at, 1, (d) => (uniforms.uYear.value > 2028 && d < 8000 ? 1 : 0));
  });

  // a ring reads best when it spans a good part of the screen
  function ringAlpha(r: number, d: number, min = 40): number {
    const px = r / pcPerPx(d);
    const h = canvas.clientHeight || 800;
    return Math.min(Math.max((px - min) / 80, 0), 1) * Math.min(Math.max((h * 1.6 - px) / (h * 0.8), 0), 1);
  }
  function pcPerPx(d: number) {
    return (2 * d * Math.tan((camera.fov * D2R) / 2)) / (canvas.clientHeight || 800);
  }

  // ---- colour and focus
  let focus: Group | null = null;
  function paint() {
    const rgb = Object.fromEntries(GROUPS.map((g) => [g.key, new THREE.Color(css(g.color))])) as Record<Group, THREE.Color>;
    order.forEach((s, i) => {
      const q = sys[s], c = rgb[q.group];
      col.set([c.r, c.g, c.b], i * 3);
      const base = q.kind === 1 ? 0.55 : 1;
      alpha[i] = focus && q.group !== focus ? base * 0.07 : base;
    });
    geom.attributes.color.needsUpdate = true;
    geom.attributes.alpha.needsUpdate = true;
  }
  paint();

  // ---- the card: intro, overview, or one system
  const counts = Object.fromEntries(GROUPS.map((g) => [g.key, 0])) as Record<Group, number>;
  sys.forEach((s) => (counts[s.group] += s.rows.length));
  const legendButtons: HTMLButtonElement[] = [];

  function bar(left: Node | string, right?: Node) {
    return el("div", { class: "u-bar" }, el("span", { class: "u-bar-title" }, left), right ?? null);
  }

  // ?at=51 Peg (a star or a planet) opens straight onto that system; the address always says what is open
  function setAt(name: string | null) {
    const u = new URL(location.href);
    if (name) u.searchParams.set("at", name);
    else u.searchParams.delete("at");
    history.replaceState(history.state, "", u.pathname + u.search + u.hash);
  }
  function find(name: string): System | undefined {
    const q = name.trim().toLowerCase();
    return sys.find((s) => s.host.toLowerCase() === q || s.rows.some((i) => p.name[i].toLowerCase() === q));
  }

  function overview() {
    setAt(null);
    selPoint.visible = false;
    selected = null;
    centred.textContent = "Centred on the Sun";
    const legend = el("ul", { class: "u-legend", "aria-label": "How each planet was found; choose one to show only those" });
    legendButtons.length = 0;
    for (const g of GROUPS) {
      const b = el("button", { type: "button", "aria-pressed": String(focus === g.key) }, swatch(g),
        el("span", { class: "what" }, g.label), el("span", { class: "n" }, int(counts[g.key])));
      b.addEventListener("click", () => {
        focus = focus === g.key ? null : g.key;
        legendButtons.forEach((x, k) => x.setAttribute("aria-pressed", String(GROUPS[k].key === focus)));
        paint();
      });
      legendButtons.push(b);
      legend.append(el("li", {}, b));
    }
    const ours = sys.filter((s) => s.kind === 2).map((s) => {
      const b = el("button", { type: "button", class: "u-link" }, s.host);
      b.addEventListener("click", () => select(s));
      return b;
    });
    const replay = el("button", { type: "button", class: "u-bar-btn" }, "Replay 1992 to today");
    replay.addEventListener("click", () => startIntro());
    const c = p.counts;
    card.replaceChildren(
      bar("NASA Exoplanet Archive", replay),
      el("div", { class: "u-body" },
        el("h1", {}, `${int(c.confirmed)} planets found around other stars`),
        el("p", { class: "u-lede" }, `${int(c.waiting)} more TESS candidates are waiting to be confirmed, and Tessa has found ${words(c.tessa)} of its own.`),
        legend,
        ours.length ? el("p", { class: "u-ours" }, "Tessa's: ", ...ours) : null,
        el("p", { class: "u-hint" }, "Drag to turn. Scroll or pinch to zoom. Click any point to fly there."),
        el("p", { class: "u-small" },
          `Not shown: ${int(c.unplaced.confirmed)} planets and ${int(c.unplaced.waiting)} candidates with no measured distance. `
          + "Every point is where the archive puts it; the Milky Way behind them is a sketch at the same scale. ",
          el("a", { href: "/survey", "data-link": true }, "Tessa's survey, in full"), ".")));
  }

  const introYear = el("p", { class: "u-year" });
  const introCount = el("p", { class: "u-count" });
  const introNote = el("p", { class: "u-note" });
  function introCard() {
    const skip = el("button", { type: "button", class: "u-bar-btn" }, "Skip");
    skip.addEventListener("click", () => endIntro());
    card.replaceChildren(bar("Every planet, in the order it was found", skip),
      el("div", { class: "u-body" }, introYear, introCount, introNote));
  }
  // cumulative confirmed planets by year, for the intro's counter
  const byYear = new Map<number, number>();
  p.kind.forEach((k, i) => { if (k === 0) byYear.set(p.year[i], (byYear.get(p.year[i]) ?? 0) + 1); });
  const firstYear = Math.min(...byYear.keys());
  const cumulative = (y: number) => [...byYear].reduce((s, [yr, c]) => s + (yr <= y ? c : 0), 0);

  let selected: System | null = null;
  function systemCard(s: System) {
    const back = el("button", { type: "button", class: "u-bar-btn" }, "All planets");
    back.addEventListener("click", () => { overview(); flyTo(new THREE.Vector3(), HOME.dist); });
    const g = GROUPS.find((x) => x.key === s.group)!;
    const kindWord = s.kind === 2 ? "Found by Tessa" : s.kind === 1 ? "TESS candidate" : `${s.rows.length} planet${s.rows.length > 1 ? "s" : ""}`;
    const rows = [...s.rows].sort((a, b) => (p.period[a] ?? 1e9) - (p.period[b] ?? 1e9));
    const list = el("ol", { class: "u-planets" }, ...rows.map((i) => {
      const how = s.kind === 2 ? `Found by Tessa in ${p.year[i]} in NASA TESS data. Not confirmed yet.`
        : s.kind === 1 ? `A TESS candidate since ${p.year[i]}, waiting to be confirmed.`
          : `Found in ${p.year[i]}: ${HOW[p.methods[p.method[i]]] ?? p.methods[p.method[i]]}.`;
      return el("li", {}, el("h3", {}, p.name[i]), el("p", {}, how, " ", facts(p, i)));
    }));
    const tic = s.host.match(/^TIC (\d+)$/)?.[1];
    card.replaceChildren(bar(back, el("span", { class: "u-bar-tag" }, swatch(g), kindWord)),
      el("div", { class: "u-body" },
        el("h2", {}, s.host),
        el("p", { class: "u-lede" }, `${lyNum(s.dist)} light-years away. ${lightLeft(s.dist)}`),
        orrery(p, rows, css(g.color)),
        list,
        s.kind === 2 && tic ? el("p", {}, el("a", { href: `/star/${tic}`, "data-link": true, class: "u-cta" }, "Open its case file")) : null,
        share()));
  }
  function share() {
    const b = el("button", { type: "button", class: "u-bar-btn u-share" }, "Copy a link to this system");
    b.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(location.href);
        b.textContent = "Link copied";
      } catch {
        b.textContent = location.href;
      }
    });
    return b;
  }

  function select(s: System) {
    endIntro(false);
    selected = s;
    (selGeom.attributes.position.array as Float32Array).set([s.at.x, s.at.y, s.at.z]);
    selGeom.attributes.position.needsUpdate = true;
    selPoint.visible = true;
    setAt(s.host);
    centred.textContent = `Centred on ${s.host}`;
    systemCard(s);
    flyTo(s.at, Math.min(Math.max(s.dist * 0.14, 1.5), 700));
  }

  // ---- camera moves
  let tween: { from: THREE.Vector3; to: THREE.Vector3; fromD: number; toD: number; dir: THREE.Vector3; t0: number; ms: number } | null = null;
  function flyTo(to: THREE.Vector3, dist: number) {
    const dir = camera.position.clone().sub(controls.target).normalize();
    tween = { from: controls.target.clone(), to: to.clone(), fromD: camera.position.distanceTo(controls.target), toD: dist, dir,
      t0: performance.now(), ms: still ? 1 : 1800 };
  }
  const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

  let intro: { t0: number } | null = null;
  function startIntro() {
    if (still) { endIntro(); return; }
    focus = null;
    paint();
    selected = null;
    selPoint.visible = false;
    tween = null;
    controls.target.set(0, 0, 0);
    intro = { t0: performance.now() };
    introCard();
  }
  function endIntro(toOverview = true) {
    if (intro) {
      intro = null;
      introDone = true;
      uniforms.uYear.value = 3000;
      if (toOverview) overview();
    } else if (!introDone) {
      introDone = true;
      uniforms.uYear.value = 3000;
      if (toOverview) overview();
    }
  }
  function clockYear(sec: number) {
    for (let k = 1; k < CLOCK.length; k++) {
      const [t1, y1] = CLOCK[k - 1], [t2, y2] = CLOCK[k];
      if (sec <= t2) return y1 + ((sec - t1) / (t2 - t1)) * (y2 - y1);
    }
    return CLOCK[CLOCK.length - 1][1];
  }

  // ---- input
  controls.addEventListener("start", () => {
    if (intro) endIntro();
    tween = null;
    controls.autoRotate = false;
    idleSince = Infinity;
  });
  controls.addEventListener("end", () => { idleSince = performance.now(); });
  let idleSince = 0;

  function pick(mx: number, my: number): System | null {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    const m = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse).elements;
    let best = -1, bestD = Infinity;
    for (let i = n - 1; i >= 0; i--) {
      if (alpha[i] < 0.3 || year[i] > uniforms.uYear.value) continue;
      const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
      const cw = m[3] * x + m[7] * y + m[11] * z + m[15];
      if (cw <= 0) continue;
      const sx = (((m[0] * x + m[4] * y + m[8] * z + m[12]) / cw + 1) / 2) * w;
      const sy = ((1 - (m[1] * x + m[5] * y + m[9] * z + m[13]) / cw) / 2) * h;
      const d = Math.hypot(sx - mx, sy - my) - size[i] * 0.25;
      if (d < 9 && d < bestD) { bestD = d; best = i; }
    }
    return best >= 0 ? sys[order[best]] : null;
  }
  let down: { x: number; y: number } | null = null;
  canvas.addEventListener("pointerdown", (ev) => { down = { x: ev.clientX, y: ev.clientY }; });
  canvas.addEventListener("pointermove", (ev) => {
    if (ev.buttons) { tip.hidden = true; return; }
    const r = canvas.getBoundingClientRect();
    const s = pick(ev.clientX - r.left, ev.clientY - r.top);
    canvas.style.cursor = s ? "pointer" : "";
    if (!s) { tip.hidden = true; return; }
    tip.replaceChildren(el("b", {}, s.host), el("br"),
      `${s.kind === 1 ? "TESS candidate" : s.kind === 2 ? "Found by Tessa" : `${s.rows.length} planet${s.rows.length > 1 ? "s" : ""}`}, ${lyNum(s.dist)} light-years away`);
    tip.hidden = false;
    tip.style.left = `${ev.clientX - r.left}px`;
    tip.style.top = `${ev.clientY - r.top}px`;
  });
  canvas.addEventListener("pointerleave", () => { tip.hidden = true; });
  canvas.addEventListener("pointerup", (ev) => {
    if (!down || Math.hypot(ev.clientX - down.x, ev.clientY - down.y) > 4) return;
    const r = canvas.getBoundingClientRect();
    const s = pick(ev.clientX - r.left, ev.clientY - r.top);
    if (s) select(s);
  });
  canvas.addEventListener("keydown", (ev) => {
    const off = camera.position.clone().sub(controls.target);
    const sph = new THREE.Spherical().setFromVector3(off);
    if (ev.key === "ArrowLeft") sph.theta -= 0.12;
    else if (ev.key === "ArrowRight") sph.theta += 0.12;
    else if (ev.key === "ArrowUp") sph.phi = Math.max(0.05, sph.phi - 0.1);
    else if (ev.key === "ArrowDown") sph.phi = Math.min(Math.PI - 0.05, sph.phi + 0.1);
    else if (ev.key === "+" || ev.key === "=") sph.radius = Math.max(controls.minDistance, sph.radius / 1.4);
    else if (ev.key === "-") sph.radius = Math.min(controls.maxDistance, sph.radius * 1.4);
    else if (ev.key === "Escape" && selected) { overview(); flyTo(new THREE.Vector3(), HOME.dist); return; }
    else return;
    ev.preventDefault();
    if (intro) endIntro();
    controls.autoRotate = false;
    idleSince = performance.now();
    camera.position.copy(controls.target).add(new THREE.Vector3().setFromSpherical(sph));
  });

  // ---- frame
  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    renderer.setPixelRatio(dpr);
    renderer.setSize(w, h, false);
    uniforms.dpr.value = dpr;
    camera.aspect = w / h;
    // on wide screens the card covers the left, so centre the view in the space right of it
    const shift = w > 900 ? Math.min(card.offsetWidth, w * 0.4) / 2 : 0;
    camera.setViewOffset(w, h, -shift, w > 900 ? 0 : h * 0.16, w, h);
    camera.updateProjectionMatrix();
  }
  const ro = new ResizeObserver(resize);
  ro.observe(canvas);
  ro.observe(card);

  const v = new THREE.Vector3();
  let raf = 0;
  let lastScale = "";
  function frame(now: number) {
    raf = requestAnimationFrame(frame);
    if (intro) {
      const sec = (now - intro.t0) / 1000;
      const y = clockYear(sec);
      uniforms.uYear.value = y;
      const k = Math.min(sec / 8.4, 1);
      const d = Math.exp(Math.log(INTRO_FROM) + (Math.log(HOME.dist) - Math.log(INTRO_FROM)) * ease(k));
      const dir = HOME.dir.clone().applyAxisAngle(new THREE.Vector3(0, 1, 0), -0.5 * (1 - ease(k)));
      camera.position.copy(dir).multiplyScalar(d);
      const shown = Math.min(Math.floor(y), NOW);
      introYear.textContent = y < firstYear ? "" : String(shown);
      introCount.textContent = y > 2027 ? `+ ${int(p.counts.waiting)} TESS candidates waiting${y > 2028 ? `, and Tessa's ${words(p.counts.tessa)}` : ""}`
        : y < firstYear ? "" : `${int(cumulative(shown))} planet${cumulative(shown) === 1 ? "" : "s"} known`;
      introNote.textContent = y > 2027 ? "Rings are candidates waiting for a second look. The amber diamonds are Tessa's."
        : [...MILESTONES].reverse().find(([yr]) => yr <= shown)?.[1] ?? "";
      if (sec > 12.5) endIntro();
    } else if (tween) {
      const t = Math.min((now - tween.t0) / tween.ms, 1), e = ease(t);
      controls.target.lerpVectors(tween.from, tween.to, e);
      const d = Math.exp(Math.log(tween.fromD) + (Math.log(tween.toD) - Math.log(tween.fromD)) * e);
      camera.position.copy(controls.target).addScaledVector(tween.dir, d);
      if (t >= 1) tween = null;
    }
    if (!still && !controls.autoRotate && now - idleSince > 15000) controls.autoRotate = true;
    controls.update();
    const dist = camera.position.distanceTo(controls.target);
    const fromSun = camera.position.length();
    dustMat.uniforms.uFade.value = 0.35 + 0.65 * Math.min(Math.max((fromSun - 300) / 5000, 0), 1);
    sun.scale.setScalar(Math.max(fromSun * 0.004, 0.02));
    for (const r of rings) (r.line.material as THREE.LineBasicMaterial).opacity = 0.5 * ringAlpha(r.r, fromSun);
    renderer.render(scene, camera);

    const w = canvas.clientWidth, h = canvas.clientHeight;
    const taken: number[][] = [];
    for (const l of labels) {
      v.copy(l.at).project(camera);
      const a = l.show ? l.show(fromSun) : 1;
      // labels near the right edge hang to the left of their point
      const x = ((v.x + 1) / 2) * w, y = ((1 - v.y) / 2) * h, flip = x > w - 240;
      let hide = v.z > 1 || Math.abs(v.x) > 1.1 || Math.abs(v.y) > 1.1 || a < 0.02;
      if (!hide) {
        l.w ??= l.node.offsetWidth;
        l.h ??= l.node.offsetHeight;
        const box = [flip ? x - l.w : x, y, flip ? x : x + l.w, y + l.h];
        hide = taken.some((t) => box[0] < t[2] && box[2] > t[0] && box[1] < t[3] && box[3] > t[1]);
        if (!hide) taken.push(box);
      }
      l.node.style.opacity = hide ? "0" : String(a);
      l.node.style.visibility = hide ? "hidden" : "visible";
      l.node.style.transform = `translate(${x}px, ${y}px)${flip ? " translateX(-100%)" : ""}`;
      l.node.classList.toggle("flip", flip);
    }
    // a scale bar of round light-years about 110 px long, at the distance the view is centred on
    const per = pcPerPx(dist) * LY;
    const raw = per * 110, mag = 10 ** Math.floor(Math.log10(raw));
    const nice = [1, 2, 5, 10].map((m) => m * mag).reduce((a, b) => (Math.abs(b - raw) < Math.abs(a - raw) ? b : a));
    const text = `${nice < 1 ? fix(nice, 1) : int(nice)} light-year${nice === 1 ? "" : "s"}`;
    if (text !== lastScale) {
      lastScale = text;
      scaleText.textContent = text;
    }
    scaleBar.style.width = `${nice / per}px`;
  }

  const at = new URLSearchParams(location.search).get("at");
  const start = at ? find(at) : undefined;
  if (introDone || still || start) {
    introDone = true;
    uniforms.uYear.value = 3000;
    camera.position.copy(HOME.dir).multiplyScalar(HOME.dist);
    if (start) select(start);
    else overview();
  } else startIntro();
  resize();
  raf = requestAnimationFrame(frame);

  return {
    showHost(name) {
      const s = find(name);
      if (!s) return false;
      select(s);
      return true;
    },
    dispose() {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      for (const g of [geom, selGeom, dustGeom]) g.dispose();
      material.dispose();
      dustMat.dispose();
      renderer.dispose();
    },
  };
}
