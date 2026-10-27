// The window everything happens in: every planet found around another star and every TESS candidate still
// waiting, at its true place around the Sun (NASA Exoplanet Archive), inside a sketch of the Milky Way drawn to the
// same scale. On the first visit the planets arrive in the order they were found. Drag to turn, scroll or pinch to
// zoom, click a point to fly into its system, drawn to scale.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { getPlanets, getStar, getSummary } from "./data";
import type { Planets, StarPage, Summary } from "./data";
import { STOPS } from "./tour";
import type { Group } from "./tour";
import { tourCard } from "./tourcards";
import { searchCard } from "./search/card";
import { AU_PER_PC, jdNow, systemView } from "./system";
import type { SystemView } from "./system";
import { css, el, fix, int, navigate } from "./util";

const D2R = Math.PI / 180;
const LY = 3.26156;  // light-years per parsec
const GC = 8178;     // parsecs from the Sun to the centre of the galaxy (GRAVITY Collaboration 2019)
const NOW = new Date().getFullYear();
// ICRS to galactic unit vectors: x to the galactic centre, y to longitude 90, z to the north galactic pole
const T = [-0.0548755604, -0.8734370902, -0.4838350155, 0.4941094279, -0.44482963, 0.7469822445,
  -0.867666149, -0.1980763734, 0.4559837762];
const HOME = { dist: 3000, dir: new THREE.Vector3(-0.35, 0.62, 1).normalize() };
const INTRO_FROM = 1200;  // close enough to see 1992's pulsar planets, 600 pc out, flare

// Colour is how hot a world is (its estimated equilibrium temperature): pale near Earth's 255 K, bluer when
// colder, orange to red when hotter, a diverging scale on log temperature. Planets with no estimate, TESS
// candidates still waiting and Tessa's own keep one colour each. How a planet was found is a filter, not a colour.
// Most known worlds sit between 320 and 1,800 K, so the hot side gets most of the hues.
const HEAT: [number, string][] = [[60, "#4f7cff"], [180, "#93b6ff"], [255, "#f4f1ea"], [420, "#ffe6a3"], [700, "#ffbf5e"],
  [1100, "#ff8a4c"], [1600, "#ff5c5c"], [2400, "#e2559b"]];
// Tessa's own are teal, a hue no planet uses, so they never blend in
const NO_HEAT = "#b9b6ae", WAITING = "#7484b0", TESSA = "#5eead4";
const FILTERS: { key: Group; label: string; title: string }[] = [
  { key: "transit", label: "Transit", title: "It crossed in front of its star and dimmed it" },
  { key: "rv", label: "Wobble", title: "Its pull made its star wobble" },
  { key: "lens", label: "Lensing", title: "Its gravity bent the light of a star far behind it" },
  { key: "other", label: "Other", title: "Photographs of the planet itself, timing and the rest" },
];
const GROUP_OF: Record<string, Group> = { Transit: "transit", "Radial Velocity": "rv", Microlensing: "lens" };
// How each was found, in a word, for the system card
const HOW: Record<string, string> = {
  Transit: "transit", "Radial Velocity": "wobble", Microlensing: "lensing", Imaging: "photo", "Pulsar Timing": "pulsar",
  "Transit Timing Variations": "timing", "Eclipse Timing Variations": "timing", "Orbital Brightness Modulation": "glow",
  Astrometry: "astrometry", "Pulsation Timing Variations": "timing", "Disk Kinematics": "disk",
};

function heat(t: number | null): THREE.Color {
  if (t == null) return new THREE.Color(NO_HEAT);
  const x = Math.log(Math.min(Math.max(t, HEAT[0][0]), HEAT[HEAT.length - 1][0]));
  let k = 1;
  while (k < HEAT.length - 1 && x > Math.log(HEAT[k][0])) k++;
  const a = Math.log(HEAT[k - 1][0]), b = Math.log(HEAT[k][0]);
  return new THREE.Color(HEAT[k - 1][1]).lerp(new THREE.Color(HEAT[k][1]), (x - a) / (b - a));
}
const heatHex = (t: number | null) => `#${heat(t).getHexString()}`;
// the colour a planet wears on the map, so the card matches it
const planetHex = (p: Planets, i: number) => (p.kind[i] === 2 ? TESSA : p.kind[i] === 1 ? WAITING : heatHex(p.teq[i]));

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

// A star's place in the scene from RA, Dec (degrees) and distance (parsecs). Scene axes: x to the galactic centre,
// y to the north galactic pole, z = -galactic y (still right-handed).
function place(ra: number, dec: number, d: number): THREE.Vector3 {
  const a = ra * D2R, b = dec * D2R;
  const e0 = Math.cos(b) * Math.cos(a), e1 = Math.cos(b) * Math.sin(a), e2 = Math.sin(b);
  const gx = T[0] * e0 + T[1] * e1 + T[2] * e2, gy = T[3] * e0 + T[4] * e1 + T[5] * e2, gz = T[6] * e0 + T[7] * e1 + T[8] * e2;
  return new THREE.Vector3(d * gx, d * gz, -d * gy);
}

function systems(p: Planets): System[] {
  const by = new Map<string, System>();
  p.kind.forEach((kind, i) => {
    const key = `${kind}|${p.host[i]}`;
    let s = by.get(key);
    if (!s) {
      const d = p.dist[i];
      s = { host: p.host[i], kind, group: "other", rows: [], at: place(p.ra[i], p.dec[i], d), dist: d, year: Infinity };
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
  uniform float uTime;
  varying vec3 vColor;
  varying float vShape;
  varying float vAlpha;
  void main() {
    float age = uYear - year;                 // years since it was found, on the intro's clock
    float on = clamp(age / 0.7, 0.0, 1.0);
    vColor = color; vShape = shape; vAlpha = alpha * on;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    float beat = shape > 1.5 && shape < 2.5 ? 1.0 + 0.14 * sin(uTime * 2.2) : 1.0;  // Tessa's finds breathe
    gl_PointSize = age < 0.0 ? 0.0 : size * dpr * beat * (1.0 + 1.6 * (1.0 - on));  // a new find flares, then settles
  }`;

const pointFragment = /* glsl */ `
  varying vec3 vColor;
  varying float vShape;
  varying float vAlpha;
  void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r = length(p);
    float glow = 0.5 * pow(max(1.0 - r, 0.0), 2.6);
    float a;
    if (vShape < 0.5) a = max(1.0 - smoothstep(0.2, 0.32, r), glow);                            // a world: core and glow
    else if (vShape < 1.5) a = 1.0 - smoothstep(0.04, 0.1, abs(r - 0.64));                       // the selection ring
    else if (vShape < 2.5) a = max(1.0 - smoothstep(0.36, 0.46, abs(p.x) + abs(p.y)), glow * 1.6); // Tessa's
    else a = 0.9 * pow(max(1.0 - r, 0.0), 1.6);                                                   // waiting: faint dust
    a *= vAlpha;
    if (a < 0.01) discard;
    gl_FragColor = vec4(vColor, a);
    #include <colorspace_fragment>
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
const words = (k: number) => ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"][k] ?? int(k);
const dot = (color: string) => el("span", { class: "u-dot", style: `background: ${color}` });

// The year the light we see tonight set off: a date for near stars, a rounded era for far ones
function lightYear(pc: number): string {
  const ly = pc * LY;
  const y = ly < 1000 ? NOW - Math.round(ly) : Math.round((NOW - Number(ly.toPrecision(2))) / 100) * 100;
  return y > 0 ? String(y) : `${int(Math.max(-y, 1))} BCE`;
}
const orbitText = (d: number) => (d < 2 ? `${int(d * 24)} hours` : d < 730 ? `${fix(d, d < 10 ? 1 : 0)} days` : `${fix(d / 365.25, 1)} years`);
const tempText = (k: number) => `${int(Math.round((k - 273.15) / 10) * 10)} °C`;

interface World { showHost(host: string): boolean; search(n: number): void; dispose(): void }

let introDone = false;
let current: World | null = null;

export async function universe(main: HTMLElement) {
  // the tour reads its counts from the summary; without it the map still works, just without a tour
  const [data, sum] = await Promise.all([getPlanets(), getSummary().catch(() => null)]);
  // the tour tells the story of Tessa's candidate, from its case file (light curve, checks, predictions)
  const i = data.kind.indexOf(2);
  const star = i >= 0 ? await getStar(Number(data.host[i].replace(/^TIC /, ""))).catch(() => null) : null;
  current = mount(main, data, sum, star);
}

export function leaveUniverse() {
  current?.dispose();
  current = null;
}

// Ask Tessa's "show it on the map", for a star in this view (TESS candidates and Tessa's own are named by TIC)
export function flyToTic(tic: number): boolean {
  return current?.showHost(`TIC ${tic}`) ?? false;
}
// Ask Tessa's search button: here if the universe is open, otherwise by going to it
export function startSearch(n: number) {
  if (current) current.search(n);
  else navigate(`/search?go=${n}`);
}

function mount(host: HTMLElement, p: Planets, sum: Summary | null, star: StarPage | null): World | null {
  const sys = systems(p);
  const tour = sum && star ? STOPS : [];
  const ctx = { p, s: sum as Summary, star, host: star ? `TIC ${star.tic}` : "" };
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
  renderer.autoClear = false;  // the system view draws over the universe in the same frame
  const tip = el("div", { class: "u-tip", hidden: true });
  const labelLayer = el("div", { class: "u-labels", "aria-hidden": "true" });
  const card = el("section", { class: "u-card", "aria-live": "polite" });
  const scaleBar = el("span", { class: "u-scale-bar" });
  const scaleText = el("span", {});
  const centred = el("span", { class: "u-centred" });
  const hint = el("p", { class: "u-hint" }, "Drag to turn, scroll to zoom, click a world");
  const readout = el("div", { class: "u-readout", "aria-hidden": "true" }, hint, el("div", {}, scaleBar, scaleText), centred);
  const dim = el("div", { class: "t-dim" });  // darkens the universe behind a tour card with its own picture
  section.append(canvas, labelLayer, dim, tip, card, readout);
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
    const rMax = Math.max(...q.rows.map((r) => p.radius[r] ?? 0)) || 2.5;
    size[i] = q.kind === 2 ? 22 : q.kind === 1 ? 5 : 6.5 + Math.min(Math.max(2.4 * Math.log2(rMax), -1.5), 8) + (q.rows.length > 1 ? 1 : 0);
    shape[i] = q.kind === 2 ? 2 : q.kind === 1 ? 3 : 0;
    year[i] = q.kind === 2 ? 2028 : q.kind === 1 ? 2027 : q.year;
  });
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  for (const [name, arr, k] of [["color", col, 3], ["size", size, 1], ["shape", shape, 1], ["alpha", alpha, 1], ["year", year, 1]] as const) {
    geom.setAttribute(name, new THREE.BufferAttribute(arr, k));
  }
  const uniforms = { dpr: { value: 1 }, uYear: { value: introDone ? 3000 : 1990 }, uTime: { value: 0 } };
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
  const circle = (r: number, color: string) => {
    const pts = Array.from({ length: 257 }, (_, i) => new THREE.Vector3(r * Math.cos((i / 256) * Math.PI * 2), 0, r * Math.sin((i / 256) * Math.PI * 2)));
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0 }));
    scene.add(line);
    return line;
  };
  const RINGS = [100, 1000, 10000];
  const rings = RINGS.map((ly) => ({ r: ly / LY, line: circle(ly / LY, css("--axis")) }));
  // Tessa's patch: every star it searches lies within 100 parsecs; the tour shows the edge
  const patch = circle(100, TESSA);

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
    const l: Label = { node, at, rank, show };
    labelLayer.append(node);
    labels.push(l);
    labels.sort((a, b) => a.rank - b.rank);
    return l;
  };
  const unlabel = (l: Label) => {
    l.node.remove();
    labels.splice(labels.indexOf(l), 1);
  };
  label(el("span", { class: "u-label sun" }, "Sun"), new THREE.Vector3(0, 0, 0), 0, (d) => (d < 15000 ? 1 : 0));
  label(el("span", { class: "u-label sun" }, "You are here", el("small", {}, "Nearly every planet we know is in this corner")),
    new THREE.Vector3(0, 0, 0), 0, (d) => (d >= 15000 ? 1 : 0));
  label(el("span", { class: "u-label" }, "Centre of the Milky Way", el("small", {}, `${lyNum(GC)} light-years`)),
    new THREE.Vector3(GC, 0, 0), 3, (d) => (d > 4000 ? 1 : 0));
  RINGS.forEach((ly, k) => {
    const r = rings[k].r, a = -0.6;
    label(el("span", { class: "u-label ring" }, `${int(ly)} light-years`, el("small", {}, `light from here left in ${ly < NOW ? NOW - ly : lightYear(r)}`)),
    new THREE.Vector3(r * Math.cos(a), 0, r * Math.sin(a)), 2, (d) => ringAlpha(r, d, 160));
  });
  sys.filter((s) => s.kind === 2).forEach((s) => {
    const r0 = s.rows[0], rp = p.radius[r0], per = p.period[r0];
    const b = el("button", { type: "button", class: "u-label tessa", tabindex: -1 }, s.host, el("small", {}, "found by Tessa"),
      rp && per ? el("small", {}, `${rp.toFixed(1)} times Earth's width, ${Math.round(per * 24)}-hour year`) : null);
    b.addEventListener("click", () => select(s));
    label(b, s.at, 1, (d) => (uniforms.uYear.value > 2028 && d < 8000 && !(svSys === s && svFade > 0) ? 1 : 0));
  });
  // the star a visitor is searching
  const searchTag = label(el("span", { class: "u-label tessa" }), new THREE.Vector3(), 1, () => (searching ? 1 : 0));
  label(el("span", { class: "u-label patch" }, "Tessa's patch", el("small", {}, "within 326 light-years")),
    new THREE.Vector3(100 * Math.cos(2.3), 0, 100 * Math.sin(2.3)), 1, () => (tour[stopAt]?.patch ? 1 : 0));

  // a ring reads best when it spans a good part of the screen
  function ringAlpha(r: number, d: number, min = 40): number {
    const px = r / pcPerPx(d);
    const h = canvas.clientHeight || 800;
    return Math.min(Math.max((px - min) / 80, 0), 1) * Math.min(Math.max((h * 1.6 - px) / (h * 0.8), 0), 1);
  }
  function pcPerPx(d: number) {
    return (2 * d * Math.tan((camera.fov * D2R) / 2)) / (canvas.clientHeight || 800);
  }

  // ---- the system view: the selected system to scale in its own scene, measured in AU (system.ts)
  let sv: SystemView | null = null, svSys: System | null = null, svFade = 0, hostHidden = false;
  let svLabels: [Label, { at: THREE.Vector3 }][] = [];
  const svCam = new THREE.PerspectiveCamera();
  // the planets' clock: today, then a day every second unless the visitor picks real time
  let simJd = jdNow(), simAt = 0, simSpeed = 86400;
  const simNow = (now: number) => simJd + ((now - simAt) / 86400000) * simSpeed;
  const simDate = el("span", { class: "u-time-date" });
  let lastDate = "";
  function openSystem(s: System) {
    closeSystem();
    const rows = [...s.rows].sort((a, b) => (p.period[a] ?? 1e9) - (p.period[b] ?? 1e9));
    sv = systemView(p, rows, s.at, (i) => planetHex(p, i));
    svSys = s;
    simJd = jdNow();
    simAt = performance.now();
    controls.minDistance = sv.minDist / AU_PER_PC;
    const short = (name: string) => (name === s.host ? "candidate" : name.startsWith(`${s.host} `) ? name.slice(s.host.length + 1) : name);
    svLabels = sv.marks.map((m) => {
      const node = m.kind === "planet" ? el("span", { class: "u-label sys" }, short(p.name[m.row!]))
        : m.kind === "earth" ? el("span", { class: "u-label ring" }, "Earth's orbit", el("small", {}, "for scale"))
        : m.kind === "hz" ? el("span", { class: "u-label ring" }, "Water could be liquid here")
        : el("span", { class: "u-label ring" }, "To Earth");
      return [label(node, new THREE.Vector3(), m.kind === "planet" ? 1 : 2, () => (svFade > 0.6 ? 1 : 0)), m];
    });
  }
  function closeSystem() {
    if (!sv) return;
    sv.dispose();
    sv = null;
    svSys = null;
    svFade = 0;
    for (const [l] of svLabels) unlabel(l);
    svLabels = [];
    controls.minDistance = 0.3;
    if (hostHidden) { hostHidden = false; paint(); }
  }

  // ---- colour and focus
  let focus: Group | null = null;
  const hottest = (q: System) => {
    const t = q.rows.map((i) => p.teq[i]).filter((x): x is number => x != null);
    return t.length ? Math.max(...t) : null;
  };
  function paint() {
    order.forEach((s, i) => {
      const q = sys[s];
      const c = q.kind === 2 ? new THREE.Color(TESSA) : q.kind === 1 ? new THREE.Color(WAITING) : heat(hottest(q));
      col.set([c.r, c.g, c.b], i * 3);
      const base = q.kind === 1 ? 0.5 : 1;
      // inside a system its own point would sit over the drawing of it
      alpha[i] = q === svSys && hostHidden ? 0 : focus && q.group !== focus ? base * 0.06 : base;
    });
    geom.attributes.color.needsUpdate = true;
    geom.attributes.alpha.needsUpdate = true;
  }
  paint();

  // ---- the card: intro, overview, or one system
  const toggles: [Group, HTMLButtonElement][] = [];
  function toggle(key: Group, kids: (Node | string)[], title?: string) {
    const b = el("button", { type: "button", "aria-pressed": String(focus === key), title }, ...kids);
    b.addEventListener("click", () => {
      focus = focus === key ? null : key;
      for (const [k, x] of toggles) x.setAttribute("aria-pressed", String(k === focus));
      paint();
    });
    toggles.push([key, b]);
    return b;
  }
  function bar(left: (Node | string)[], right?: Node | null) {
    return el("div", { class: "u-bar" }, el("span", { class: "u-bar-title" }, ...left), right ?? null);
  }
  function heatKey() {
    const lo = Math.log(HEAT[0][0]), hi = Math.log(HEAT[HEAT.length - 1][0]);
    const at = (t: number) => ((Math.log(t) - lo) / (hi - lo)) * 100;
    const grad = HEAT.map(([t, c]) => `${c} ${at(t).toFixed(1)}%`).join(", ");
    return el("div", { class: "u-heat" },
      el("span", { class: "u-heat-bar", style: `background: linear-gradient(90deg, ${grad})` }),
      el("span", { class: "u-heat-labels" }, el("span", {}, "colder"),
        el("span", { class: "mid", style: `left: ${at(255).toFixed(1)}%` }, "Earth"), el("span", {}, "hotter")));
  }

  // ?at=51 Peg (a star or a planet) opens straight onto that system and ?tour=3 onto the tour's third stop; the
  // address always says what is open
  function setAt(name: string | null, stop?: number) {
    const u = new URL(location.href);
    if (name) u.searchParams.set("at", name);
    else u.searchParams.delete("at");
    if (stop) u.searchParams.set("tour", String(stop));
    else u.searchParams.delete("tour");
    history.replaceState(history.state, "", u.pathname + u.search + u.hash);
  }
  function find(name: string): System | undefined {
    const q = name.trim().toLowerCase();
    return sys.find((s) => s.host.toLowerCase() === q || s.rows.some((i) => p.name[i].toLowerCase() === q));
  }

  function overview() {
    closeSystem();
    setAt(null);
    selPoint.visible = false;
    selected = null;
    centred.textContent = "Centred on the Sun";
    toggles.length = 0;
    const replay = el("button", { type: "button", class: "u-bar-btn" }, "Replay");
    replay.addEventListener("click", () => startIntro());
    const c = p.counts;
    card.replaceChildren(
      bar([dot("var(--ok)"), "Every known planet"], replay),
      el("div", { class: "u-body" },
        el("p", { class: "u-big" }, int(c.confirmed)),
        el("p", { class: "u-sub" }, "planets found around other stars"),
        el("div", { class: "u-actions u-start" }, tour.length ? tourButton() : null, sum ? searchButton() : null),
        el("div", { class: "u-stats" },
          toggle("waiting", [el("b", {}, int(c.waiting)), el("span", {}, "waiting to be confirmed")]),
          toggle("tessa", [el("b", {}, int(c.tessa)), el("span", {}, "found by Tessa")])),
        el("div", { class: "u-key" }, el("span", { class: "u-k" }, "Colour"), heatKey()),
        el("div", { class: "u-key" }, el("span", { class: "u-k" }, "Found by"),
          el("div", { class: "u-chips" }, ...FILTERS.map((f) => toggle(f.key, [f.label], f.title)))),
        el("details", { class: "u-about" }, el("summary", {}, "About this map"),
          el("p", {}, `Every point sits at its measured distance, from the NASA Exoplanet Archive. ${int(c.unplaced.confirmed)} planets `
            + `and ${int(c.unplaced.waiting)} candidates have no distance yet and aren't shown. Colour is each planet's estimated `
            + "temperature. The Milky Way is a sketch at the same scale."),
          el("p", {}, el("a", { href: "/survey", "data-link": true }, "Tessa's survey, in full")))));
  }

  const introYear = el("p", { class: "u-year" });
  const introCount = el("p", { class: "u-count" });
  const introNote = el("p", { class: "u-note" });
  function introCard() {
    const skip = el("button", { type: "button", class: "u-bar-btn" }, "Skip");
    skip.addEventListener("click", () => endIntro());
    card.replaceChildren(bar([dot("var(--ok)"), "1992 to today"], skip),
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
    const kindWord = s.kind === 2 ? "found by Tessa" : s.kind === 1 ? "TESS candidate" : `${s.rows.length} planet${s.rows.length > 1 ? "s" : ""}`;
    const rows = [...s.rows].sort((a, b) => (p.period[a] ?? 1e9) - (p.period[b] ?? 1e9));
    const short = (name: string) => (name === s.host ? "candidate" : name.startsWith(`${s.host} `) ? name.slice(s.host.length + 1) : name);
    const who = (i: number) => (s.kind === 2 ? "Tessa" : s.kind === 1 ? "TESS" : HOW[p.methods[p.method[i]]] ?? "");
    const table = el("table", { class: "u-table" },
      el("thead", {}, el("tr", {}, ...["", "Size", "Year", "Temp", "Found"].map((h) => el("th", { scope: "col" }, h)))),
      el("tbody", {}, ...rows.map((i) => {
        const r = p.radius[i], per = p.period[i], t = p.teq[i];
        return el("tr", {},
          el("th", { scope: "row", title: p.name[i] }, dot(planetHex(p, i)), short(p.name[i])),
          el("td", {}, r == null ? "–" : `${r < 10 ? fix(r, 1) : int(r)} Earths`),
          el("td", {}, per == null ? "–" : orbitText(per)),
          el("td", {}, t == null ? "–" : tempText(t)),
          el("td", {}, String(p.year[i]), el("small", {}, who(i))));
      })));
    const tic = s.host.match(/^TIC (\d+)$/)?.[1];
    const stat = (big: string, small: string) => el("div", {}, el("b", {}, big), el("span", {}, small));
    card.replaceChildren(bar([back], el("span", { class: "u-bar-tag" }, kindWord)),
      el("div", { class: "u-body" },
        el("h2", {}, s.host),
        el("div", { class: "u-pair" }, stat(lyNum(s.dist), "light-years away"), stat(lightYear(s.dist), "when its light left")),
        sv ? timeControl() : null,
        table,
        el("p", { class: "u-legendnote" }, "Size is width, Earth = 1. Year is one orbit. Temp is an estimate."),
        sv ? el("p", { class: "u-legendnote" }, [
          "The star and the orbits are to scale; the planets are drawn larger, or you couldn't see them.",
          sv.transits ? "Orbits that transit line up with the dashed line to Earth: that's why their planet dims its star." : "",
          sv.guessed ? "A dashed orbit is partly a guess: its tilt, size or place hasn't been measured." : "",
          sv.hz ? "The green band is where a planet could keep liquid water." : "",
        ].filter(Boolean).join(" ")) : null,
        s.kind === 2 && tic ? el("a", { href: `/star/${tic}`, "data-link": true, class: "u-btn" }, "Open the case file") : null,
        share()));
  }
  function timeControl() {
    const speeds: [string, number][] = [["A day a second", 86400], ["Real time", 1]];
    const buttons = speeds.map(([name, v]) => {
      const b = el("button", { type: "button", "aria-pressed": String(simSpeed === v) }, name);
      b.addEventListener("click", () => {
        simJd = simNow(performance.now());
        simAt = performance.now();
        simSpeed = v;
        if (v === 1) simJd = jdNow();  // real time starts from now
        buttons.forEach((x, k) => x.setAttribute("aria-pressed", String(speeds[k][1] === v)));
      });
      return b;
    });
    return el("div", { class: "u-time" }, el("span", { class: "u-k" }, "Planets on"), simDate, el("div", { class: "u-chips" }, ...buttons));
  }
  function share() {
    const b = el("button", { type: "button", class: "u-bar-btn u-share" }, "Copy link");
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

  function ring(s: { at: THREE.Vector3 } | null) {
    selPoint.visible = !!s;
    if (!s) return;
    (selGeom.attributes.position.array as Float32Array).set([s.at.x, s.at.y, s.at.z]);
    selGeom.attributes.position.needsUpdate = true;
  }
  function select(s: System) {
    endIntro(false);
    leaveTour(false);
    leaveSearch(false);
    selected = s;
    ring(s);
    hint.hidden = true;
    setAt(s.host);
    centred.textContent = `Centred on ${s.host}`;
    openSystem(s);
    systemCard(s);
    // into the system, framing its widest orbit, from a little above the orbits
    flyTo(s.at, (sv!.outer * 3) / AU_PER_PC, sv!.fitDir, 3200);
  }

  // ---- camera moves
  type Tween = { from: THREE.Vector3; to: THREE.Vector3; fromD: number; toD: number; fromDir: THREE.Vector3; toDir: THREE.Vector3;
    t0: number; ms: number; deep: boolean };
  let tween: Tween | null = null;
  // fly the view's centre to a point and end up dist away from it, looking along dir if given
  function flyTo(to: THREE.Vector3, dist: number, dir?: THREE.Vector3, ms = 1800) {
    const from = camera.position.clone().sub(controls.target).normalize();
    tween = { from: controls.target.clone(), to: to.clone(), fromD: camera.position.distanceTo(controls.target), toD: dist,
      fromDir: from, toDir: (dir ?? from).clone().normalize(), t0: performance.now(), ms: still ? 1 : ms,
      deep: dist < camera.position.distanceTo(controls.target) * 1e-3 };
  }
  const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

  let intro: { t0: number } | null = null;
  function startIntro() {
    if (still) { endIntro(); return; }
    closeSystem();
    focus = null;
    paint();
    selected = null;
    selPoint.visible = false;
    tween = null;
    stopAt = -1;
    tourSys = null;
    clock = null;
    tourEl?.dispose();
    tourEl = null;
    card.hidden = false;
    section.classList.remove("touring");
    dim.classList.remove("on");
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

  // ---- the tour: each stop sets the universe behind its card: a flight, a year on the discovery clock, a filter
  let stopAt = -1;
  let tourSys: System | null = null;
  let tourEl: { node: HTMLElement; dispose(): void } | null = null;
  let clock: { from: number; to: number; t0: number; ms: number } | null = null;
  // run the clock to a year: planets found later fade out, ones found by then flare in
  function runClock(to: number, ms: number) {
    clock = { from: Math.min(uniforms.uYear.value, 2028.9), to, t0: performance.now(), ms: still ? 1 : ms };
  }
  // show the Sun and a system together, looking across the line between them from a little above, in the space
  // the tour card leaves free
  function withSun(at: THREE.Vector3) {
    const side = new THREE.Vector3().crossVectors(at, new THREE.Vector3(0, 1, 0));
    if (side.lengthSq() < 1e-6) side.copy(HOME.dir);
    side.normalize();
    if (side.dot(camera.position.clone().sub(controls.target)) < 0) side.negate();
    side.y += 0.45;
    const fit = Math.max(at.length() * (innerWidth > 900 ? 1.6 : 2.6), 30);
    flyTo(at.clone().multiplyScalar(0.6), fit, side, 2400);  // a little nearer the system than the Sun: it is the subject
  }
  function tourButton() {
    const b = el("button", { type: "button", class: "u-btn u-tour" }, "Take the tour");
    b.addEventListener("click", () => tourStop(0));
    return b;
  }
  function tourStop(k: number) {
    endIntro(false);
    closeSystem();
    const s = tour[k];
    stopAt = k;
    selected = null;
    hint.hidden = true;
    controls.autoRotate = false;
    idleSince = performance.now();
    focus = s.focus ?? null;
    paint();
    runClock(s.year, 2400);
    tourSys = s.at ? find(ctx.host) ?? null : null;
    ring(tourSys);
    dim.classList.toggle("on", !!s.dim);
    card.hidden = true;
    section.classList.add("touring");  // the scale readout gives way to the tour's words
    tourEl?.dispose();
    tourEl = tourCard({ kind: s.card, ctx, index: k, total: tour.length, onSearch: () => searchMode(),
      onNext: () => (k < tour.length - 1 ? tourStop(k + 1) : leaveTour()),
      onBack: () => { if (k > 0) tourStop(k - 1); },
      onLeave: () => leaveTour(),
      onFocus: (g) => { focus = g; paint(); } });
    section.append(tourEl.node);
    ro.observe(tourEl.node.firstElementChild as HTMLElement);
    resize();
    if (tourSys && s.dist) {
      centred.textContent = `Centred on ${tourSys.host}`;
      flyTo(tourSys.at, s.dist, undefined, 2400);
    } else if (tourSys) {
      centred.textContent = `The Sun and ${tourSys.host}`;
      withSun(tourSys.at);
    } else {
      centred.textContent = "Centred on the Sun";
      flyTo(new THREE.Vector3(), s.dist ?? HOME.dist, HOME.dir, 2400);
    }
    setAt(null, k + 1);
  }
  // ---- "Search a star": the visitor's own search of a held-back star, in the tour's layout
  let searching = false;
  const g1 = (sum?.survey_gate1 ?? { floor: 14.3 }) as { floor: number };
  const odds = sum && sum.by_status.candidate ? `1 in ${int(Number((Number(sum.searched_total ?? sum.survey_searched) / sum.by_status.candidate).toPrecision(2)))}` : "1 in 5,000";
  function searchButton() {
    const b = el("button", { type: "button", class: "u-btn u-tour" }, "Search a star");
    b.addEventListener("click", () => searchMode());
    return b;
  }
  // start: a batch size to begin at once (/search?go=10, Ask Tessa's button)
  function searchMode(start = 0) {
    endIntro(false);
    closeSystem();
    leaveTour(false);
    tourEl?.dispose();
    searching = true;
    selected = null;
    hint.hidden = true;
    focus = null;
    paint();
    runClock(3000, 1200);
    ring(null);
    dim.classList.remove("on");
    card.hidden = true;
    section.classList.add("touring");
    history.replaceState(history.state, "", "/search");
    tourEl = searchCard({ floor: g1.floor, odds, start, onLeave: () => leaveSearch(), onAgain: () => searchMode(),
      onStar: (st) => {
        const at = place(st.ra, st.dec, st.dist);
        ring({ at });
        searchTag.at.copy(at);
        searchTag.node.replaceChildren(`TIC ${st.tic}`, el("small", {}, "your star"));
        searchTag.w = searchTag.h = undefined;
        centred.textContent = `The Sun and TIC ${st.tic}`;
        withSun(at);
      } });
    section.append(tourEl.node);
    ro.observe(tourEl.node.firstElementChild as HTMLElement);
    resize();
    centred.textContent = "Centred on the Sun";
    flyTo(new THREE.Vector3(), 260, HOME.dir, 2400);
  }
  function leaveSearch(toOverview = true) {
    if (!searching) return;
    searching = false;
    tourEl?.dispose();
    tourEl = null;
    ring(null);
    card.hidden = false;
    section.classList.remove("touring");
    history.replaceState(history.state, "", "/");
    resize();
    if (toOverview) {
      overview();
      flyTo(new THREE.Vector3(), HOME.dist, HOME.dir, 2400);
    }
  }
  function leaveTour(toOverview = true) {
    if (stopAt < 0) return;
    stopAt = -1;
    tourSys = null;
    tourEl?.dispose();
    tourEl = null;
    dim.classList.remove("on");
    card.hidden = false;
    section.classList.remove("touring");
    runClock(3000, 1200);
    focus = null;
    paint();
    resize();
    if (toOverview) {
      overview();
      flyTo(new THREE.Vector3(), HOME.dist, HOME.dir, 2400);
    }
  }

  // ---- input
  controls.addEventListener("start", () => {
    hint.hidden = true;
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
    // a click mid-tour or mid-search would fly off and break it
    if (stopAt >= 0 || searching) return;
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
    else if (ev.key === "Escape" && stopAt >= 0) { leaveTour(); return; }
    else return;
    ev.preventDefault();
    if (intro) endIntro();
    controls.autoRotate = false;
    idleSince = performance.now();
    camera.position.copy(controls.target).add(new THREE.Vector3().setFromSpherical(sph));
  });

  // ---- frame
  let keepOut: number[][] = [];
  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    renderer.setPixelRatio(dpr);
    renderer.setSize(w, h, false);
    uniforms.dpr.value = dpr;
    camera.aspect = w / h;
    // on wide screens the card (or the tour's words) covers the left, so centre the view in the space right of it
    // (the tour's words run along the bottom instead, so then nothing shifts sideways)
    const side = tourEl ? null : card;
    const shift = side && w > 900 ? Math.min(side.getBoundingClientRect().right - section.getBoundingClientRect().left, w * 0.45) / 2 : 0;
    camera.setViewOffset(w, h, -shift, w > 900 ? 0 : h * 0.16, w, h);
    camera.updateProjectionMatrix();
    // labels keep clear of the header, the card and the readout
    const top = document.querySelector<HTMLElement>(".bar")?.offsetHeight ?? 64;
    const base = section.getBoundingClientRect();
    keepOut = [[0, 0, w, top], ...[card, readout, ...(tourEl ? [...tourEl.node.querySelectorAll<HTMLElement>(".t-head,.t-say,.t-stat,.t-ctrl,.t-stage")] : [])]
      .filter((e) => e.offsetWidth > 0).map((e) => {
      const b = e.getBoundingClientRect();
      return [b.left - base.left, b.top - base.top, b.right - base.left, b.bottom - base.top];
    })];
  }
  const ro = new ResizeObserver(resize);
  ro.observe(canvas);
  ro.observe(card);
  ro.observe(readout);

  const v = new THREE.Vector3();
  let raf = 0;
  let lastScale = "";
  function frame(now: number) {
    raf = requestAnimationFrame(frame);
    if (!still) uniforms.uTime.value = now / 1000;
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
      introNote.textContent = y > 2027 ? "Faint dust: candidates waiting for a second look. Teal: Tessa's."
        : [...MILESTONES].reverse().find(([yr]) => yr <= shown)?.[1] ?? "";
      if (sec > 12.5) endIntro();
    } else if (tween) {
      const t = Math.min((now - tween.t0) / tween.ms, 1), e = ease(t);
      // diving into a system, the view's centre reaches the star in the first half and the rest is zoom, or the
      // camera would plunge into the empty space on the way and find the star only at the end
      controls.target.lerpVectors(tween.from, tween.to, tween.deep ? ease(Math.min(t / 0.5, 1)) : e);
      const d = Math.exp(Math.log(tween.fromD) + (Math.log(tween.toD) - Math.log(tween.fromD)) * e);
      v.lerpVectors(tween.fromDir, tween.toDir, e);
      if (v.lengthSq() < 1e-6) v.copy(tween.toDir);
      camera.position.copy(controls.target).addScaledVector(v.normalize(), d);
      if (t >= 1) tween = null;
    }
    if (clock) {
      const t = Math.min((now - clock.t0) / clock.ms, 1);
      uniforms.uYear.value = t < 1 ? clock.from + (Math.min(clock.to, 2028.9) - clock.from) * ease(t) : clock.to;
      if (t >= 1) clock = null;
    }
    if (!still && !controls.autoRotate && now - idleSince > 15000) controls.autoRotate = true;
    controls.update();
    const dist = camera.position.distanceTo(controls.target);
    // the near plane follows the view in, or a system a millionth of a parsec across would be clipped away
    const near = Math.min(Math.max(dist * 0.002, 1e-12), 0.05);
    if (Math.abs(camera.near / near - 1) > 0.25) {
      camera.near = near;
      camera.updateProjectionMatrix();
    }
    controls.zoomSpeed = dist < 0.5 ? 2.4 : 1.1;  // many powers of ten between a system and the galaxy
    if (sv && svSys) {
      const dAu = camera.position.distanceTo(svSys.at) * AU_PER_PC;
      svFade = Math.min(Math.max((sv.outer * 400 - dAu) / (sv.outer * 300), 0), 1);
      if (svFade > 0.3 !== hostHidden) {
        hostHidden = svFade > 0.3;
        paint();
        selPoint.visible = !hostHidden && selected === svSys;
      }
      if (svFade > 0) {
        const jd = simNow(now);
        sv.update(jd, dAu, svFade);
        svCam.copy(camera);
        svCam.position.copy(camera.position).sub(svSys.at).multiplyScalar(AU_PER_PC);
        svCam.near = Math.max(dAu * 0.001, 1e-7);
        svCam.far = dAu * 10 + sv.outer * 100;
        svCam.updateProjectionMatrix();
        const d = new Date((jd - 2440587.5) * 86400000);
        const text = d.toLocaleString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC",
          ...(simSpeed === 1 ? { hour: "2-digit", minute: "2-digit" } : {}) }) + (simSpeed === 1 ? " UTC" : "");
        if (text !== lastDate) { lastDate = text; simDate.textContent = text; }
      }
      for (const [l, m] of svLabels) l.at.copy(svSys.at).addScaledVector(m.at, 1 / AU_PER_PC);
    }
    const fromSun = camera.position.length();
    dustMat.uniforms.uFade.value = 0.35 + 0.65 * Math.min(Math.max((fromSun - 300) / 5000, 0), 1);
    sun.scale.setScalar(Math.max(fromSun * 0.004, 0.02));
    for (const r of rings) (r.line.material as THREE.LineBasicMaterial).opacity = 0.5 * ringAlpha(r.r, fromSun) * (1 - svFade);
    const pm = patch.material as THREE.LineBasicMaterial;
    pm.opacity += ((tour[stopAt]?.patch ? 0.6 : 0) - pm.opacity) * (still ? 1 : 0.06);
    renderer.clear();
    renderer.render(scene, camera);
    if (sv && svFade > 0) {
      renderer.clearDepth();
      renderer.render(sv.scene, svCam);
    }

    const w = canvas.clientWidth, h = canvas.clientHeight;
    const taken = [...keepOut];
    for (const l of labels) {
      v.copy(l.at).project(camera);
      const a = l.show ? l.show(fromSun) : 1;
      const x = ((v.x + 1) / 2) * w, y = ((1 - v.y) / 2) * h;
      let hide = v.z > 1 || Math.abs(v.x) > 1.1 || Math.abs(v.y) > 1.1 || a < 0.02;
      let flip = false;
      if (!hide) {
        l.w ??= l.node.offsetWidth;
        l.h ??= l.node.offsetHeight;
        // a label that would run off the right edge hangs to the left of its point
        flip = x + l.w > w - 12;
        const box = [flip ? x - l.w : x, y, flip ? x : x + l.w, y + l.h];
        hide = box[0] < 0 || box[2] > w || box[3] > h || taken.some((t) => box[0] < t[2] && box[2] > t[0] && box[1] < t[3] && box[3] > t[1]);
        if (!hide) taken.push(box);
      }
      l.node.style.opacity = hide ? "0" : String(a);
      l.node.style.visibility = hide ? "hidden" : "visible";
      l.node.style.transform = `translate(${x}px, ${y}px)${flip ? " translateX(-100%)" : ""}`;
      l.node.classList.toggle("flip", flip);
    }
    // a scale bar of round light-years about 110 px long, at the distance the view is centred on
    // light-years, or inside a system AU (the Sun to the Earth)
    const au = pcPerPx(dist) * LY * 110 < 0.005;
    const per = pcPerPx(dist) * (au ? AU_PER_PC : LY);
    const raw = per * 110, mag = 10 ** Math.floor(Math.log10(raw));
    const nice = [1, 2, 5, 10].map((m) => m * mag).reduce((a, b) => (Math.abs(b - raw) < Math.abs(a - raw) ? b : a));
    const text = au ? `${nice < 1 ? String(Number(nice.toPrecision(1))) : int(nice)} AU` + (nice === 1 ? ", the Sun to the Earth" : "")
      : `${nice < 1 ? fix(nice, 1) : int(nice)} light-year${nice === 1 ? "" : "s"}`;
    if (text !== lastScale) {
      lastScale = text;
      scaleText.textContent = text;
    }
    scaleBar.style.width = `${nice / per}px`;
  }

  const params = new URLSearchParams(location.search);
  const at = params.get("at"), stop = Number(params.get("tour")) - 1;
  const start = at ? find(at) : undefined;
  const onTour = stop >= 0 && stop < tour.length, onSearch = /^\/search\/?$/.test(location.pathname) && !!sum;
  if (introDone || still || start || onTour || onSearch) {
    introDone = true;
    uniforms.uYear.value = 3000;
    camera.position.copy(HOME.dir).multiplyScalar(HOME.dist);
    if (start) select(start);
    else if (onTour) tourStop(stop);
    else if (onSearch) searchMode([1, 10, 50, 100].includes(Number(params.get("go"))) ? Number(params.get("go")) : 0);
    else overview();
  } else startIntro();
  resize();
  raf = requestAnimationFrame(frame);

  return {
    search(n) { if (sum) searchMode(n); },
    showHost(name) {
      const s = find(name);
      if (!s) return false;
      select(s);
      return true;
    },
    dispose() {
      closeSystem();
      tourEl?.dispose();
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
