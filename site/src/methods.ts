// The four ways planets are found, each as a small 3D scene with its signal drawn live underneath, told one
// sentence at a time. Sizes and speeds are exaggerated so the effect is visible; every scene says so.
import * as THREE from "three";
import { el } from "./util";

export interface Beat { at: number; text: string; focus?: string[] }
// Something the tour plays: sentences on a clock, and a draw call for the picture at time t, step k
export interface Show { node: HTMLElement; beats: Beat[]; draw(t: number, k: number): void; dispose(): void }
export type Method = "dip" | "wobble" | "bent" | "photo";

const TAU = Math.PI * 2;
const css = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const wrapAngle = (th: number) => Math.atan2(Math.sin(th), Math.cos(th));
export const smooth = (a: number, b: number, x: number) => { const k = Math.min(Math.max((x - a) / (b - a), 0), 1); return k * k * (3 - 2 * k); };

// a 2D canvas sized to its box, cleared, with the label font set
export function fit(c: HTMLCanvasElement) {
  const dpr = Math.min(devicePixelRatio || 1, 2), w = c.clientWidth, h = c.clientHeight;
  if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
  const g = c.getContext("2d")!;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  g.font = "12px 'Courier Prime', monospace";
  g.textBaseline = "alphabetic";
  return { g, w, h };
}
// a plot area with both axes named
export function axes(c: HTMLCanvasElement, ytitle: string, xtitle: string) {
  const { g, w, h } = fit(c);
  const L = 20, B = h - 20, T = 6;
  g.strokeStyle = css("--axis"); g.lineWidth = 1;
  g.beginPath(); g.moveTo(L + 0.5, T); g.lineTo(L + 0.5, B + 0.5); g.lineTo(w, B + 0.5); g.stroke();
  g.fillStyle = css("--muted");
  g.save(); g.translate(13, B); g.rotate(-Math.PI / 2); g.fillText(ytitle, 0, 0); g.restore();
  g.fillText(xtitle, w - g.measureText(xtitle).width, h - 4);
  return { g, w, h, L, B, T };
}
// a curve traced as it happens, with a playhead; earlier passes stay faint
export function trace(g: CanvasRenderingContext2D, xOf: (q: number) => number, yOf: (q: number) => number,
  from: number, to: number, cur: number, faint: boolean, bottom: number) {
  const path = (a: number, b: number) => {
    g.beginPath();
    for (let i = 0; i <= 200; i++) { const q = a + ((b - a) * i) / 200; if (i) g.lineTo(xOf(q), yOf(q)); else g.moveTo(xOf(q), yOf(q)); }
    g.stroke();
  };
  g.lineWidth = 2; g.lineJoin = "round";
  if (faint) { g.strokeStyle = "rgba(236,235,230,0.22)"; path(from, to); }
  if (cur > from) { g.strokeStyle = css("--ink"); path(from, Math.min(cur, to)); }
  if (cur >= from && cur <= to) {
    const x = xOf(cur);
    g.strokeStyle = "rgba(236,235,230,0.3)"; g.lineWidth = 1; g.setLineDash([2, 3]);
    g.beginPath(); g.moveTo(x, 0); g.lineTo(x, bottom); g.stroke(); g.setLineDash([]);
    g.fillStyle = css("--ink"); g.beginPath(); g.arc(x, yOf(cur), 3.5, 0, 7); g.fill();
  }
}

function glow(rgb: string, inner = 0.55) {
  const c = document.createElement("canvas"); c.width = c.height = 128;
  const g = c.getContext("2d")!, gr = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  gr.addColorStop(0, `rgba(${rgb},${inner})`); gr.addColorStop(0.35, `rgba(${rgb},${inner * 0.3})`); gr.addColorStop(1, `rgba(${rgb},0)`);
  g.fillStyle = gr; g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}
const sprite = (rgb: string, inner: number, additive = false) => new THREE.Sprite(new THREE.SpriteMaterial({ map: glow(rgb, inner),
  depthWrite: false, transparent: true, blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending }));
// a star lit from inside: brighter and whiter at the centre, darker at the edge (limb darkening)
const starMaterial = () => new THREE.ShaderMaterial({
  uniforms: { tint: { value: new THREE.Color(1, 1, 1) } },
  vertexShader: "varying vec3 n; varying vec3 v; void main(){ n = normalize(normalMatrix * normal); vec4 p = modelViewMatrix * vec4(position,1.0); v = normalize(-p.xyz); gl_Position = projectionMatrix * p; }",
  fragmentShader: "uniform vec3 tint; varying vec3 n; varying vec3 v; void main(){ float mu = max(dot(n, v), 0.0); vec3 c = mix(vec3(1.0,0.66,0.40), vec3(1.0,0.96,0.89), pow(mu, 0.7)); gl_FragColor = vec4(c * tint * (0.55 + 0.45 * mu), 1.0); }",
});
function orbitLine(r: number, opacity = 0.22) {
  return new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(Array.from({ length: 160 }, (_, i) => {
    const th = (i / 160) * TAU; return new THREE.Vector3(r * Math.sin(th), 0, r * Math.cos(th));
  })), new THREE.LineBasicMaterial({ color: 0xecebe6, transparent: true, opacity }));
}

type Label = [THREE.Vector3 | "px", string, number?, number?, string?] | null;
// a small 3D stage: a WebGL canvas, an overlay for labels, and the camera
function stage(fov: number, pos: THREE.Vector3) {
  const canvas = el("canvas", { class: "m-stage" });
  const over = el("canvas", { class: "m-stage m-over" });
  const box = el("div", { class: "m-wrap" }, canvas, over);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(fov, 1, 0.1, 400);
  camera.position.copy(pos); camera.lookAt(0, 0, 0);
  let seed = 7;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const dust = new THREE.BufferGeometry();
  dust.setAttribute("position", new THREE.Float32BufferAttribute(Array.from({ length: 900 }, () => (rnd() - 0.5) * 140), 3));
  scene.add(new THREE.Points(dust, new THREE.PointsMaterial({ color: 0x8890b0, size: 0.12, transparent: true, opacity: 0.5 })));
  const render = () => {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2)); renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix(); renderer.render(scene, camera);
  };
  const screen = (p: THREE.Vector3) => { const v = p.clone().project(camera); return [((v.x + 1) / 2) * canvas.clientWidth, ((1 - v.y) / 2) * canvas.clientHeight]; };
  // labels follow points in the scene, white when the current sentence is about them; a dashed line can run from an
  // object down to the graph where its signal is being drawn; "not to scale" always sits top right
  const label = (items: Label[], focus: string[] = [], links: THREE.Vector3[] = []) => {
    const { g, w, h } = fit(over);
    g.fillStyle = css("--muted");
    g.fillText("not to scale", w - g.measureText("not to scale").width, 14);
    for (const p of links) {
      const [x, y] = screen(p);
      g.strokeStyle = "rgba(236,235,230,0.3)"; g.setLineDash([2, 3]); g.lineWidth = 1;
      g.beginPath(); g.moveTo(x, y + 14); g.lineTo(x, h); g.stroke(); g.setLineDash([]);
    }
    for (const it of items) {
      if (!it) continue;
      const [at, text, dx = 10, dy = -8, color] = it;
      g.fillStyle = color ?? (focus.some((f) => text.startsWith(f)) ? css("--ink") : css("--muted"));
      if (at === "px") { g.fillText(text, dx, dy < 0 ? h + dy : dy); continue; }
      const [x, y] = screen(at);
      g.fillText(text, x + dx, y + dy);
    }
  };
  const dispose = () => { renderer.dispose(); renderer.forceContextLoss(); };
  return { box, scene, camera, render, x: (p: THREE.Vector3) => screen(p)[0], label, dispose };
}
const graph = () => el("canvas", { class: "m-graph" });
const later = <T extends HTMLElement>(node: T, from: number): T => { node.classList.add("later"); node.dataset.from = String(from); return node; };

// ---- the dip
function dip(last: string): Show {
  const D = { K: 0.13, A: 3.2, tilt: 0.095, P: 7, U: 0.6 };
  const s = stage(30, new THREE.Vector3(0, 9 * Math.sin(D.tilt), 9 * Math.cos(D.tilt)));
  const halo = sprite("255,200,140", 0.55); halo.scale.setScalar(5.5);
  const planet = new THREE.Mesh(new THREE.SphereGeometry(D.K, 32, 24), new THREE.MeshStandardMaterial({ color: 0xe9ecf3, roughness: 0.9 }));
  s.scene.add(halo, new THREE.Mesh(new THREE.SphereGeometry(1, 64, 48), starMaterial()), planet, orbitLine(D.A),
    new THREE.PointLight(0xffe2c0, 60, 0, 2), new THREE.AmbientLight(0x223355, 0.35));
  const overlap = (d: number, r: number) => {
    if (d >= 1 + r) return 0;
    if (d <= 1 - r) return Math.PI * r * r;
    const a = r * r * Math.acos((d * d + r * r - 1) / (2 * d * r)), b = Math.acos((d * d + 1 - r * r) / (2 * d));
    return a + b - 0.5 * Math.sqrt((-d + r + 1) * (d + r - 1) * (d - r + 1) * (d + r + 1));
  };
  const flux = (th: number) => {
    if (Math.cos(th) <= 0) return 1;
    const x = D.A * Math.sin(th), y = D.A * Math.cos(th) * Math.sin(D.tilt), d = Math.hypot(x, y);
    const mu = Math.sqrt(Math.max(1 - Math.min(d, 1) ** 2, 0));
    return 1 - (overlap(d, D.K) * (1 - D.U * (1 - mu))) / (Math.PI * (1 - D.U / 3));
  };
  const DEPTH = 1 - flux(0), phase = (t: number) => (t / D.P) * TAU - 2.4;
  const g1 = later(graph(), 2);
  const beats: Beat[] = [{ at: 0, text: "A star, and a planet whose orbit we see edge-on.", focus: ["star", "planet"] },
    { at: 1.6, text: "Once every orbit, the planet passes in front of the star.", focus: ["planet"] },
    { at: 8.6, text: "While it does, it blocks a little light, and the star dims.", focus: ["star"] },
    { at: 15, text: last }];
  return {
    node: el("div", { class: "m-show" }, s.box, g1), beats,
    draw(t, k) {
      const th = phase(t);
      planet.position.set(D.A * Math.sin(th), 0, D.A * Math.cos(th));
      s.render();
      const onDisc = Math.cos(th) > 0 && Math.abs(planet.position.x) < 1.3;
      s.label([[new THREE.Vector3(0, -1, 0), "star", -14, 24],
        Math.cos(th) > 0 ? (onDisc ? [new THREE.Vector3(planet.position.x, -1.3, 0), "planet", 6, 0] : [planet.position, "planet", 10, -12]) : null,
        ["px", "seen from Earth, along the edge of the orbit", 0, -4, css("--muted")]], beats[k].focus, k >= 2 && onDisc ? [planet.position] : []);
      const { g, B, T } = axes(g1, "brightness", "time →");
      const lo = 1 - DEPTH * 1.5, Y = (f: number) => T + 8 + ((1 - f) / (1 - lo)) * (B - T - 16);
      const cur = wrapAngle(th), pass = Math.floor((th + Math.PI / 2) / TAU), first = Math.floor((phase(beats[2].at) + Math.PI / 2) / TAU);
      trace(g, (q) => s.x(new THREE.Vector3(D.A * Math.sin(q), 0, D.A * Math.cos(q))), (q) => Y(flux(q)), -Math.PI / 2, Math.PI / 2, cur, pass > first, B);
      g.fillStyle = css("--muted");
      g.fillText(`dims by ${(DEPTH * 100).toFixed(1)}%`, s.x(new THREE.Vector3(0, 0, D.A)) + 50, Y(1 - DEPTH) + 4);
    },
    dispose: s.dispose,
  };
}

// ---- the wobble
function wobble(): Show {
  const W = { A: 3.0, a: 0.42, P: 8, tilt: 0.42 };
  const s = stage(30, new THREE.Vector3(0, 10 * Math.sin(W.tilt), 10 * Math.cos(W.tilt)));
  const star = new THREE.Mesh(new THREE.SphereGeometry(0.8, 64, 48), starMaterial());
  const halo = sprite("255,200,140", 0.55); halo.scale.setScalar(4.5);
  const planet = new THREE.Mesh(new THREE.SphereGeometry(0.22, 32, 24), new THREE.MeshStandardMaterial({ color: 0xff7650, roughness: 0.8 }));
  const light = new THREE.PointLight(0xffe2c0, 50, 0, 2);
  const starPath = orbitLine(W.a, 0.6);
  s.scene.add(halo, star, planet, light, orbitLine(W.A), starPath, new THREE.AmbientLight(0x223355, 0.4));
  const phase = (t: number) => (t / W.P) * TAU - 1.2;
  const rv = (th: number) => -Math.sin(th);  // + is away from us
  const strip = later(el("canvas", { class: "m-strip" }), 2), g0 = later(graph(), 3);
  const beats: Beat[] = [{ at: 0, text: "A star and its planet circle a point between them.", focus: ["star", "planet"] },
    { at: 4, text: "So the planet's pull rocks the star, toward us and away.", focus: ["star", "↓", "↑"] },
    { at: 9, text: "Coming toward us, the star's light turns a little bluer. Going away, a little redder.", focus: ["star"] },
    { at: 15, text: "Plot the star's speed and you get a wave: the planet's signature." },
    { at: 24, text: "That is how 51 Pegasi b was found. Jupiter rocks our Sun at about the speed of a car in town." }];
  return {
    node: el("div", { class: "m-show" }, s.box, strip, g0), beats,
    draw(t, k) {
      const th = phase(t), v = rv(th);
      planet.position.set(W.A * Math.sin(th), 0, W.A * Math.cos(th));
      star.position.set(-W.a * Math.sin(th), 0, -W.a * Math.cos(th));
      halo.position.copy(star.position); light.position.copy(star.position);
      (starPath.material as THREE.LineBasicMaterial).opacity = k >= 1 ? 0.6 : 0;
      const c = k >= 2 ? v : 0;
      (star.material as THREE.ShaderMaterial).uniforms.tint.value.setRGB(1 + 0.25 * Math.max(c, 0) - 0.15 * Math.max(-c, 0),
        1 - 0.06 * Math.abs(c), 1 + 0.35 * Math.max(-c, 0) - 0.3 * Math.max(c, 0));
      s.render();
      s.label([[star.position, "star", 34, -30], [planet.position, "planet", 12, -8],
        k >= 1 ? [new THREE.Vector3(0, 0, W.a), v < -0.2 ? "↓ toward us" : v > 0.2 ? "↑ away from us" : "", -40, 46] : null,
        ["px", "seen from the side, as from Earth", 0, -4, css("--muted")]], beats[k].focus);
      { const { g, w, h } = fit(strip);
        const grad = g.createLinearGradient(0, 0, w, 0);
        ["#3b4fd6", "#3fa7d8", "#57c48a", "#d9d05a", "#e0913f", "#c9483c"].forEach((cc, i) => grad.addColorStop(i / 5, cc));
        g.globalAlpha = 0.9; g.fillStyle = grad; g.fillRect(0, 5, w, 16); g.globalAlpha = 1;
        [0.14, 0.31, 0.47, 0.6, 0.78, 0.9].forEach((f) => {
          g.fillStyle = "rgba(236,235,230,0.6)"; g.fillRect(f * w - 0.5, 0, 1, 5);
          g.fillStyle = css("--page"); g.fillRect(f * w + v * 18 - 1.5, 5, 3, 16);
        });
        g.fillStyle = css("--muted"); g.fillText("← bluer", 0, h - 2); g.fillText("redder →", w - 64, h - 2);
        g.fillText("the star's light, split into colours", w / 2 - 128, h - 2); }
      { const T3 = beats[3].at;
        const { g, w, B, T, L } = axes(g0, "star's speed", "time: one orbit →");
        const mid = (B + T) / 2, amp = (B - T) / 2 - 6;
        g.strokeStyle = css("--line"); g.beginPath(); g.moveTo(L, mid); g.lineTo(w, mid); g.stroke();
        g.fillStyle = css("--muted"); g.fillText("away", L + 6, T + 10); g.fillText("toward", L + 6, B - 4);
        const since = Math.max(t - T3, 0), n = Math.floor(since / W.P), cur = since - n * W.P;
        trace(g, (q) => L + (q / W.P) * (w - L), (q) => mid - amp * rv(phase(T3 + q)), 0, W.P, cur, n > 0, B); }
    },
    dispose: s.dispose,
  };
}

// ---- bent light: one slow pass of a near star (with a planet) across a far one
function bent(): Show {
  const LN = { P: 16, from: -3.4, to: 3.4, y: 0.22, rE: 1.0, planet: new THREE.Vector3(-1.15, -0.22, 0) };
  const s = stage(21, new THREE.Vector3(0, 0, 9));
  const far = sprite("190,215,255", 1, true); far.position.set(0, 0, -5);
  const lens = new THREE.Mesh(new THREE.SphereGeometry(0.34, 32, 24), starMaterial());
  (lens.material as THREE.ShaderMaterial).uniforms.tint.value.setRGB(1, 0.75, 0.55);
  const lensPlanet = new THREE.Mesh(new THREE.SphereGeometry(0.13, 16, 12), new THREE.MeshBasicMaterial({ color: 0x4f7cff }));
  const ringTex = (() => {
    const c = document.createElement("canvas"); c.width = c.height = 256;
    const g = c.getContext("2d")!, gr = g.createRadialGradient(128, 128, 90, 128, 128, 120);
    gr.addColorStop(0, "rgba(190,215,255,0)"); gr.addColorStop(0.5, "rgba(190,215,255,0.95)"); gr.addColorStop(1, "rgba(190,215,255,0)");
    g.fillStyle = gr; g.fillRect(0, 0, 256, 256);
    return new THREE.CanvasTexture(c);
  })();
  const ring = new THREE.Sprite(new THREE.SpriteMaterial({ map: ringTex, depthWrite: false, transparent: true, blending: THREE.AdditiveBlending }));
  ring.scale.setScalar(LN.rE * 2.2);
  s.scene.add(far, ring, lens, lensPlanet);
  const magnify = (x: number) => {
    const u = Math.hypot(x, LN.y) / LN.rE, m = (u * u + 2) / (u * Math.sqrt(u * u + 4));
    const p = Math.hypot(x + LN.planet.x, LN.y + LN.planet.y) / LN.rE;
    return m + 1.2 * Math.exp(-((p / 0.1) ** 2));
  };
  const lensX = (t: number) => LN.from + (LN.to - LN.from) * Math.min(t / LN.P, 1);
  const g2 = later(graph(), 1);
  const beats: Beat[] = [{ at: 0, text: "Far away, a star. Closer to us, another star with a planet, slowly drifting across.", focus: ["far star", "near star", "its planet"] },
    { at: 4.5, text: "As they line up, the near star's gravity bends the far star's light, like a lens.", focus: ["the far star's light", "near star"] },
    { at: 7.4, text: "So the far star looks brighter, for a few weeks.", focus: ["far star", "the far star's light"] },
    { at: 10.6, text: "The near star's planet bends the light too: a short extra blip.", focus: ["its planet"] },
    { at: 15, text: "It happens once and never repeats, so telescopes have to catch it as it happens." }];
  return {
    node: el("div", { class: "m-show" }, s.box, g2), beats,
    draw(t, k) {
      const x = lensX(t);
      lens.position.set(x, LN.y, 0);
      lensPlanet.position.copy(lens.position).add(LN.planet);
      far.scale.setScalar(0.7 + 0.6 * Math.sqrt(magnify(x)));
      ring.position.set(x, LN.y, -0.01);
      ring.material.opacity = Math.max(0, 1 - Math.hypot(x, LN.y) / LN.rE) ** 1.5;
      s.render();
      const lined = ring.material.opacity > 0.12;  // lined up, the far star hides behind the near one: name the ring instead
      s.label([lined ? null : [far.position, "far star", -30, -34], lined ? [lens.position, "the far star's light, bent into a ring", -120, -78] : null,
        [lens.position, "near star", -32, lined ? 92 : 40], [lensPlanet.position, "its planet", -36, 30],
        ["px", "seen from Earth", 0, -4, css("--muted")]], beats[k].focus, k >= 1 && x < LN.to ? [lens.position] : []);
      const { g, B, T, L } = axes(g2, "brightness", "time →");
      const Y = (v: number) => B - 4 - ((v - 1) / 6.2) * (B - T - 12);
      trace(g, (q) => Math.max(s.x(new THREE.Vector3(q, LN.y, 0)), L), (q) => Y(Math.min(magnify(q), 7.1)), LN.from, LN.to, x, false, B);
      g.fillStyle = css("--muted");
      if (k >= 3) g.fillText("the planet's blip", s.x(new THREE.Vector3(-LN.planet.x, LN.y, 0)) + 10, Y(2.2));
      g.fillText("of the far star", L + 6, T + 10);
    },
    dispose: s.dispose,
  };
}

// ---- photographs: a mask slides over the star and the planets appear
function photo(): Show {
  const s = stage(34, new THREE.Vector3(0, 9.2, 4.2));
  const glare = sprite("255,236,210", 1, true); glare.scale.setScalar(10);
  const star = new THREE.Mesh(new THREE.SphereGeometry(0.35, 32, 24), starMaterial());
  // the mask: a dark disc on a thin arm, like the stop inside a coronagraph
  const mask = new THREE.Group();
  const rim = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(Array.from({ length: 96 }, (_, i) =>
    new THREE.Vector3(0.75 * Math.cos((i / 96) * TAU), 0.75 * Math.sin((i / 96) * TAU), 0))),
  new THREE.LineBasicMaterial({ color: 0xecebe6, transparent: true, opacity: 0.7 }));
  const arm = new THREE.Mesh(new THREE.PlaneGeometry(9, 0.07), new THREE.MeshBasicMaterial({ color: 0x343b5c }));
  arm.position.x = -5.2;
  mask.add(arm, new THREE.Mesh(new THREE.CircleGeometry(0.75, 64), new THREE.MeshBasicMaterial({ color: 0x0b0d16 })), rim);
  mask.lookAt(s.camera.position);
  const kids = [1.5, 2.2, 3.0, 3.8].map((r, k) => {
    const p = sprite("255,138,76", 1, true);
    p.scale.setScalar(0.5); s.scene.add(p, orbitLine(r, 0.1));
    return { p, r, speed: 0.9 / r ** 1.5, at: k * 1.7 };
  });
  s.scene.add(star, glare, mask);
  const maskIn = (t: number) => smooth(4.8, 7.4, t), planetsIn = (t: number) => smooth(8, 9.5, t);
  const from = s.camera.position.clone().normalize();
  const g3 = later(graph(), 1);
  const beats: Beat[] = [{ at: 0, text: "A young star with giant planets. Its glare drowns them out.", focus: ["star"] },
    { at: 4.5, text: "A telescope can block the star with a small mask, called a coronagraph.", focus: ["mask"] },
    { at: 8, text: "With the glare gone, the planets show up, each a faint dot of its own light.", focus: ["planet"] },
    { at: 11, text: "Each is tens of thousands of times fainter than its star.", focus: ["planet"] },
    { at: 14, text: "HR 8799's planets were found this way: the first family of planets ever photographed." }];
  return {
    node: el("div", { class: "m-show" }, s.box, g3), beats,
    draw(t, k) {
      const m = maskIn(t);
      mask.position.set(-7 * (1 - m), 0, 0).addScaledVector(from, 0.6);
      glare.material.opacity = 1 - 0.9 * m;
      for (const q of kids) {
        const a = q.at + t * q.speed;
        q.p.position.set(q.r * Math.cos(a), 0, q.r * Math.sin(a));
        q.p.material.opacity = 0.04 + 0.96 * planetsIn(t);
      }
      s.render();
      s.label([m < 0.5 ? [star.position, "star (the glare)", 30, -34] : null, k >= 1 ? [mask.position, "mask", 40, 6] : null,
        planetsIn(t) > 0.5 ? [kids[3].p.position, "planet", 12, -8] : null], beats[k].focus);
      const { g, w, B, T, L } = axes(g3, "starlight", "time →");
      const Y = (v: number) => B - 4 - v * (B - T - 12), end = 14;
      trace(g, (q) => L + (q / end) * (w - L), (q) => Y(1 - 0.9 * maskIn(q)), 0, end, Math.min(t, end), false, B);
      g.fillStyle = css("--muted"); g.fillText("reaching the camera", L + 6, B - 6);
    },
    dispose: s.dispose,
  };
}

export function method(m: Method, dipLast = "That dip, coming back every orbit, is the signal Tessa looks for."): Show {
  return m === "dip" ? dip(dipLast) : m === "wobble" ? wobble() : m === "bent" ? bent() : photo();
}
