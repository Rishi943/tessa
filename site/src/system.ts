// One planetary system drawn to scale, in its own scene measured in AU: a system is millionths of a parsec across,
// too small for the main scene's float32 positions, so the main camera is copied into this scene every frame
// (same direction, offset from the star converted to AU) and zooming from the galaxy into a system is one move.
//
// Each orbit is as measured (NASA Exoplanet Archive): size, shape, tilt to our line of sight, and the time of a
// transit or of closest approach, which puts each planet where it is today. TESS candidates and Tessa's own
// transit, so their orbits are edge-on to Earth. Where something isn't measured it is guessed and the orbit is
// dashed. The node (which way an orbit's tilt points on the sky) is never measured; all orbits share one.
import * as THREE from "three";
import type { Planets } from "./data";

export const AU_PER_PC = 206264.806;
const R_SUN = 0.00465047, R_EARTH = 4.2635e-5;  // in AU
const GUESS_INCL = 60;  // degrees, for an orbit whose tilt isn't measured: the average tilt of a random orbit
export const jdNow = (ms = Date.now()) => ms / 86400000 + 2440587.5;

export const starHex = (teff: number | null) => (teff == null ? "#ffe9c9" : teff < 3900 ? "#ffbf80" : teff < 5300 ? "#ffd9a8" : teff < 6300 ? "#fff1dc" : "#dfe8ff");

interface Orbit {
  row: number; a: number; e: number; inc: number; w: number;
  period: number | null; m0: number; t0: number;  // mean anomaly m0 at time t0 (JD)
  guessed: boolean;  // tilt, size or place not measured
  r: number;         // true radius, AU
}

function kepler(m: number, e: number): number {
  let E = e < 0.8 ? m : Math.PI;
  for (let k = 0; k < 12; k++) E -= (E - e * Math.sin(E) - m) / (1 - e * Math.cos(E));
  return E;
}

function orbitsOf(p: Planets, rows: number[], starMass: number): Orbit[] {
  const out: Orbit[] = [];
  rows.forEach((i, k) => {
    const period = p.period[i];
    let a = p.a[i];
    if (a == null && period != null) a = Math.cbrt(starMass * (period / 365.25) ** 2);  // Kepler's third law
    if (a == null) return;
    const e = Math.min(p.ecc[i] ?? 0, 0.95), w = ((p.omega[i] ?? 90) * Math.PI) / 180;
    const transits = p.kind[i] !== 0 || p.methods[p.method[i]] === "Transit";
    const inc = p.incl[i] ?? (transits ? 90 : GUESS_INCL);
    let m0 = k * 2.39996, t0 = 2451545, placed = false;  // unplaced planets start a golden angle apart
    const tc = p.tc[i], tp = p.tp[i];
    if (tc != null && period != null) {
      // mid-transit: the planet is between its star and us, at true anomaly 90 degrees minus the periastron angle
      const f = Math.PI / 2 - w, E = 2 * Math.atan(Math.sqrt((1 - e) / (1 + e)) * Math.tan(f / 2));
      m0 = E - e * Math.sin(E); t0 = tc; placed = true;
    } else if (tp != null && period != null) {
      m0 = 0; t0 = tp; placed = true;
    }
    out.push({ row: i, a, e, inc: (inc * Math.PI) / 180, w, period, m0, t0,
      guessed: (p.incl[i] == null && !transits) || (p.a[i] == null && p.smass[i] == null) || (!placed && period != null),
      r: (p.radius[i] ?? 2) * R_EARTH });
  });
  return out;
}

export interface SystemView {
  scene: THREE.Scene;
  outer: number;           // the widest orbit, AU
  minDist: number;         // closest the camera may come, AU
  fitDir: THREE.Vector3;   // a good direction to look from: the orbits a little from above, Earth's side in view
  guessed: boolean;
  transits: boolean;
  hz: [number, number] | null;
  marks: { kind: "planet" | "earth" | "hz" | "los"; row?: number; at: THREE.Vector3 }[];  // label anchors, local AU
  update(jd: number, camDist: number, fade: number): void;
  dispose(): void;
}

export function systemView(p: Planets, rows: number[], at: THREE.Vector3, color: (i: number) => string): SystemView {
  const i0 = rows[0];
  const teff = p.teff[i0];
  const srad = p.srad[i0] ?? (teff != null && teff < 4000 ? 0.4 : 1), smass = p.smass[i0] ?? srad;
  const orbits = orbitsOf(p, rows, smass);
  const outer = Math.max(...orbits.map((o) => o.a * (1 + o.e)), srad * R_SUN * 20);

  // axes: z from the star toward us (the Sun), x the galaxy's north projected onto the sky, y completing the set
  const z = at.clone().negate().normalize();
  const x = new THREE.Vector3(0, 1, 0).addScaledVector(z, -z.y);
  if (x.lengthSq() < 1e-6) x.set(1, 0, 0).addScaledVector(z, -z.x);
  x.normalize();
  const y = new THREE.Vector3().crossVectors(z, x);
  const pos = (o: Orbit, f: number, out = new THREE.Vector3()) => {
    const r = (o.a * (1 - o.e * o.e)) / (1 + o.e * Math.cos(f)), u = o.w + f;
    return out.copy(x).multiplyScalar(r * Math.cos(u))
      .addScaledVector(y, r * Math.sin(u) * Math.cos(o.inc)).addScaledVector(z, r * Math.sin(u) * Math.sin(o.inc));
  };
  // the plane of the innermost orbit carries Earth's orbit and the zone where water could be liquid
  const ref = orbits[0];
  const normal = ref ? new THREE.Vector3().addScaledVector(z, Math.cos(ref.inc)).addScaledVector(y, -Math.sin(ref.inc)) : y.clone();
  const fitDir = normal.clone().multiplyScalar(0.55).addScaledVector(z, 0.35).addScaledVector(x, 0.75).normalize();
  const tilt = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal);

  const scene = new THREE.Scene();
  const disposables: { dispose(): void }[] = [];
  const keep = <T extends { dispose(): void }>(o: T) => (disposables.push(o), o);
  const faders: [THREE.Material & { opacity: number }, number][] = [];
  const fade = <T extends THREE.Material & { opacity: number }>(m: T, base: number) => {
    m.transparent = true; faders.push([m, base]); return keep(m);
  };

  // the star: true size, plus a glow so it shows from far out
  const starCol = new THREE.Color(starHex(teff));
  const star = new THREE.Mesh(keep(new THREE.SphereGeometry(srad * R_SUN, 32, 16)), fade(new THREE.MeshBasicMaterial({ color: starCol }), 1));
  const glowTex = (() => {
    const c = document.createElement("canvas");
    c.width = c.height = 64;
    const g = c.getContext("2d")!, grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    grad.addColorStop(0, "rgba(255,255,255,1)"); grad.addColorStop(0.25, "rgba(255,255,255,0.35)"); grad.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = grad; g.fillRect(0, 0, 64, 64);
    return keep(new THREE.CanvasTexture(c));
  })();
  const glow = new THREE.Sprite(fade(new THREE.SpriteMaterial({ map: glowTex, color: starCol, depthWrite: false, blending: THREE.AdditiveBlending }), 0.9));
  scene.add(star, glow);

  const ellipse = (o: Orbit) => Array.from({ length: 361 }, (_, k) => pos(o, (k / 360) * Math.PI * 2));
  const line = (pts: THREE.Vector3[], col: string, base: number, dashed: boolean, dash = 1) => {
    const g = keep(new THREE.BufferGeometry().setFromPoints(pts));
    const m = dashed ? new THREE.LineDashedMaterial({ color: col, dashSize: dash, gapSize: dash * 0.8 }) : new THREE.LineBasicMaterial({ color: col });
    const l = new THREE.Line(g, fade(m, base));
    if (dashed) l.computeLineDistances();
    scene.add(l);
    return l;
  };
  const planets = orbits.map((o) => {
    line(ellipse(o), color(o.row), 0.5, o.guessed, o.a * 0.05);
    const mesh = new THREE.Mesh(keep(new THREE.SphereGeometry(1, 20, 12)), fade(new THREE.MeshBasicMaterial({ color: color(o.row) }), 1));
    scene.add(mesh);
    return { o, mesh };
  });

  // Earth's orbit for scale, and the zone where a planet could keep liquid water (Kopparapu-style bounds from the
  // star's light: L = R^2 (T / 5772)^4, 0.95 to 1.67 times the square root of L)
  const earth = line(Array.from({ length: 361 }, (_, k) => new THREE.Vector3(Math.cos((k / 360) * Math.PI * 2), Math.sin((k / 360) * Math.PI * 2), 0)),
    "#7f86a6", 0.55, true, 0.05);
  earth.quaternion.copy(tilt);
  let hz: [number, number] | null = null;
  const lum = teff != null && p.srad[i0] != null ? Math.sqrt(srad * srad * (teff / 5772) ** 4) : null;
  if (lum != null && 0.95 * lum < outer * 3) {  // only when it is near the orbits: far out it reads as a stray arc
    hz = [0.95 * lum, 1.67 * lum];
    const band = new THREE.Mesh(keep(new THREE.RingGeometry(hz[0], hz[1], 128)),
      fade(new THREE.MeshBasicMaterial({ color: "#8CD49B", side: THREE.DoubleSide, depthWrite: false }), 0.1));
    band.quaternion.copy(tilt);
    scene.add(band);
  }
  // the line from the star to us: transiting orbits lie along it, which is why they dim their star
  const toUs = z.clone().multiplyScalar(outer * 1.7);
  line([new THREE.Vector3(), toUs], "#ECEBE6", 0.35, true, outer * 0.04);

  const marks: SystemView["marks"] = [
    ...planets.map(({ o, mesh }) => ({ kind: "planet" as const, row: o.row, at: mesh.position })),
    { kind: "earth", at: x.clone().applyAxisAngle(normal, -0.6) },
    ...(hz ? [{ kind: "hz" as const, at: x.clone().applyAxisAngle(normal, 2.4).multiplyScalar(hz[1]) }] : []),
    { kind: "los", at: toUs },
  ];
  const tmp = new THREE.Vector3();

  return {
    scene, outer, minDist: srad * R_SUN * 3, fitDir, hz, marks,
    guessed: orbits.some((o) => o.guessed), transits: orbits.some((o) => o.inc > 1.5 && o.inc < 1.64),
    update(jd, camDist, f) {
      for (const [m, base] of faders) m.opacity = base * f;
      glow.scale.setScalar(Math.max(srad * R_SUN * 10, camDist * 0.05));
      for (const { o, mesh } of planets) {
        const m = o.period ? o.m0 + ((2 * Math.PI) / o.period) * (jd - o.t0) : o.m0;
        const E = kepler(((m % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI), o.e);
        const fTrue = 2 * Math.atan2(Math.sqrt(1 + o.e) * Math.sin(E / 2), Math.sqrt(1 - o.e) * Math.cos(E / 2));
        pos(o, fTrue, tmp);
        mesh.position.copy(tmp);
        mesh.scale.setScalar(Math.max(o.r, camDist * 0.006));  // drawn larger than life, or they'd be invisible
      }
    },
    dispose() { for (const d of disposables) d.dispose(); },
  };
}
