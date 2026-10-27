// Tessa's search for one star, as pipeline/search.py runs it on the GPU, here in plain TypeScript so a visitor's
// browser can run it: decode the binned light curve, build the same grid of trial orbits, run the box search,
// pick the peaks, then gate 1 (a dip must clear the floor and beat the best bump in the same light curve,
// the upside-down test). No DOM here, so Node can run it too (scripts/search-parity.mjs).

export const P_MIN = 0.5, P_MAX = 15.0;
const PHASE_BIN_D = 0.25 / 24;
export const DUR_BINS = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24];
const DUR_RANGE = [0.3, 1.5];
const OVERSAMPLE = 3, MIN_IN_POINTS = 4, NBMAX = 2048;
const TOP_PEAKS = 10, PEAK_SEPARATION = 0.002;
const MIN_TRANSITS = 3, MIN_COVERAGE = 0.5, MAX_EVENT_SHARE = 0.5;
export const MARGIN = 1.2;
const RSUN_PER_AU = 215.03;

export interface LC { t: Float64Array; y: Float64Array; w: Float64Array; span: Float64Array; sector: Int16Array; tRef: number }

// float16 bits -> number
function half(h: number): number {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}

// the file pipeline/visitor.py writes
export function decode(buf: ArrayBuffer): LC {
  const v = new DataView(buf);
  const n = v.getUint32(0, true), tRef = v.getFloat64(4, true), ns = v.getUint8(12);
  const secs: { sector: number; iv: number; cad: number }[] = [];
  let o = 13;
  for (let i = 0; i < ns; i++, o += 10) secs.push({ sector: v.getInt16(o, true), iv: v.getFloat32(o + 2, true), cad: v.getFloat32(o + 6, true) });
  const t = new Float64Array(n), y = new Float64Array(n), w = new Float64Array(n), span = new Float64Array(n), sector = new Int16Array(n);
  const oy = o + 4 * n, oi = oy + 2 * n, oc = oi + n;
  for (let i = 0; i < n; i++) {
    const s = secs[v.getUint8(oi + i)], c = v.getUint8(oc + i);
    t[i] = tRef + v.getUint32(o + 4 * i, true) / 86400;
    y[i] = half(v.getUint16(oy + 2 * i, true));
    w[i] = c * s.iv;
    span[i] = c * s.cad;
    sector[i] = s.sector;
  }
  return { t, y, w, span, sector, tRef };
}

const centralDuration = (p: number, rad: number, mass: number) => {
  const a = RSUN_PER_AU * Math.pow(mass * (p / 365.25) ** 2, 1 / 3);
  return (p / Math.PI) * Math.asin(Math.min(1, rad / a));
};

export interface Grid { freq: Float64Array; nbin: Int32Array; klo: Int32Array; khi: Int32Array }
// Trial frequencies, uniform in f^(1/3) so the phase drift over the baseline stays under a third of a transit
export function grid(baseline: number, rad: number, mass: number): Grid {
  const pMax = Math.min(P_MAX, baseline / (MIN_TRANSITS - 1));
  const c = centralDuration(1, rad, mass);
  const du = c / (3 * OVERSAMPLE * baseline);
  const u0 = Math.pow(1 / pMax, 1 / 3), u1 = Math.pow(1 / P_MIN, 1 / 3);
  const n = Math.max(Math.ceil((u1 - u0) / du), 0);
  const freq = new Float64Array(n), nbin = new Int32Array(n), klo = new Int32Array(n), khi = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const u = u0 + i * du, f = u * u * u, p = 1 / f;
    const nb = Math.ceil(p / PHASE_BIN_D);
    const db = centralDuration(p, rad, mass) / (p / nb);
    const hi = Math.min(Math.max(Math.ceil(DUR_RANGE[1] * db), DUR_BINS[0]), Math.min(DUR_BINS[DUR_BINS.length - 1], Math.floor(nb / 4)));
    freq[i] = f; nbin[i] = nb; khi[i] = hi; klo[i] = Math.min(Math.floor(DUR_RANGE[0] * db), hi);
  }
  return { freq, nbin, klo, khi };
}

// The box search over trial frequencies [i0, i1): fold into phase bins, slide boxes of each allowed duration, keep
// the strongest dip (side 0) and bump (side 1) per frequency. t is relative to the first point.
export function blsChunk(t: Float64Array, y: Float64Array, w: Float64Array, g: Grid, i0: number, i1: number) {
  let wtot = 0;
  for (let j = 0; j < w.length; j++) wtot += w[j];
  const wy = new Float64Array(y.length);
  for (let j = 0; j < y.length; j++) wy[j] = w[j] * y[j];
  const sw = new Float64Array(NBMAX), sy = new Float64Array(NBMAX), sc = new Float64Array(NBMAX);
  const cw = new Float64Array(NBMAX + 32), cy = new Float64Array(NBMAX + 32), cc = new Float64Array(NBMAX + 32);
  const stat = new Float32Array(2 * (i1 - i0)), code = new Int32Array(2 * (i1 - i0)).fill(-1);
  for (let m = i0; m < i1; m++) {
    const fr = g.freq[m], nb = g.nbin[m], kmax = g.khi[m], kmin = g.klo[m];
    sw.fill(0, 0, nb); sy.fill(0, 0, nb); sc.fill(0, 0, nb);
    for (let j = 0; j < t.length; j++) {
      const ph = t[j] * fr;
      let b = ((ph - Math.floor(ph)) * nb) | 0;
      if (b > nb - 1) b = nb - 1;
      sw[b] += w[j]; sy[b] += wy[j]; sc[b] += 1;
    }
    cw[0] = cy[0] = cc[0] = 0;
    for (let i = 0; i < nb + kmax; i++) {
      const s = i < nb ? i : i - nb;
      cw[i + 1] = cw[i] + sw[s]; cy[i + 1] = cy[i] + sy[s]; cc[i + 1] = cc[i] + sc[s];
    }
    const at = 2 * (m - i0);
    let best0 = 0, best1 = 0;
    for (let q = 0; q < DUR_BINS.length; q++) {
      const k = DUR_BINS[q];
      if (k > kmax) break;
      if (k < kmin) continue;
      // first index of the maximum on each side, as numpy's argmax picks it
      let m0 = 0, i0b = -1, m1 = 0, i1b = -1;
      for (let i = 0; i < nb; i++) {
        const r = cw[i + k] - cw[i], s = cy[i + k] - cy[i], c = cc[i + k] - cc[i];
        const st = c >= MIN_IN_POINTS && r > 0 && r < wtot ? (s * s * wtot) / (r * (wtot - r)) : 0;
        if (s < 0) { if (st > m0) { m0 = st; i0b = i; } } else if (st > m1) { m1 = st; i1b = i; }
      }
      if (i0b < 0) i0b = 0;
      if (i1b < 0) i1b = 0;
      if (m0 > best0) { best0 = m0; stat[at] = m0; code[at] = q * NBMAX + i0b; }
      if (m1 > best1) { best1 = m1; stat[at + 1] = m1; code[at + 1] = q * NBMAX + i1b; }
    }
  }
  return { stat, code };
}

// numpy rounds halves to even
const rint = (x: number) => { const r = Math.round(x); return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r; };
const median = (a: Float64Array) => {
  const s = Float64Array.from(a).sort(), n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
};

export interface Peak { snr: number; period: number; t0: number; dur_h: number; depth_ppm: number; transits: number; share: number }
// Depth, SNR and per-transit evidence for a box; sign -1 for a dip, +1 for a bump
export function checkPeak(lc: LC, period: number, t0: number, dur: number, sign: number): Peak {
  let W = 0, r = 0, s = 0;
  const ep = new Map<number, [number, number, number]>();  // epoch -> weight, weighted flux, time covered
  for (let j = 0; j < lc.t.length; j++) {
    W += lc.w[j];
    const e = rint((lc.t[j] - t0) / period);
    if (Math.abs(lc.t[j] - t0 - e * period) < dur / 2) {
      r += lc.w[j]; s += lc.w[j] * lc.y[j];
      const a = ep.get(e) ?? [0, 0, 0];
      a[0] += lc.w[j]; a[1] += lc.w[j] * lc.y[j]; a[2] += lc.span[j];
      ep.set(e, a);
    }
  }
  const amp = r > 0 && r < W ? (sign * s * W) / (r * (W - r)) : 0;
  const snr = r > 0 && r < W ? amp * Math.sqrt((r * (W - r)) / W) : 0;
  let total = 0, top = 0, transits = 0;
  for (const [re, se, cover] of ep.values()) {
    const v = Math.max((sign * se) / Math.sqrt(re), 0);
    total += v * v; top = Math.max(top, v);
    if (cover / dur >= MIN_COVERAGE) transits++;
  }
  return { snr, period, t0, dur_h: dur * 24, depth_ppm: amp * 1e6, transits, share: total > 0 ? (top * top) / total : 1 };
}

export function topPeaks(lc: LC, g: Grid, stat: Float32Array, code: Int32Array, side: number): Peak[] {
  const n = g.freq.length, snr = new Float64Array(n);
  for (let m = 0; m < n; m++) snr[m] = Math.sqrt(stat[2 * m + side]);
  const tMed = median(lc.t), peaks: Peak[] = [];
  for (let k = 0; k < TOP_PEAKS; k++) {
    let m = 0;
    for (let i = 1; i < n; i++) if (snr[i] > snr[m]) m = i;
    if (!(snr[m] > 0)) break;
    const c = code[2 * m + side], dk = DUR_BINS[Math.floor(c / NBMAX)], i = c % NBMAX;
    const p = 1 / g.freq[m], nb = g.nbin[m];
    let t0 = lc.tRef + ((i + dk / 2) / nb) * p;
    t0 += p * rint((tMed - t0) / p);
    peaks.push(checkPeak(lc, p, t0, (dk * p) / nb, side === 0 ? -1 : 1));
    for (let q = 0; q < n; q++) if (Math.abs(1 / g.freq[q] - p) < PEAK_SEPARATION * p) snr[q] = 0;
  }
  return peaks;
}

export const passes = (p: Peak) => p.transits >= MIN_TRANSITS && p.share <= MAX_EVENT_SHARE && p.snr > 0;

export interface Verdict { pass: boolean; reason: string; dip: Peak | null; bump: Peak | null; topDip: number; topBump: number }
// Gate 1: the best dip with enough transits must clear the floor and beat the best bump by MARGIN
export function gate(dips: Peak[], bumps: Peak[], floor: number): Verdict {
  const dip = dips.find(passes) ?? null, bump = bumps.find(passes) ?? null;
  const topDip = dips[0]?.snr ?? 0, topBump = bumps[0]?.snr ?? 0;
  const other = bump?.snr || topBump || 0;
  const base = { dip, bump, topDip, topBump };
  if (!dip) return { ...base, pass: false, reason: `no dip with at least ${MIN_TRANSITS} transits` };
  if (dip.snr < floor) return { ...base, pass: false, reason: `dip score ${dip.snr.toFixed(1)} is under the bar of ${floor.toFixed(1)}` };
  if (dip.snr < MARGIN * other) return { ...base, pass: false, reason: `dip score ${dip.snr.toFixed(1)} doesn't beat the upside-down score ${other.toFixed(1)} by ${MARGIN}x` };
  return { ...base, pass: true, reason: `dip score ${dip.snr.toFixed(1)} clears the bar of ${floor.toFixed(1)} and beats the upside-down score ${other.toFixed(1)}` };
}

// What the search needs about the star: its size and mass set how long a transit can last
export function starShape(rad: number | null, mass: number | null) {
  let r = rad ?? NaN, m = mass ?? NaN;
  if (!(r > 0)) r = m > 0 ? m : 0.5;
  if (!(m > 0)) m = r;  // for K and M dwarfs, mass in solar units is close to radius
  return { rad: r, mass: m };
}
