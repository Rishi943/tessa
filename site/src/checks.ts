// The code checks from pipeline/vet.py in plain words. Thresholds mirror vet.py so the page says what
// the pipeline did, not a second opinion.
import type { Checks, StarPage } from "./data";
import { fix, int, svg } from "./util";

export type State = "pass" | "concern" | "fail" | "none";
export interface Line { name: string; state: State; says: string }

const ODD_EVEN_SIGMA = 4, SECONDARY_SNR = 7, SECONDARY_OVER_BUMP = 1.5, MAX_PLANET_REARTH = 25;
const VARIABILITY_RATIO = 0.5, VARIABILITY_CHI2 = 100, MAX_DURATION_RATIO = 3, FLAT_MAX = 0.25, CENTROID_REJECT_SIGMA = 5;

export function stateIcon(state: State): SVGElement {
  const paths: Record<State, string> = {
    pass: `<circle cx="8" cy="8" r="6.5" fill="none" stroke="var(--searched)" stroke-width="1.5"/><path d="M5 8.2l2 2 4-4.4" fill="none" stroke="var(--ink)" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>`,
    concern: `<path d="M8 1.8 14.6 13.6H1.4z" fill="none" stroke="var(--flagged)" stroke-width="1.5" stroke-linejoin="round"/><path d="M8 6v3.6" stroke="var(--ink)" stroke-width="1.7" stroke-linecap="round"/><circle cx="8" cy="11.6" r="0.95" fill="var(--ink)"/>`,
    fail: `<circle cx="8" cy="8" r="6.5" fill="var(--rejected)"/><path d="M5.6 5.6l4.8 4.8M10.4 5.6l-4.8 4.8" stroke="var(--surface)" stroke-width="1.8" stroke-linecap="round"/>`,
    none: `<circle cx="8" cy="8" r="6.5" fill="none" stroke="var(--axis)" stroke-width="1.5"/>`,
  };
  const label = { pass: "passed", concern: "a concern", fail: "failed", none: "not run" }[state];
  return svg(`<svg viewBox="0 0 16 16" role="img" aria-label="${label}">${paths[state]}</svg>`);
}

export function plainChecks(page: StarPage): Line[] {
  const c: Checks = page.checks;
  const out: Line[] = [];
  const oe = c.odd_even;
  if (oe) {
    const s = oe.diff_sigma;
    out.push({ name: "Odd and even dips", state: s >= ODD_EVEN_SIGMA ? "fail" : s >= 3 ? "concern" : "pass",
      says: `Odd dips average ${int(oe.depth_odd_ppm)} ppm and even ones ${int(oe.depth_even_ppm)} ppm, ${fix(s, 1)} sigma apart. `
        + (s >= ODD_EVEN_SIGMA ? "Two stars of different size eclipsing each other in turn look like this." : "A planet makes every dip the same.") });
  }
  const sec = c.secondary;
  if (sec) {
    const fail = sec.snr >= SECONDARY_SNR && sec.snr >= SECONDARY_OVER_BUMP * sec.best_brightening_snr;
    out.push({ name: "A second, fainter dip", state: fail ? "fail" : sec.snr >= 5 ? "concern" : "pass",
      says: fail
        ? `There is a second dip at phase ${fix(sec.phase, 2)} (signal-to-noise ${fix(sec.snr, 1)}): the far side of an eclipsing binary.`
        : `The strongest dip elsewhere in the orbit has signal-to-noise ${fix(sec.snr, 1)}, no more than noise (bright bumps reach ${fix(sec.best_brightening_snr, 1)}).` });
  }
  const size = c.size;
  if (size?.rp_rearth != null) {
    const r = size.rp_rearth;
    out.push({ name: "Size", state: r > MAX_PLANET_REARTH ? "fail" : r > 15 ? "concern" : "pass",
      says: `After allowing for light from other stars in the aperture, the body would be ${fix(r, r < 10 ? 1 : 0)} times the size of Earth`
        + (r > MAX_PLANET_REARTH ? ": too big for a planet." : r > 15 ? ", larger than Jupiter." : ".") });
  }
  const du = c.duration;
  if (du?.ratio != null) {
    out.push({ name: "Length of the dip", state: du.ratio > MAX_DURATION_RATIO ? "fail" : du.ratio > 1.5 ? "concern" : "pass",
      says: `The dip lasts ${fix(du.observed_h, 1)} hours; a planet crossing the middle of this star on this orbit would take ${fix(du.central_h, 1)}.`
        + (du.ratio > 1.5 ? " Too long for this star, unless the star is bigger than the catalogue says." : "") });
  }
  const sh = c.shape;
  if (sh?.best_ingress_fraction != null) {
    const flat = sh.best_ingress_fraction <= FLAT_MAX;
    out.push({ name: "Shape", state: flat ? "pass" : sh.v_sigma > 3 ? "concern" : "pass",
      says: flat ? `Flat-bottomed, as when a small body sits wholly in front of the star.`
        : `Closer to a V than a flat bottom (V favoured by ${fix(sh.v_sigma, 1)} sigma), as in a grazing eclipse.` });
  }
  const pc = c.phase_curve;
  if (pc) {
    const big = pc.ratio_to_depth >= VARIABILITY_RATIO, sig = pc.chi2 >= VARIABILITY_CHI2;
    out.push({ name: "Brightness between dips", state: big && sig ? "fail" : big || sig ? "concern" : "pass",
      says: `Outside the dip the star swings by ${int(pc.amplitude_ppm)} ppm at this period, ${fix(pc.ratio_to_depth, 2)} times the dip. `
        + (big && sig ? "Spots turning with the star, or a stretched binary, do that." : "Small enough not to explain the dip.") });
  }
  const sd = c.sectors;
  if (sd?.per_sector?.length > 1) {
    out.push({ name: "Depth from sector to sector", state: sd.p_constant < 0.01 ? "concern" : "pass",
      says: `Measured separately in ${sd.per_sector.length} sectors, the depth ${sd.p_constant < 0.01 ? "changes more than noise allows" : "holds steady"} `
        + `(chance of this spread from noise: ${fix(sd.p_constant * 100, sd.p_constant < 0.01 ? 2 : 0)}%). A dip from a neighbour changes as the camera's view of it changes.` });
  }
  const nb = c.neighbours;
  if (nb) {
    out.push({ name: "Neighbours that could fake it", state: nb.plausible_within_1px > 0 ? "concern" : "pass",
      says: `${int(nb.gaia_stars_within)} Gaia stars lie within ${nb.radius_arcsec}". ${int(nb.could_cause)} are bright enough to make this dip if they were eclipsing binaries, `
        + `${nb.could_cause_within_1px} of them within one TESS pixel.` });
  }
  const ce = c.centroid;
  if (ce?.verdict) {
    const fail = ce.on_star && ce.offset_sigma >= CENTROID_REJECT_SIGMA;
    const state: State = ce.verdict === "on_target" ? "pass" : fail ? "fail" : ce.verdict === "off_target" ? "concern" : "concern";
    const where = ce.offset_arcsec != null
      ? `Difference images from ${ce.sectors_used} sectors put the dip ${fix(ce.offset_arcsec, 1)}" from this star (${fix(ce.offset_sigma, 1)} sigma). `
      : "";
    const why = ce.verdict === "on_target" ? "It sits on the target, and on none of the neighbours that could fake it."
      : fail ? `It sits on Gaia ${ce.on_star.source_id}, ${ce.on_star.sep_arcsec}" away.` : `${ce.why[0].toUpperCase()}${ce.why.slice(1)}.`;
    out.push({ name: "Where the dip comes from", state, says: where + why });
  } else {
    out.push({ name: "Where the dip comes from", state: "none", says: "Not measured: another check had already explained the dip." });
  }
  const known = (c.known ?? []) as Checks[];
  const hits = known.filter((k) => k.harmonic != null);
  out.push({ name: "Already known", state: hits.length ? "fail" : "pass",
    says: hits.length ? hits.map((k) => `${k.list} ${k.id} at ${k.harmonic}x this period`).join("; ") + "."
      : "No known planet, candidate or eclipsing binary at this period on this star or its neighbours." });
  return out;
}
