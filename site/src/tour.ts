// The tour: ten stops from the first planets to Tessa's own. Every count comes from the data files, so the words
// stay true after the next batch; the rest is from NASA and the discovery papers.
import type { Planets, Summary } from "./data";
import { int } from "./util";

export type Group = "transit" | "rv" | "lens" | "other" | "waiting" | "tessa";

export interface Stop {
  figure?: string;
  title: string;
  text: string;
  at?: string;      // a host to show together with the Sun; without one the view centres on the Sun
  dist?: number;    // how far back the camera sits, in parsecs (by default it fits the Sun and the host)
  year: number;     // the discovery clock: planets found later stay hidden (waiting candidates count as 2027, Tessa's 2028)
  focus?: Group;    // light these up and dim the rest
  links?: { label: string; href: string }[];
  ask?: boolean;    // offer Ask Tessa
  patch?: boolean;  // draw the edge of Tessa's patch
}

const LY = 3.26156;
const minMax = (xs: string[]) => {
  const n = xs.map((x) => Number(x.replace(/,/g, "")));
  return [xs[n.indexOf(Math.min(...n))], xs[n.indexOf(Math.max(...n))]];
};
const words = (k: number) => ["No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"][k] ?? int(k);

export function stops(p: Planets, s: Summary): Stop[] {
  const method = (m: string) => int(p.counts.by_method[m]?.planets ?? 0);
  const facility = (f: string) => int(p.kind.filter((k, i) => k === 0 && p.facilities[p.facility[i]] === f).length);
  const ly = (host: string) => p.dist[p.host.indexOf(host)] * LY;
  const searched = Number(s.survey_searched ?? 0);
  const perFile = s.dossiers ? s.costs_usd.vetting / s.dossiers : 0;
  const ours = p.kind.flatMap((k, i) => (k === 2 ? [i] : [])).sort((a, b) => p.dist[a] - p.dist[b]);
  // "1.4 and 2.2", "1.4, 1.9 and 2.2", or a range once there are many
  const list = (xs: string[]) => (xs.length > 3 ? `${minMax(xs)[0]} to ${minMax(xs)[1]}`
    : xs.length > 1 ? `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}` : xs[0] ?? "");
  const widths = list(ours.map((i) => p.radius[i]).filter((r): r is number => r != null).map((r) => r.toFixed(1)));
  const dists = list(ours.map((i) => int(Math.round(p.dist[i] * LY))));
  const one = ours.length === 1;

  const tour: Stop[] = [
    { figure: "1992", title: "The first planets", at: "PSR B1257+12", year: 1992.9,
      text: "The first planets confirmed outside the solar system circle a pulsar, the spinning core left after a star "
        + "exploded. They showed up as tiny shifts in the timing of its radio pulses." },
    { figure: "1995", title: "The wobble", at: "51 Peg", dist: 150, year: 2026.9, focus: "rv",
      text: "51 Pegasi b was the first found around a star like the Sun. Its pull rocks the star toward us and away, and "
        + `that shows in the star's light. ${method("Radial Velocity")} planets have been found this way.` },
    { figure: method("Transit"), title: "The dip", at: "TRAPPIST-1", dist: 60, year: 2026.9, focus: "transit",
      text: "A planet crossing its star dims it a little, the same way every orbit. Three in four known planets were found "
        + "like this, all seven of TRAPPIST-1's among them." },
    { figure: method("Microlensing"), title: "Bent light", at: "OGLE-2005-BLG-390L", year: 2026.9, focus: "lens",
      text: "A star drifting in front of a far one bends its light like a lens, and a planet adds a brief blip. It works at "
        + `great distances: this one is ${int(Number(ly("OGLE-2005-BLG-390L").toPrecision(3)))} light-years away, toward the centre of the galaxy.` },
    { figure: method("Imaging"), title: "Photographs", at: "HR 8799", dist: 120, year: 2026.9, focus: "other",
      text: "A few young giants glow hot enough, far enough from their star, to be photographed. HR 8799's were the first "
        + "family of planets ever pictured, in 2008." },
    { figure: "0.008%", title: "How much Earth would dim the Sun", year: 2026.9, focus: "transit",
      text: "Dips that small need a telescope above the air, but it can watch many stars at once. Kepler watched 150,000 in "
        + `one patch of sky for four years: that column holds ${facility("Kepler")} of its planets.` },
    { figure: int(p.counts.waiting), title: "TESS candidates still waiting", year: 2027.9, focus: "waiting",
      text: `Since 2018 NASA's TESS has swept nearly the whole sky and found ${facility("Transiting Exoplanet Survey Satellite (TESS)")} `
        + "of the planets here. The faint dust is its candidates, still waiting to be confirmed or ruled out." },
    { figure: int(s.pool), title: "Cool stars close to home", dist: 230, year: 2027.9, patch: true,
      text: "Tessa's patch: small, cool stars within 326 light-years with no known planet. It has searched "
        + `${int(searched)}; a Meta model writes up each dip that survives the checks, for about $${perFile.toFixed(4)} a case file.` },
    { figure: String(ours.length), title: one ? "Tessa's candidate" : "Tessa's candidates", dist: 170, year: 3000, focus: "tessa",
      text: `${one ? "One dip" : `${words(ours.length)} dips`} passed every check: ${one ? "a world" : "worlds"} ${widths} times `
        + `Earth's width, ${dists} light-years away. ${one ? "It stays a candidate" : "They stay candidates"} until telescopes `
        + `confirm ${one ? "it" : "them"}; ${one ? "the" : "each"} case file says when to look.`,
      links: ours.map((i) => ({ label: `Case file: ${p.host[i]}`, href: `/star/${p.host[i].replace(/^TIC /, "")}` })) },
    { title: "Your turn", year: 3000, ask: true,
      text: "Drag to turn, scroll to zoom, and click any point to visit it. Or ask Tessa about any of these worlds." },
  ];
  return ours.length ? tour : tour.filter((t) => t.focus !== "tessa");
}
