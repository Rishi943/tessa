import { foldChart } from "./charts";
import { getDips, getSky, getStar, getSummary } from "./data";
import type { Dip, StarPage, Status, Summary } from "./data";
import { mountSky } from "./sky";
import type { SkyView } from "./sky";
import { STATUS, badge, el, fix, int, mark, navigate, spectral, usd } from "./util";

const LEGEND: Status[] = ["candidate", "flagged", "rejected", "searched", "pool"];

let sky: SkyView | null = null;
let visit = 0;  // the sky loads slowly; a late load must not mount onto a page already left
let pendingFly: number | null = null;

// Ask Tessa's "show it on the map" for a star only the survey has: go to the survey, then turn its map
export function flyTo(tic: number) {
  window.scrollTo({ top: 0, behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  if (sky && location.pathname === "/survey") sky.flyTo(tic);
  else {
    pendingFly = tic;
    if (location.pathname !== "/survey") navigate("/survey");
  }
}
export function leaveHome() {
  visit++;
  sky?.dispose();
  sky = null;
}

export async function home(main: HTMLElement) {
  const mine = visit;
  const [summary, dips] = await Promise.all([getSummary(), getDips()]);
  const searched = Number(summary.searched_total ?? Number(summary.survey_searched ?? 0) + Number(summary.batch2_searched ?? 0));
  const n = summary.by_status;

  // Hero: the sky, the claim, and the legend that is also the funnel
  const legend = el("ul", { class: "legend", "aria-label": "What Tessa did with each star; choose one to show only those" });
  const hero = el("section", { class: "hero", "aria-label": "Every star in the survey" },
    el("div", { class: "hero-text" },
      el("h1", {}, `Tessa searched ${int(searched)} nearby red stars for planets. ${n.candidate === 1 ? "One signal" : `${words(n.candidate)} signals`} survived every test.`),
      el("p", { class: "lede" },
        "Tessa is an AI agent that reads NASA's TESS light curves of K and M dwarfs within 100 parsecs, explains away every dip it can, "
        + "and turns what is left into predictions anyone with a telescope can check."),
      legend),
    el("p", { class: "sky-hint" }, `${int(summary.pool)} stars at their true distances from the Sun. Drag to turn; click a marked star.`));
  main.append(hero);

  const counts: Record<string, number> = { ...n, pool: n.pool + n["held back"], searched: n.searched };
  const buttons = LEGEND.map((s) => {
    const b = el("button", { type: "button", "aria-pressed": "false" }, mark(s), el("span", { class: "count" }, int(counts[s] ?? 0)),
      el("span", { class: "what" }, STATUS[s].short));
    legend.append(el("li", {}, b));
    return b;
  });
  getSky().then((data) => {
    if (visit !== mine) return;
    sky = mountSky(hero, data, (focus) => buttons.forEach((b, i) => b.setAttribute("aria-pressed", String(LEGEND[i] === focus))));
    if (sky && pendingFly != null) {
      const tic = pendingFly;
      pendingFly = null;
      setTimeout(() => sky?.flyTo(tic), 300);
    }
    buttons.forEach((b, i) => b.addEventListener("click", () => {
      const s = LEGEND[i];
      sky?.focus(b.getAttribute("aria-pressed") === "true" ? null : s);
    }));
  });

  main.append(await candidates(dips), method(summary, dips, searched), accuracy(), dipList(dips), footer());
}

const words = (k: number) => ["No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"][k] ?? int(k);

async function candidates(dips: Dip[]): Promise<HTMLElement> {
  const list = dips.filter((d) => d.status === "candidate").sort((a, b) => b.snr - a.snr);
  const inner = el("div", { class: "inner" },
    el("h2", {}, list.length ? "What is left, and when to look again" : "No candidates yet"),
    el("p", { class: "intro" },
      "A candidate is a dip no check could explain, that no catalogue already knows, and that predicts its own next transit. "
      + "It is not a planet until someone sees that transit on time and weighs the star's wobble."));
  const pages = await Promise.all(list.map((d) => getStar(d.tic)));
  for (const p of pages) inner.append(candidate(p));
  return el("section", { class: "section", id: "candidates" }, inner);
}

function candidate(p: StarPage): HTMLElement {
  const c = p.checks, w = p.when;
  const star = c.star;
  const href = `/star/${p.tic}`;
  const facts = el("dl", { class: "facts" },
    fact("Orbit", `${fix(c.signal.period, 3)}`, "days"),
    fact("Depth", int(c.signal.depth_ppm), "ppm"),
    fact("Size", fix(c.size.rp_rearth, 1), "Earths"),
    fact("Transits seen", int(c.refit?.transits ?? p.gate1.transits), ""),
    w ? fact("Timing now", `± ${fix(w.timing_err_now_min, 0)}`, "min") : null);
  const next: (string | Node)[] = [];
  if (w?.next_tess) {
    const t = w.next_tess;
    next.push(el("b", {}, `TESS sector ${t.sector}`), `, ${date(t.start)} to ${date(t.end)}, should catch ${t.transits} transits; the data are public from ${date(t.public_from)}. `);
  }
  if (w?.ground?.length) {
    const g = w.ground[0];
    next.push(`From the ground: ${w.ground_nights} night${w.ground_nights === 1 ? "" : "s"} in the next 90 days, the first at `,
      el("b", {}, g.site), `, ${date(g.from_utc)}, ${g.from_utc.slice(11)} to ${g.to_utc.slice(11)} UTC.`);
  }
  return el("article", { class: "cand" },
    el("div", {}, foldChart(p)),
    el("div", {},
      el("h3", {}, el("a", { href, "data-link": true }, `TIC ${p.tic}`)),
      el("p", { class: "what" }, `A body ${fix(c.size.rp_rearth, 1)} times Earth's size passing in front of a ${spectral(star.teff)} `
        + `${fix(star.distance, 0)} parsecs away, every ${fix(c.signal.period, 3)} days.`),
      p.dossier?.public_summary ? el("blockquote", { class: "said" }, p.dossier.public_summary,
        el("small", {}, `${p.dossier.model?.startsWith("glimmer") ? "Glimmer, Tessa's fine-tuned Muse Glimmer 30B" : "Muse Spark 1.3"}, from the checks`
          + (p.fpp ? `; TRICERATOPS: ${p.fpp.tier}` : ""))) : null,
      facts,
      el("p", { class: "next" }, ...next),
      el("p", {}, el("a", { href, "data-link": true }, "Open the full case file"))));
}

function fact(dt: string, dd: string, unit: string) {
  return el("div", {}, el("dt", {}, dt), el("dd", {}, dd, unit ? el("small", {}, ` ${unit}`) : null));
}

export function date(iso: string): string {
  const d = new Date(`${iso.slice(0, 10)}T00:00:00Z`);
  return d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
}

function method(summary: Summary, dips: Dip[], searched: number): HTMLElement {
  const cost = summary.costs_usd;
  const codeOut = dips.filter((d) => !d.dossier && d.status === "rejected").length;
  const agentOut = dips.filter((d) => d.dossier && d.status === "rejected").length;
  const total = Object.values(cost).reduce((a, b) => a + b, 0);
  const funnel = summary.funnel;
  const step = (title: string, body: string, by: string, big: string, small: string, model = false) =>
    el("li", { class: model ? "model" : "" }, el("div", {}, el("h3", {}, title), el("p", {}, body), el("p", { class: "by" }, by)),
      el("div", { class: "tally" }, el("b", {}, big), el("span", {}, small)));
  const steps = el("ol", { class: "steps" },
    step("Choose the stars",
      "Gaia stars within 100 parsecs that are red enough to be K or M dwarfs, matched to the TESS Input Catalog, with known planet hosts, "
      + `binaries and crowded fields taken out. ${int(funnel[funnel.length - 1].stars)} of them were never on TESS's two-minute target lists, `
      + "so their only light curves come from the full-frame images, which get searched far less.",
      "Code, from the Gaia and TESS catalogues", int(summary.pool), `stars, from ${int(funnel[0].stars)}`),
    step("Search every light curve",
      "Light curves from NASA's archive are flattened and searched for dips that repeat. The bar is set by searching the same light curves "
      + "flipped upside down, where no real dip can exist, so about one noise bump in a hundred stars gets through.",
      "Code, on two free Kaggle GPUs", int(dips.length), `repeating dips in ${int(searched)} stars`),
    step("Rule out the usual impostors",
      "Odd against even dips, a second eclipse, size, length, shape, flicker between dips, and nearby stars that could fake it. For dips that "
      + "survive, TESS's own images show where on the sky the light went missing.",
      "Code", int(codeOut), "explained away"),
    step("Glimmer makes the call",
      "Glimmer is Meta's open-weight Muse Glimmer 30B, fine-tuned on investigations by Muse Spark Open of past TESS signals whose fate "
      + "telescopes settled, and run on Tessa's own GPU. It reads every number from the checks and a planet rank from a separate scorer, "
      + "names what could explain the dip, chooses which of five measurements to look at, and rejects, flags or passes it with a dossier "
      + "anyone can read.",
      "Meta Muse Glimmer 30B, fine-tuned and self-hosted", int(summary.dossiers), `dossiers, ${agentOut} more explained`, true),
    step("Code decides",
      "A pass becomes a candidate only if no catalogue of planets, candidates or variable stars already has it, each observing season "
      + "predicts the others, it predicts its next transit closely enough to test from TESS or the ground, and TRICERATOPS does not "
      + "favour a false positive. Code can stop a pass; it cannot make one.",
      "Code and TRICERATOPS", int(summary.candidate), `candidates, ${int(summary.flagged)} still unexplained`),
    step("Notes for people",
      "Muse Spark 1.3 writes one plain line for every rejected dip and an observing note for every candidate. Ask Tessa, below, answers "
      + "questions from these same files.",
      `Meta Muse Spark 1.3, ${usd(cost.case_closed + cost.observing)}`,
      int(summary.candidate + summary.rejected), "notes", true));
  return el("section", { class: "section", id: "method" }, el("div", { class: "inner" },
    el("h2", {}, "Glimmer makes the call; code keeps the bar"),
    el("p", { class: "intro" },
      "Each step below runs on every star that reached it. A fine-tuned Meta model makes the planet call and says why; code can stop a "
      + "pass but never make one. Everything a model writes is checked for numbers it didn't get from the data"
      + (summary.invented ? `: so far ${summary.invented.with_invented_numbers} of ${summary.invented.dossiers} dossiers had one, and it is marked.` : ".")),
    steps,
    el("p", { class: "costline" }, `Meta Model API cost for the whole search so far: ${usd(total)}, or ${usd((total / searched) * 1000)} per thousand stars `
      + "searched. Glimmer runs on Tessa's own rented GPU, paid by the hour, not per call.")));
}

// How accurate is Tessa: the held-out test (README, "On known TESS candidates"; printed by python -m finetune.exo_bench).
// Fixed numbers: the test split was scored once, after every choice was made.
const TEST = { planets: 297, fps: 282, passed_planets: 213, passed_fps: 18 };
const VS = { planets: 70, fps: 94, tessa: { planets: 50, fps: 3 }, exominer: { planets: 55, fps: 7 } };

function accuracy(): HTMLElement {
  const calls = TEST.passed_planets + TEST.passed_fps;
  const pct = Math.round((100 * TEST.passed_planets) / calls);
  const waffle = el("div", { class: "acc-waffle", role: "img",
    "aria-label": `${calls} planet calls: ${TEST.passed_planets} were planets, ${TEST.passed_fps} were false alarms` });
  for (let i = 0; i < calls; i++) {
    const fp = i >= TEST.passed_planets;
    waffle.append(el("i", { class: fp ? "fp" : "", style: `--i:${i}`,
      title: fp ? "A false alarm: Tessa said planet, follow-up said no" : "A planet: Tessa said planet, follow-up agreed" }));
  }
  const key = (cls: string, label: string) => el("span", { class: "acc-key" }, el("i", { class: cls }), label);
  const row = (who: string, r: { planets: number; fps: number }, ours = false) =>
    el("div", { class: `acc-row${ours ? " ours" : ""}` },
      el("span", { class: "acc-who" }, who),
      el("span", { class: "acc-dots", title: `${r.fps} false alarms out of ${VS.fps}` },
        ...Array.from({ length: r.fps }, () => el("i", { class: "fp" })), el("b", {}, `${r.fps}`)),
      el("span", { class: "acc-pl" }, `${r.planets} of ${VS.planets}`));
  const box = el("div", { class: "acc" },
    el("div", { class: "acc-main" },
      el("p", { class: "acc-hero" }, el("b", {}, `${pct}%`), el("span", {}, "When Tessa says planet, she's right this often")),
      waffle,
      el("p", { class: "acc-keys" }, key("", `a planet (${TEST.passed_planets})`), key("fp", `a false alarm (${TEST.passed_fps})`)),
      el("p", { class: "acc-note" }, `Each square is one of her ${calls} planet calls. She let through ${Math.round((100 * TEST.passed_planets) / TEST.planets)}% of the real planets `
        + `(${TEST.passed_planets} of ${TEST.planets}) and ${(100 * TEST.passed_fps / TEST.fps).toFixed(1)}% of the false positives (${TEST.passed_fps} of ${TEST.fps}). `
        + "A planet she misses is a lost candidate, not a false claim.")),
    el("div", { class: "acc-vs" },
      el("h3", {}, "Against NASA's ExoMiner++"),
      el("p", { class: "acc-sub" }, `On the ${VS.planets + VS.fps} test signals ExoMiner++ 2.0 also scored: ${VS.planets} planets, ${VS.fps} false positives.`),
      el("div", { class: "acc-head" }, el("span", {}, ""), el("span", {}, "False alarms"), el("span", {}, "Planets passed")),
      row("Tessa", VS.tessa, true), row("ExoMiner++", VS.exominer),
      el("p", { class: "acc-note" }, "Tessa raised fewer than half the false alarms, and that holds up: the 95% bootstrap interval for the difference is "
        + "−8.5 to −1.1 points. ExoMiner++ passed 5 more planets, a gap within the noise (−20 to +5.7 points). It also reads TESS's "
        + "2-minute data; Tessa uses only the full-frame images, so she can vet stars that were never 2-minute targets.")));
  const sec = el("section", { class: "section", id: "accuracy" }, el("div", { class: "inner" },
    el("h2", {}, "How accurate is Tessa?"),
    el("p", { class: "intro" }, `Tested on ${TEST.planets + TEST.fps} past TESS signals whose fate telescopes have since settled: ${TEST.planets} planets and `
      + `${TEST.fps} false positives. None were used to train or tune her, and they were scored once, at the end.`),
    box));
  // the squares fill in when the section comes into view
  const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { sec.classList.add("in"); io.disconnect(); } }, { threshold: 0.35 });
  io.observe(sec);
  return sec;
}

function dipList(dips: Dip[]): HTMLElement {
  const order: Status[] = ["candidate", "flagged", "rejected", "awaiting agent", "known"];
  const sorted = [...dips].sort((a, b) => order.indexOf(a.status) - order.indexOf(b.status) || b.snr - a.snr);
  let filter: Status | null = null, query = "", shown = 25;
  const tbody = el("tbody");
  const more = el("button", { class: "more", type: "button" });
  const caption = el("p", { class: "intro" });
  function render() {
    const rows = sorted.filter((d) => (!filter || d.status === filter)
      && (!query || `${d.tic} ${d.line}`.toLowerCase().includes(query.toLowerCase())));
    tbody.replaceChildren(...rows.slice(0, shown).map((d) => el("tr", {},
      el("td", {}, badge(d.status)),
      el("td", {}, el("a", { href: `/star/${d.tic}`, "data-link": true }, `TIC ${d.tic}`)),
      el("td", { class: "r" }, fix(d.period_d, 3)),
      el("td", { class: "r" }, int(d.depth_ppm)),
      el("td", { class: "r" }, fix(d.rp_rearth, 1)),
      el("td", { class: "line" }, d.line))));
    more.hidden = rows.length <= shown;
    more.textContent = `Show all ${rows.length}`;
  }
  more.addEventListener("click", () => { shown = Infinity; render(); });
  const chips = order.filter((s) => dips.some((d) => d.status === s)).map((s) => {
    const b = el("button", { class: "chip", type: "button", "aria-pressed": "false" }, mark(s),
      `${STATUS[s].label} ${dips.filter((d) => d.status === s).length}`);
    b.addEventListener("click", () => {
      filter = filter === s ? null : s;
      chips.forEach((c) => c.setAttribute("aria-pressed", String(c === b && filter === s)));
      render();
    });
    return b;
  });
  const search = el("input", { type: "search", placeholder: "Find a TIC number or a word", "aria-label": "Find a dip" }) as HTMLInputElement;
  search.addEventListener("input", () => { query = search.value.trim(); render(); });
  caption.textContent = `All ${dips.length} repeating dips the search found, each with what Tessa made of it. The reason for every rejection is written down, `
    + "so anyone can check it.";
  render();
  return el("section", { class: "section", id: "dips" }, el("div", { class: "inner" },
    el("h2", {}, "Every dip, and why"),
    caption,
    el("div", { class: "filters" }, ...chips, search),
    el("div", { class: "table-wrap" }, el("table", { class: "dips" },
      el("thead", {}, el("tr", {}, el("th", {}, "Outcome"), el("th", {}, "Star"), el("th", { class: "r" }, "Orbit, days"),
        el("th", { class: "r" }, "Depth, ppm"), el("th", { class: "r" }, "Size, Earths"), el("th", {}, "What Tessa found"))),
      tbody)),
    more));
}

export function footer(): HTMLElement {
  return el("footer", { class: "foot" }, el("div", { class: "inner" },
    el("p", {}, "Data: NASA TESS light curves from MAST (SPOC, TESS-SPOC and QLP), ESA Gaia DR3, the TESS Input Catalog, ExoFOP TOI and CTOI lists, "
      + "the NASA Exoplanet Archive, the TESS eclipsing binary catalogue, VSX and SIMBAD."),
    el("p", {}, el("b", {}, "Built with Muse Spark Open."), " Models: Meta Muse Glimmer 30B, fine-tuned on traces from Muse Spark Open; Muse Spark 1.3; Muse Wonder 1.0 for a "
      + "second opinion on every candidate. False-positive probabilities: TRICERATOPS (Giacalone et al. 2021). Built for the Meta AI Developer Hackathon, "
      + "October to November 2026. ",
      el("a", { href: "https://github.com/Rishi943/tessa" }, "The code is on GitHub"), ".")));
}
