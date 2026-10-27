// A star's case file: the light curve, the checks in plain words, what the agent made of them, and for a
// candidate, when and how to catch its next transit.
import { openAsk } from "./ask";
import { foldChart, lightCurve, sectorDepths, snrBars } from "./charts";
import { plainChecks, stateIcon } from "./checks";
import { transitView } from "./transit";
import { getStar } from "./data";
import type { Checks, Dossier, StarPage } from "./data";
import { date, footer } from "./home";
import { badge, el, fix, int, pct, spectral } from "./util";

const TOOL_NAMES: Record<string, string> = {
  centroid_all: "Centroid in every sector", sector_depths_by_source: "Depth by data source", transit_depths: "Depth of each transit",
  stellar_radius: "Star's radius from Gaia", rotation: "Star's rotation", predict: "Next transits", known: "Catalogue lookup",
};

export async function starPage(main: HTMLElement, tic: number) {
  let p: StarPage;
  try {
    p = await getStar(tic);
  } catch {
    main.append(el("div", { class: "star" }, el("div", { class: "inner" },
      el("h1", {}, `No case file for TIC ${tic}`),
      el("p", {}, "Tessa keeps a case file for every star whose light curve showed a repeating dip. This star either had none or hasn't been searched yet. ",
        el("a", { href: "/#dips", "data-link": true }, "See every dip"), "."))));
    return;
  }
  document.title = `TIC ${tic}, ${p.status === "candidate" ? "a planet candidate" : "case file"}: Tessa`;
  const c = p.checks, s = c.star;
  const inner = el("div", { class: "inner" });
  inner.append(
    el("p", { class: "crumbs" }, el("a", { href: "/", "data-link": true }, "Tessa"), " / ",
      el("a", { href: "/#dips", "data-link": true }, "Every dip"), ` / TIC ${tic}`),
    el("div", { class: "star-head" },
      el("div", {}, el("h1", {}, `TIC ${tic}`),
        el("p", { class: "about" }, `A ${spectral(s.teff)} of ${int(s.teff)} K, ${fix(s.distance, 1)} parsecs away, TESS magnitude ${fix(s.tmag, 1)}. `
          + `Right ascension ${fix(s.ra, 4)}°, declination ${fix(s.dec, 4)}°.`)),
      el("div", {}, badge(p.status))),
    ...verdict(p),
    p.status === "candidate" ? orbitPanel(p) : el("span"),
    el("div", { class: "panel" }, el("h2", {}, "The light curve"),
      el("p", { class: "sub" }, `${int(c.refit?.transits ?? p.gate1.transits)} dips of ${int(c.signal.depth_ppm)} ppm every ${fix(c.signal.period, 5)} days, `
        + `found at signal-to-noise ${fix(p.gate1.dip_snr, 1)}. The same search on the flipped light curve found ${fix(p.gate1.flipped_snr, 1)}.`),
      lightCurve(p)),
    el("div", { class: "panel two" }, foldChart(p), sectorDepths(p) ?? el("div")),
    checksPanel(p),
    agentPanel(p),
  );
  if (p.when) inner.append(nextPanel(p));
  inner.append(el("p", {}, el("button", { class: "ask-open", type: "button", onclick: () => openAsk(`Tell me about TIC ${tic}.`) },
    "Ask Tessa about this star")));
  main.append(el("article", { class: "star" }, inner), footer());
}

function orbitPanel(p: StarPage): HTMLElement {
  const view = transitView(p);
  return view ? el("section", { class: "panel" }, el("h2", {}, "If it is a planet, this is the dip"),
    el("p", { class: "sub" }, "The star as TESS sees it, with the planet crossing on the path its dip implies. Drag across the chart to move through the transit."),
    view) : el("span");
}

function verdict(p: StarPage): HTMLElement[] {
  if (p.status === "rejected" && p.case_closed) {
    return [el("p", { class: "verdict" }, p.case_closed),
      el("p", { class: "verdict-by" }, `Case closed. Line by Muse Spark 1.3 from the pipeline's reasons: ${p.reasons.join("; ")}.`)];
  }
  const d = p.dossier;
  if (d?.public_summary) {
    return [el("p", { class: "verdict" }, d.public_summary),
      el("p", { class: "verdict-by" }, `Summary by ${modelName(d.model)}. ${p.status === "flagged" ? `Still open: ${p.reasons.join("; ")}.` : ""}`)];
  }
  return [el("p", { class: "verdict" }, p.reasons.length ? p.reasons.join("; ") : "Waiting for the agent."), el("p", { class: "verdict-by" }, "From the code checks.")];
}

const modelName = (m: string) => (m.startsWith("muse-wonder") ? "Muse Wonder 1.0" : m.startsWith("muse-spark-1.3") ? "Muse Spark 1.3" : m);

function checksPanel(p: StarPage): HTMLElement {
  return el("section", { class: "panel" }, el("h2", {}, "What the checks found"),
    el("p", { class: "sub" }, "Every test the code ran, in the order it ran them. A planet passes all of them; most impostors fail at least one."),
    el("ul", { class: "checks" }, ...plainChecks(p).map((l) =>
      el("li", {}, stateIcon(l.state), el("span", { class: "name" }, l.name), el("span", { class: "says" }, l.says)))));
}

function reasonsList(d: Dossier): HTMLElement {
  return el("ul", { class: "reasons" }, ...d.reasons.map((r) =>
    el("li", {}, el("div", { class: "rc" }, `${r.check}: ${r.value}`), el("div", {}, r.why))));
}

function agentPanel(p: StarPage): HTMLElement {
  const d = p.dossier;
  const box = el("section", { class: "panel" }, el("h2", {}, "What the agent made of it"));
  if (!d) {
    box.append(el("p", { class: "sub" }, "The code checks explained this dip on their own, so no model was asked."));
    return box;
  }
  const tools = d.tools_called ?? [];
  box.append(el("p", { class: "sub" },
    `${modelName(d.model)} read the checks and returned "${d.verdict === "gate2" ? "pass to gate 2" : d.verdict}"`
    + (tools.length ? `, after calling ${tools.length} tool${tools.length > 1 ? "s" : ""} to test its doubts.` : ".")
    + ` Cost: $${d.cost_usd.toFixed(4)}.`));
  if (tools.length) {
    box.append(el("ul", { class: "tools", "aria-label": "Tools the agent called" }, ...tools.map((t) => el("li", {}, TOOL_NAMES[t] ?? t))));
    for (const t of tools) {
      const line = evidenceLine(t, p.evidence?.[t]);
      if (line) box.append(el("p", {}, el("b", {}, `${TOOL_NAMES[t] ?? t}: `), line));
    }
  }
  box.append(el("details", { class: "data-toggle", open: d.reasons.length <= 4 },
    el("summary", {}, `The dossier, check by check (${d.reasons.length})`), reasonsList(d)));
  if (d.open_questions?.length) {
    box.append(el("h3", { style: "font: 600 1rem var(--sans); margin: 1.5rem 0 0.5rem" }, "What it could not settle"),
      el("ul", {}, ...d.open_questions.map((q) => el("li", {}, q))));
  }
  if (p.audit) {
    box.append(el("div", {}, el("h3", { style: "font: 400 1.3rem var(--serif); margin: 2rem 0 0.4rem" }, "Second opinion"),
      el("p", { class: "sub" }, `Muse Wonder 1.0 read the same checks without seeing the first verdict, and returned "${p.audit.verdict === "gate2" ? "pass to gate 2" : p.audit.verdict}". Cost: $${p.audit.cost_usd.toFixed(3)}.`),
      p.audit.public_summary ? el("p", { class: "quote" }, p.audit.public_summary) : null,
      p.audit.open_questions?.length ? el("ul", {}, ...p.audit.open_questions.map((q) => el("li", {}, q))) : null));
  }
  return box;
}

function evidenceLine(tool: string, e: Checks | undefined): string | null {
  if (!e) return null;
  switch (tool) {
    case "centroid_all":
      return e.offset_arcsec != null
        ? `${e.verdict.replace("_", " ")}: the dip sits ${fix(e.offset_arcsec, 1)}" from the star (${fix(e.offset_sigma, 1)} sigma) over ${e.sectors_used} sectors. ${e.why}.`
        : `${e.verdict}: ${e.why}.`;
    case "sector_depths_by_source":
      return `${String(e.verdict).replaceAll("_", " ")}. `
        + (e.by_source ?? []).map((b: Checks) => `${b.source}: ${int(b.depth_ppm)} ± ${int(b.err_ppm)} ppm over ${b.sectors} sector${b.sectors === 1 ? "" : "s"}`).join("; ") + ".";
    case "transit_depths":
      return `${e.transits} transits averaging ${int(e.mean_depth_ppm)} ppm; the chance they share one depth is ${pct(e.p_consistent)}; `
        + `the three deepest carry ${fix(e.share_of_top_3 * 100, 0)}% of the signal.`;
    case "stellar_radius":
      return (e.options ?? []).map((o: Checks) => `from ${o.radius_from}: ${fix(o.rad, 2)} solar radii, planet ${fix(o.rp_rearth, 1)} Earths, duration ratio ${fix(o.duration_ratio, 1)}`).join("; ") + ".";
    case "rotation":
      return e.rotation_d != null
        ? `the star turns every ${fix(e.rotation_d, 2)} days; the dip period is ${fix(e.signal_over_rotation, 2)} times that`
          + (e.signal_at_rotation_multiple ? ", a match, so spots may be the cause." : ", not a match.")
        : "no clear rotation.";
    default:
      return null;
  }
}

function nextPanel(p: StarPage): HTMLElement {
  const w = p.when!, o = p.observing;
  const box = el("section", { class: "panel" }, el("h2", {}, "How to prove Tessa wrong"));
  box.append(el("p", { class: "sub" }, o?.headline ?? `The next transits of TIC ${p.tic}.`));
  const items: HTMLElement[] = [];
  for (const t of w.tess_sectors.filter((t) => t.transits && !t.lost)) {
    items.push(el("li", { "data-at": t.start }, el("span", { class: "when" }, date(t.start)),
      el("span", { class: "what" }, `TESS sector ${t.sector} (camera ${t.camera}) until ${date(t.end)}: ${t.transits} transits, each predicted to within ${fix(t.timing_err_min, 0)} min. Data public from ${date(t.public_from)}.`)));
  }
  for (const g of w.ground) {
    items.push(el("li", { "data-at": g.from_utc }, el("span", { class: "when" }, date(g.from_utc)),
      el("span", { class: "what" }, `${g.site}: transit ${g.from_utc.slice(11)} to ${g.to_utc.slice(11)} UTC, ${g.alt_mid}° up at mid-transit, timing ± ${fix(g.timing_err_min, 0)} min.`)));
  }
  items.sort((a, b) => a.dataset.at!.localeCompare(b.dataset.at!));
  box.append(el("div", { class: "two" },
    el("div", {}, el("ol", { class: "timeline" }, ...items),
      el("p", { class: "by", style: "margin-top: 0.75rem" }, `Ground nights at Las Cumbres Observatory sites with the star 30° up and the sky dark. ${w.ground_nights} in the next 90 days.`)),
    el("div", {}, snrBars(w.ground_snr_per_transit))));
  if (o) {
    box.append(el("div", { style: "margin-top: 1.5rem; max-width: 46rem" },
      el("p", {}, el("b", {}, "Telescope. "), o.telescope),
      el("p", {}, el("b", {}, "Proved wrong if: "), o.refuted_if),
      el("p", {}, el("b", {}, "A planet if: "), o.confirmed_by),
      el("p", { class: "by" }, `Note by ${modelName(o.model)} from the predictions above; every number in it was checked against them.`)));
  }
  if (p.novelty) {
    box.append(el("p", { class: "by" }, p.novelty.new
      ? "Not on the TOI or community TOI lists, the confirmed planet list, the TESS eclipsing binary catalogue, VSX, Gaia's eclipsing binaries or SIMBAD's binaries, as of this run."
      : "Already on a list: " + p.novelty.matches.map((m) => JSON.stringify(m)).join("; ")));
  }
  return box;
}
