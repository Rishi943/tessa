// "Run a star": the whole pipeline on one star, on this machine, as it happens. Only on a local copy of the site
// (npm run local), where tools/local_run.py streams tools/demo.py's steps; the hosted site has no such server and
// says so. Glimmer is called live when the repo's .env has its endpoint, and its recorded answer is replayed,
// labelled, when it doesn't.
import { openAsk } from "./ask";
import { foldChart, lightCurve } from "./charts";
import type { StarPage } from "./data";
import { el, fix } from "./util";

interface Health { glimmer: "live" | "replay"; meta: boolean; stars: { tic: number; note: string }[] }
type Ev = Record<string, any> & { type: string };

const TOOL: Record<string, string> = {
  centroid_all: "Where the dip is, in every sector's pixels", sector_depths_by_source: "Depth in each kind of data",
  transit_depths: "Depth of each transit", stellar_radius: "The star's size from Gaia", rotation: "The star's spin",
};

// lines from libraries, not from Tessa: astroquery's query notices, the Gaia archive's release banner, Python
// warnings, download progress; the page shows Tessa's own steps and leaves these in the terminal
const NOISE = /^(INFO:|WARNING:|Downloading |Getting TessCut|Warning:|[a-z_]+\.py:\d+)|warnings\.warn|pkg_resources|archive release|questions or feedback|helpdesk|tokens in, .* out, |^tools it chose|^\s*verdict:/i;

// the local runner, called directly: a run can outlast the dev server's proxy, which closes requests after 5 minutes
const LOCAL = /^(localhost|127\.0\.0\.1)$/.test(location.hostname) ? "http://127.0.0.1:8788" : "";

// what Ask Tessa is told about the run (see ask.ts): the steps so far, kept short
let state: Record<string, unknown> | null = null;
export const runState = () => state;

export async function runPage(main: HTMLElement, want: number | null) {
  document.title = "Run a star: Tessa";
  const inner = el("div", { class: "inner" });
  main.append(el("div", { class: "star run" }, inner));
  inner.append(el("p", { class: "crumbs" }, el("a", { href: "/", "data-link": true }, "Tessa"), " / Run a star"));
  let h: Health;
  try {
    const r = await fetch(`${LOCAL}/local/health`);
    if (!r.ok) throw new Error();
    h = await r.json();
  } catch {
    inner.append(el("h1", {}, "Run a star"),
      el("p", { class: "sub" }, "This page runs Tessa's whole pipeline on one star, on your own machine. It needs a local copy: clone the repo, put your keys in .env, then run ",
        el("code", {}, "cd site && npm run local"), ". The hosted site only runs the first step, the search, in your browser."));
    return;
  }
  const tic = want ?? h.stars[0].tic;
  const glimmerLive = h.glimmer === "live";
  inner.append(
    el("div", { class: "star-head" }, el("div", {}, el("h1", {}, `TIC ${tic}`),
      el("p", { class: "about" }, h.stars.find((s) => s.tic === tic)?.note ?? ""))),
    el("p", { class: "run-modes" },
      el("span", { class: `run-mode ${glimmerLive ? "on" : "rec"}` }, glimmerLive ? "Glimmer: live call" : "Glimmer: REPLAY of its recorded answer"),
      el("span", { class: `run-mode ${h.meta ? "on" : ""}` }, h.meta ? "Ask Tessa: live (Muse Spark)" : "Ask Tessa: no key"),
      el("span", { class: "run-mode on" }, "Everything else: live, on this machine")),
    el("p", { class: "run-pick" }, "Bundled stars: ", ...h.stars.map((s) =>
      el("a", { href: `/run/${s.tic}`, "data-link": true, class: s.tic === tic ? "here" : "" }, `TIC ${s.tic}`))),
  );
  const go = el("button", { type: "button", class: "u-btn run-go" }, `Run TIC ${tic}`) as HTMLButtonElement;
  const log = el("ol", { class: "run-log" });
  inner.append(go, log);
  go.addEventListener("click", () => start(tic, go, log));
}

function start(tic: number, go: HTMLButtonElement, log: HTMLElement) {
  go.disabled = true;
  go.textContent = "Running";
  log.replaceChildren();
  state = { local_run: `TIC ${tic}, running on the reader's machine`, steps: [] as string[] };
  const steps = state.steps as string[];
  let body: HTMLElement = log, glimmer: HTMLElement | null = null, afterDossier = false;
  const t0 = performance.now();
  const clock = el("span", { class: "run-clock" });
  const tick = setInterval(() => { const s = Math.round((performance.now() - t0) / 1000); clock.textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; }, 500);
  go.after(clock);
  const es = new EventSource(`${LOCAL}/local/run?tic=${tic}`);
  const done = () => { es.close(); clearInterval(tick); go.disabled = false; go.textContent = `Run TIC ${tic} again`; };

  es.onmessage = (m) => {
    const e = JSON.parse(m.data) as Ev;
    switch (e.type) {
      case "step": {
        afterDossier = false;
        body = el("div", { class: "run-body" });
        log.append(el("li", { class: "run-step" }, el("h2", {}, e.title.replace(/^\d+\.\s*/, "")), body));
        steps.push(e.title);
        body.scrollIntoView({ behavior: "smooth", block: "nearest" });
        break;
      }
      case "text":
        // the pipeline's own log, minus library chatter and what the events above already show
        if (!afterDossier && !e.text.startsWith("[") && !/^\s*\.+$/.test(e.text) && !NOISE.test(e.text.trim())) body.append(el("p", { class: "run-line" }, e.text.trim()));
        break;
      case "chart": {
        const page = { tic, series: e.series, fold: e.fold, ephem: { period: e.period, t0: e.t0 },
          checks: { signal: { depth_ppm: e.depth_ppm, dur_h: e.dur_h, period: e.period } } } as unknown as StarPage;
        body.append(el("div", { class: "run-charts" }, lightCurve(page), foldChart(page)));
        break;
      }
      case "search":
        steps.push(`search: dip every ${fix(e.period, 4)} d, ${Math.round(e.depth_ppm)} ppm, score ${fix(e.snr, 1)} vs bar ${e.floor}: ${e.ok ? "cleared" : "did not clear"}`);
        body.append(el("p", { class: `run-verdict ${e.ok ? "ok" : "no"}` }, e.ok ? `Cleared the bar: score ${fix(e.snr, 1)} against ${e.floor}` : `Below the bar: ${fix(e.snr, 1)} against ${e.floor}`));
        break;
      case "checks":
        steps.push(`code checks: ${e.verdict}; XGBoost rank ${e.xgb_rank == null ? "n/a" : fix(e.xgb_rank, 3)} (its own cut ${e.xgb_cut})`);
        break;
      case "glimmer_start":
        glimmer = el("div", { class: "run-tools" }, el("p", { class: "run-line live" }, "Glimmer is reading the case. Each tool it asks for appears here as it asks."));
        body.append(glimmer);
        break;
      case "tool":
        (glimmer ?? body).append(el("details", { class: "run-tool" }, el("summary", {}, `Glimmer asked: ${TOOL[e.name] ?? e.name}`), el("pre", {}, e.answer)));
        steps.push(`Glimmer called ${e.name}`);
        break;
      case "dossier": {
        steps.push(`Glimmer's verdict${e.replay ? " (replayed)" : ""}: ${e.verdict}`);
        const v = el("div", { class: "run-dossier" },
          el("p", { class: `run-verdict ${e.verdict === "gate2" || e.verdict === "pass" ? "ok" : "no"}` },
            `${e.replay ? "Recorded verdict (replay): " : "Glimmer's verdict: "}${e.verdict === "gate2" ? "a planet, pass it on" : e.verdict}`
            + (e.tokens ? `  ·  ${e.tokens.input.toLocaleString()} tokens in, ${e.tokens.output.toLocaleString()} out, ${e.seconds} s` : "")),
          e.summary ? el("blockquote", { class: "t-quote" }, e.summary) : "",
          el("ul", { class: "run-reasons" }, ...(e.reasons as { check: string; why: string }[]).map((r) => el("li", {}, el("b", {}, r.check), ` ${r.why}`))));
        if (e.replay) v.prepend(el("p", { class: "run-mode rec" }, "REPLAY: Glimmer's recorded answer from the survey run, not a live call"));
        body.append(v);
        afterDossier = true;  // the log prints the same dossier again as text
        break;
      }
      case "gate2":
        steps.push(`gate 2: ${e.status}`);
        body.append(el("p", { class: `run-verdict ${e.status === "candidate" ? "ok" : "no"}` }, `Gate 2: ${e.status}`));
        break;
      case "fpp_start":
        body.append(el("p", { class: "run-line live" }, "TRICERATOPS is fetching the star's pixels and every neighbour within 10 pixels, then weighing a planet against binaries and neighbours."));
        break;
      case "fpp_recorded":
        body.append(el("p", { class: "run-mode rec" }, "RECORDED: the survey run's TRICERATOPS result, not a live run"));
        break;
      case "fpp": {
        steps.push(`TRICERATOPS${e.live ? "" : " (recorded)"}: FPP ${fix(e.fpp, 4)}, NFPP ${e.nfpp.toExponential(1)}, ${e.tier}`);
        body.append(el("div", { class: "run-fpp" },
          el("p", {}, el("b", {}, fix(e.fpp * 100, 2) + "%"), " chance it's a false positive of any kind (FPP)"),
          el("p", {}, el("b", {}, (e.nfpp * 100).toPrecision(2) + "%"), " chance the dip is on a neighbouring star (NFPP)"),
          el("p", { class: `run-verdict ${e.tier === "likely planet" ? "ok" : "no"}` }, `TRICERATOPS: ${e.tier}`)));
        break;
      }
      case "done":
        body.append(el("p", { class: "run-final" }, `TIC ${tic}: ${e.status}`),
          el("button", { type: "button", class: "u-btn", onclick: () => openAsk(`I just ran TIC ${tic} through your pipeline here. What did each step find, and what happens next?`) }, "Ask Tessa about this run"));
        break;
      case "exit":
        if (e.code) body.append(el("p", { class: "err" }, `The run stopped (exit ${e.code}). The lines above say why.`));
        done();
        break;
    }
  };
  es.onerror = () => { body.append(el("p", { class: "err" }, "Lost the local server. Is npm run local still running?")); done(); };
}
