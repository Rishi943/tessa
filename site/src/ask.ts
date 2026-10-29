// Ask Tessa: one line at the bottom of every screen. The conversation rises above it, over a fade, and folds
// away again without losing what was said, so it can stay open while you explore or search. It talks to the
// Worker's /api/ask, where Muse Spark answers from Tessa's own files with tools. The Worker streams one JSON
// object per line: steps as she works, pictures and buttons as she calls for them, then the answer.
import { foldChart, lightCurve, sectorDepths, snrBars } from "./charts";
import { getStar } from "./data";
import { explainFigure } from "./explain";
import type { Topic } from "./explain";
import { flyTo } from "./home";
import { searchState } from "./search/card";
import { flyToTic, startSearch } from "./universe";
import { el, navigate } from "./util";

type Turn = { role: "user" | "assistant"; content: string };
const history: Turn[] = [];
const STARTERS = ["What is Tessa, in one minute?", "What has Tessa found so far?", "How do you find a planet you can't see?"];
const SEARCH_STARTER = "Search some stars for me";
const ARROW = '<svg aria-hidden="true" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12h16M14 6l6 6-6 6"></path></svg>';

let root: HTMLElement, talk: HTMLElement, log: HTMLElement, starters: HTMLElement;
let input: HTMLTextAreaElement, send: HTMLButtonElement;

export function setupAsk() {
  root = document.querySelector("aside.ask")!;
  log = el("div", { class: "log", "aria-live": "polite" });
  const close = el("button", { type: "button", class: "ask-close" }, "Close");
  close.addEventListener("click", closeAsk);
  talk = el("div", { class: "ask-talk", hidden: true },
    el("div", { class: "ask-top" }, el("span", {}, "Ask Tessa"), close), log,
    el("p", { class: "foot" }, "Tessa is an AI and can make mistakes."));
  const chip = (q: string, go = false) => el("button", { type: "button", class: go ? "go" : "", onclick: () => ask(q) }, q);
  starters = el("div", { class: "ask-starters", hidden: true }, ...STARTERS.map((q) => chip(q)), chip(SEARCH_STARTER, true));
  input = el("textarea", { rows: 1, placeholder: "Ask Tessa anything, or ask her to search some stars", "aria-label": "Ask Tessa" }) as HTMLTextAreaElement;
  send = el("button", { type: "submit", "aria-label": "Send" }) as HTMLButtonElement;
  send.innerHTML = ARROW;
  const form = el("form", { class: "ask-line" }, input, send);
  root.append(talk, starters, form, el("div", { class: "ask-rule", "aria-hidden": "true" }));
  root.addEventListener("keydown", (e) => { if (e.key === "Escape") closeAsk(); });
  form.addEventListener("submit", (e) => { e.preventDefault(); ask(input.value); });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(input.value); }
  });
  input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = `${input.scrollHeight}px`; });
  // the starters offer themselves when the line is empty and nothing has been asked yet
  input.addEventListener("focus", () => { starters.hidden = history.length > 0 || !talk.hidden; });
  root.addEventListener("focusout", (e) => { if (!root.contains(e.relatedTarget as Node)) starters.hidden = true; });
  log.addEventListener("click", (e) => {
    const a = (e.target as HTMLElement).closest("a");
    if (a?.getAttribute("href")?.startsWith("/")) {
      e.preventDefault();
      navigate(a.getAttribute("href")!);
    }
  });
  // every jargon word on the site asks Tessa about itself when clicked
  const term = (e: Event) => {
    const t = (e.target as HTMLElement).closest<HTMLElement>(".term");
    if (!t || (e instanceof KeyboardEvent && e.key !== "Enter")) return;
    e.stopPropagation();
    openAsk(`What does "${t.textContent}" mean, simply?`);
  };
  document.addEventListener("click", term, true);
  document.addEventListener("keydown", term, true);
  // a search tells Tessa when a star clears the bar and when it ends; she says so only while the talk is open
  addEventListener("tessa:search", (e) => {
    const d = (e as CustomEvent).detail as { kind: string; tic?: number; period?: number; k: number; n: number; found: number };
    if (talk.hidden) return;
    const text = d.kind === "cleared"
      ? [el("b", {}, `TIC ${d.tic} cleared the bar:`), ` a dip every ${d.period!.toFixed(1)} days. I'll put it through the rest of my checks; most turn out to be two stars eclipsing.`]
      : [`Your search is done: ${d.k} star${d.k === 1 ? "" : "s"}, ${d.found ? `${d.found} cleared the bar` : "none cleared the bar, which is what most stars do"}.`];
    log.append(el("div", { class: "ping" }, el("span", { class: "when" }, d.kind === "cleared" ? `Star ${d.k} of ${d.n}` : "Search finished"), el("p", {}, ...text)));
    log.scrollTop = log.scrollHeight;
  });
}

// the tour's first stop points at the line, which pulses three times
export function pulseAsk() {
  root.classList.add("pulse");
  setTimeout(() => root.classList.remove("pulse"), 5200);
}

export function openAsk(question?: string) {
  talk.hidden = false;
  starters.hidden = true;
  document.body.classList.add("asking");
  input.focus();
  if (question) ask(question);
}

function closeAsk() {
  talk.hidden = true;
  starters.hidden = true;
  document.body.classList.remove("asking");
}

// Plain text in, a little structure out: paragraphs, bullets, bold, and TIC numbers as links
function render(text: string): HTMLElement {
  const box = el("div", { class: "msg-tessa" });
  const esc = (s: string) => s.replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[ch]!);
  const inline = (s: string) => esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/\bTIC (\d{4,10})\b/g, '<a href="/star/$1">TIC $1</a>');
  for (const block of text.trim().split(/\n\s*\n/)) {
    const lines = block.split("\n");
    if (lines.every((l) => /^\s*([-*]|\d+\.)\s/.test(l))) {
      const ul = el("ul");
      ul.innerHTML = lines.map((l) => `<li>${inline(l.replace(/^\s*([-*]|\d+\.)\s/, ""))}</li>`).join("");
      box.append(ul);
    } else {
      const p = el("p");
      p.innerHTML = lines.map(inline).join("<br>");
      box.append(p);
    }
  }
  return box;
}

// A chart from the star's own exported data, drawn by the same code as its page, or a flight into its system
async function picture(tic: number, view: string): Promise<HTMLElement | null> {
  if (view === "map") {
    if (!flyToTic(tic)) flyTo(tic);
    return null;
  }
  try {
    const page = await getStar(tic);
    const chart = view === "fold" ? foldChart(page) : view === "light_curve" ? lightCurve(page)
      : view === "depth_by_sector" ? sectorDepths(page) : page.when ? snrBars(page.when.ground_snr_per_transit) : null;
    return chart && el("figure", { class: "x-fig" }, el("figcaption", {}, el("a", { href: `/star/${tic}` }, `TIC ${tic}`), ", from its case file"), chart);
  } catch {
    return null;
  }
}

// Tessa offers a search; it starts only when the reader clicks, and the talk folds away so the search shows
function offer(n: number): HTMLElement {
  const go = (k: number) => () => { startSearch(k); closeAsk(); };
  return el("div", { class: "acts" },
    el("button", { type: "button", class: "solid", onclick: go(n) }, n === 1 ? "Search one star" : `Start a ${n}-star search`),
    n > 1 ? el("button", { type: "button", onclick: go(1) }, "Just one star") : null);
}

async function ask(question: string) {
  const q = question.trim();
  if (!q || send.disabled) return;
  talk.hidden = false;
  starters.hidden = true;
  document.body.classList.add("asking");
  input.value = "";
  input.style.height = "auto";
  input.placeholder = "Ask a follow-up";
  log.append(el("p", { class: "msg-user" }, q));
  history.push({ role: "user", content: q });
  send.disabled = true;
  const turn = el("div", { class: "turn" }, el("div", { class: "who" }, el("b", {}, "Tessa"), "Muse Spark"));
  const body = el("div", { class: "turn-body" });
  turn.append(body);
  const steps = el("div", { class: "steps" });
  const thinking = el("div", { class: "thinking" }, "Thinking");
  steps.append(thinking);
  body.append(steps);
  log.append(turn);
  log.scrollTop = log.scrollHeight;
  try {
    const res = await fetch("/api/ask", { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages: history.slice(-8), page: location.pathname, search: searchState() }) });
    if (res.status === 429) throw new Error("I'm answering a lot of questions right now. Try again in a minute.");
    if (!res.ok || !res.body) throw new Error(`I couldn't answer that (error ${res.status}). Try again.`);
    const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
    let buf = "", answered = false;
    const after: HTMLElement[] = [];  // buttons go under the answer
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += value;
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        const ev = JSON.parse(line) as { type: string; text: string; tic: number | null; view: string; topic: Topic; stars: number };
        if (ev.type === "step") {
          thinking.remove();
          steps.append(el("div", { class: "step" }, ev.text));
        } else if (ev.type === "show" && ev.tic) {
          const fig = await picture(ev.tic, ev.view);
          if (fig) body.append(fig);
        } else if (ev.type === "explain") {
          const fig = await explainFigure(ev.topic, ev.tic);
          if (fig) body.append(fig);
        } else if (ev.type === "offer") {
          after.push(offer(ev.stars));
        } else if (ev.type === "answer") {
          thinking.remove();
          body.append(render(ev.text), ...after);
          history.push({ role: "assistant", content: ev.text });
          answered = true;
        } else if (ev.type === "error") {
          throw new Error(ev.text);
        }
        log.scrollTop = log.scrollHeight;
      }
    }
    if (!answered) throw new Error("My answer was cut off. Try asking again.");
  } catch (e) {
    thinking.remove();
    history.pop();
    body.append(el("p", { class: "err" }, (e as Error).message));
  } finally {
    send.disabled = false;
    log.scrollTop = log.scrollHeight;
  }
}
