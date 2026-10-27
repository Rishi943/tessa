// Ask Tessa: a drawer that talks to the Worker's /api/ask, where Muse Spark answers from Tessa's own
// files with tools. The Worker streams one JSON object per line: steps as it works, then the answer.
import { foldChart, lightCurve, sectorDepths, snrBars } from "./charts";
import { getStar } from "./data";
import { flyTo } from "./home";
import { el, navigate } from "./util";

type Turn = { role: "user" | "assistant"; content: string };
const history: Turn[] = [];
const STARTERS = [
  "Which dip is most likely a planet, and why?",
  "When is the next transit of TIC 458191673 I could see?",
  "Why was TIC 110220637 rejected?",
  "What are the flagged dips that nobody can explain yet?",
];

let drawer: HTMLElement, log: HTMLElement, input: HTMLTextAreaElement, send: HTMLButtonElement, opener: HTMLElement | null = null;

export function setupAsk() {
  drawer = document.querySelector("aside.ask")!;
  log = el("div", { class: "log", "aria-live": "polite" });
  input = el("textarea", { rows: 1, placeholder: "Ask about a star, a dip or the method", "aria-label": "Your question" }) as HTMLTextAreaElement;
  send = el("button", { type: "submit" }, "Ask") as HTMLButtonElement;
  const form = el("form", {}, input, send);
  const close = el("button", { class: "close", type: "button", "aria-label": "Close" }, "×");
  drawer.append(
    el("header", {}, el("h2", {}, "Ask Tessa"), el("p", {}, "Muse Spark 1.3, reading Tessa's files"), close),
    log, form,
    el("p", { class: "foot" }, "Answers come from the survey's own data through tools. They can still be wrong; every star page shows the evidence."));
  starters();
  close.addEventListener("click", closeAsk);
  drawer.addEventListener("keydown", (e) => { if (e.key === "Escape") closeAsk(); });
  form.addEventListener("submit", (e) => { e.preventDefault(); ask(input.value); });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(input.value); }
  });
  input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = `${input.scrollHeight}px`; });
  document.querySelectorAll<HTMLElement>(".bar .ask-open").forEach((b) => b.addEventListener("click", () => openAsk()));
  log.addEventListener("click", (e) => {
    const a = (e.target as HTMLElement).closest("a");
    if (a?.getAttribute("href")?.startsWith("/")) {
      e.preventDefault();
      navigate(a.getAttribute("href")!);
    }
  });
}

function starters() {
  log.append(el("div", { class: "starters" }, ...STARTERS.map((q) => el("button", { type: "button", onclick: () => ask(q) }, q))));
}

export function openAsk(question?: string) {
  opener = document.activeElement as HTMLElement;
  drawer.hidden = false;
  input.focus();
  if (question) ask(question);
}

function closeAsk() {
  drawer.hidden = true;
  opener?.focus();
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

// A chart from the star's own exported data, drawn by the same code as its page
async function picture(tic: number, view: string): Promise<HTMLElement | null> {
  if (view === "map") {
    flyTo(tic);
    return null;
  }
  try {
    const page = await getStar(tic);
    const chart = view === "fold" ? foldChart(page) : view === "light_curve" ? lightCurve(page)
      : view === "depth_by_sector" ? sectorDepths(page) : page.when ? snrBars(page.when.ground_snr_per_transit) : null;
    return chart && el("figure", { class: "ask-fig" }, chart,
      el("figcaption", {}, el("a", { href: `/star/${tic}` }, `TIC ${tic}`), ", from its case file"));
  } catch {
    return null;
  }
}

async function ask(question: string) {
  const q = question.trim();
  if (!q || send.disabled) return;
  log.querySelector(".starters")?.remove();
  input.value = "";
  input.style.height = "auto";
  log.append(el("div", { class: "msg-user" }, q));
  history.push({ role: "user", content: q });
  send.disabled = true;
  const steps = el("div", { class: "steps-live" });
  log.append(steps);
  const thinking = el("div", { class: "step" }, "Thinking");
  steps.append(thinking);
  log.scrollTop = log.scrollHeight;
  try {
    const res = await fetch("/api/ask", { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages: history.slice(-8), page: location.pathname }) });
    if (res.status === 429) throw new Error("Tessa is answering a lot of questions right now. Try again in a minute.");
    if (!res.ok || !res.body) throw new Error(`Tessa could not answer (error ${res.status}). Try again.`);
    const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
    let buf = "", answered = false;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += value;
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        const ev = JSON.parse(line) as { type: string; text: string; tic: number; view: string };
        if (ev.type === "step") {
          thinking.remove();
          steps.append(el("div", { class: "step" }, ev.text));
        } else if (ev.type === "show") {
          thinking.remove();
          const fig = await picture(ev.tic, ev.view);
          if (fig) log.append(fig);
        } else if (ev.type === "answer") {
          thinking.remove();
          log.append(render(ev.text));
          history.push({ role: "assistant", content: ev.text });
          answered = true;
        } else if (ev.type === "error") {
          throw new Error(ev.text);
        }
        log.scrollTop = log.scrollHeight;
      }
    }
    if (!answered) throw new Error("The answer was cut off. Try asking again.");
  } catch (e) {
    thinking.remove();
    history.pop();
    log.append(el("p", { class: "err" }, (e as Error).message));
  } finally {
    send.disabled = false;
    log.scrollTop = log.scrollHeight;
  }
}
