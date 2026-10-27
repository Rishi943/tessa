import type { Status } from "./data";

type Attrs = Record<string, string | number | boolean | null | undefined | EventListener>;
type Kid = Node | string | number | null | undefined | false;

// Tiny DOM builder: el("p", { class: "x", onclick: f }, "text", child)
export function el<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Attrs = {}, ...kids: Kid[]): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v as EventListener);
    else node.setAttribute(k, v === true ? "" : String(v));
  }
  for (const kid of kids) if (kid != null && kid !== false) node.append(kid instanceof Node ? kid : String(kid));
  return node;
}

export function svg(markup: string): SVGElement {
  const t = document.createElement("template");
  t.innerHTML = markup.trim();
  return t.content.firstElementChild as SVGElement;
}

export const int = (n: number) => Math.round(n).toLocaleString("en-US");
export const fix = (n: number | null | undefined, d: number) => (n == null || !isFinite(n) ? "n/a" : n.toFixed(d));
export const usd = (n: number) => (n < 0.01 ? `$${n.toFixed(4)}` : `$${n.toFixed(2)}`);

// Every status has a plain label, a colour token and a shape, so colour never carries it alone.
export const STATUS: Record<Status, { label: string; short: string; color: string; shape: string }> = {
  candidate: { label: "Candidate", short: "candidates", color: "var(--candidate)", shape: "diamond" },
  flagged: { label: "Unexplained", short: "dips Tessa can't explain yet", color: "var(--flagged)", shape: "triangle" },
  rejected: { label: "Explained away", short: "dips explained away, each with a reason", color: "var(--rejected)", shape: "cross" },
  known: { label: "Already known", short: "already known", color: "var(--searched)", shape: "square" },
  "awaiting agent": { label: "Waiting for the agent", short: "waiting for the agent", color: "var(--searched)", shape: "ring" },
  searched: { label: "Searched, no dip", short: "searched, no dip worth a look", color: "var(--searched)", shape: "dot" },
  "held back": { label: "Held back", short: "held back to test the method", color: "var(--pool)", shape: "hollow" },
  pool: { label: "Not searched yet", short: "nearby red stars not searched yet", color: "var(--pool)", shape: "hollow" },
};

export function mark(status: Status, size = 14): SVGElement {
  const { color, shape } = STATUS[status];
  const body: Record<string, string> = {
    diamond: `<path d="M8 1.5 14.5 8 8 14.5 1.5 8z" fill="${color}"/>`,
    triangle: `<path d="M8 2 14.5 13.5h-13z" fill="${color}"/>`,
    cross: `<path d="M3.5 3.5l9 9M12.5 3.5l-9 9" stroke="${color}" stroke-width="2.6" stroke-linecap="round"/>`,
    square: `<rect x="3" y="3" width="10" height="10" rx="1.5" fill="${color}"/>`,
    ring: `<circle cx="8" cy="8" r="4.5" fill="none" stroke="${color}" stroke-width="2"/>`,
    dot: `<circle cx="8" cy="8" r="4" fill="${color}"/>`,
    hollow: `<circle cx="8" cy="8" r="4" fill="none" stroke="${color}" stroke-width="1.6"/>`,
  };
  return svg(`<svg class="sw" viewBox="0 0 16 16" width="${size}" height="${size}" aria-hidden="true">${body[shape]}</svg>`);
}

export function badge(status: Status): HTMLElement {
  return el("span", { class: "badge" }, mark(status), STATUS[status].label);
}

export function navigate(path: string) {
  history.pushState(null, "", path);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

export function css(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// Spectral class from effective temperature, as the TESS Input Catalog would have it
export function spectral(teff: number | null | undefined): string {
  if (teff == null) return "red dwarf";
  return teff >= 3900 ? "K dwarf" : "M dwarf";
}
