import "./style.css";
import { openAsk, setupAsk } from "./ask";
import { home, leaveHome } from "./home";
import { starPage } from "./star";
import { el, navigate } from "./util";

const main = document.getElementById("main")!;
const TITLE = document.title;

async function route() {
  leaveHome();
  main.replaceChildren(el("p", { class: "loading" }, "Loading the survey"));
  const star = location.pathname.match(/^\/star\/(\d+)\/?$/);
  const view = el("div");
  try {
    if (star) await starPage(view, Number(star[1]));
    else {
      document.title = TITLE;
      await home(view);
    }
  } catch (e) {
    view.replaceChildren(el("p", { class: "loading" }, `Could not load the survey data (${(e as Error).message}). Reload to try again.`));
  }
  main.replaceChildren(...view.childNodes);
  // children moved out of `view` keep their listeners; the sky canvas re-measures on its own
  const id = location.hash.slice(1);
  const target = id ? document.getElementById(id) : null;
  if (target) target.scrollIntoView();
  else window.scrollTo(0, 0);
  if (star) main.focus({ preventScroll: true });
}

document.addEventListener("click", (e) => {
  const a = (e.target as HTMLElement).closest("a[data-link]") as HTMLAnchorElement | null;
  if (!a || e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
  const url = new URL(a.href);
  if (url.origin !== location.origin) return;
  e.preventDefault();
  if (url.pathname === location.pathname && url.hash) {
    history.pushState(null, "", url.pathname + url.hash);
    document.getElementById(url.hash.slice(1))?.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    return;
  }
  navigate(url.pathname + url.hash);
});
window.addEventListener("popstate", route);

document.querySelector(".theme")!.addEventListener("click", () => {
  const dark = getComputedStyle(document.documentElement).colorScheme.includes("dark");
  const next = dark ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch { /* private mode: the toggle still works for this visit */ }
});

setupAsk();
route();
// Shareable questions: /?ask=When+is+the+next+transit
const shared = new URLSearchParams(location.search).get("ask");
if (shared) openAsk(shared.slice(0, 300));

