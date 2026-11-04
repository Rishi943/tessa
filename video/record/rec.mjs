// Records the demo video's screen footage from the real site: a real Chrome (GPU on) driven by a script, the page
// recorded at 1920x1080 by Chrome's own screencast, so every clip is the app running, with smooth, repeatable
// camera moves and no stray cursor. Clips land in video/film/assets/footage/.
//
//   node video/record/rec.mjs <clip> [base]     clip: 03 05 run 10 11 12 (see video/SHOTS.md); base default
//                                                http://localhost:5173 (cd site && npm run local, or its 3 parts)
import { mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "file:///C:/Users/rushi/AppData/Local/npm-cache/_npx/cc5149aa1e8073f3/node_modules/puppeteer-core/lib/puppeteer/puppeteer-core.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = resolve(HERE, "../film/assets/footage");
const CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe";
const FFMPEG = "C:/Users/rushi/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1.2-full_build/bin/ffmpeg.exe";
const [clip, base = "http://localhost:5173"] = process.argv.slice(2);
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
mkdirSync(OUT, { recursive: true });

const browser = await puppeteer.launch({
  executablePath: CHROME, headless: false, defaultViewport: { width: 1536, height: 864, deviceScaleFactor: 1.25 },  // 125% zoom: 1920x1080 out, readable on video
  args: ["--window-size=1920,1160", "--hide-scrollbars", "--no-first-run",
    "--disable-infobars", "--autoplay-policy=no-user-gesture-required", "--ignore-gpu-blocklist", "--enable-gpu-rasterization"],
  ignoreDefaultArgs: ["--enable-automation"],
});
const [page] = await browser.pages();
page.on("pageerror", (e) => console.log("page error:", e.message));
page.on("framenavigated", (f) => { if (f === page.mainFrame()) console.log(new Date().toISOString().slice(11, 19), "navigated:", f.url()); });

// smooth scroll of the document to a y (or an element's top minus an offset), eased both ways
async function scrollTo(target, ms, offset = 80) {
  await page.evaluate(async (target, ms, offset) => {
    const y1 = typeof target === "number" ? target : document.querySelector(target).getBoundingClientRect().top + scrollY - offset;
    const y0 = scrollY, t0 = performance.now();
    const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
    await new Promise((done) => {
      const step = (now) => {
        const t = Math.min((now - t0) / ms, 1);
        scrollTo(0, y0 + (y1 - y0) * ease(t));
        t < 1 ? requestAnimationFrame(step) : done();
      };
      requestAnimationFrame(step);
    });
  }, target, ms, offset);
}
// keep an element in view while it grows (the run log), by easing the page to its bottom now and then
async function follow(sel, ms, every = 700) {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    await page.evaluate((sel) => {
      const el = document.querySelector(sel);
      if (!el) return;
      const want = el.getBoundingClientRect().bottom + scrollY - innerHeight + 160;
      if (want > scrollY + 40) scrollTo({ top: want, behavior: "smooth" });
    }, sel);
    await wait(every);
  }
}
async function go(path, settle = 2500) {
  await page.goto(base + path, { waitUntil: "networkidle2", timeout: 90000 });
  await wait(settle);
}
async function record(name, fn) {
  const rec = await page.screencast({ path: resolve(OUT, `${name}.webm`), fps: 30, quality: 14, ffmpegPath: FFMPEG });
  const t0 = Date.now();
  try { await fn(); } finally { await rec.stop(); }
  console.log(`${name}.webm, ${((Date.now() - t0) / 1000).toFixed(1)} s`);
}

// an element by its visible text (buttons and summaries)
async function byText(sel, text, timeout = 60000) {
  const h = await page.waitForFunction((sel, text) => [...document.querySelectorAll(sel)].find((e) => e.textContent.includes(text) && e.offsetWidth > 0),
    { timeout, polling: 250 }, sel, text);
  return h.asElement();
}
// a slow mouse drag across the middle of the 3D map, the way a person turns it
async function drag(dx, dy, ms) {
  const x = 768, y = 470, n = Math.round(ms / 16);
  await page.mouse.move(x, y);
  await page.mouse.down();
  for (let i = 1; i <= n; i++) {
    const t = i / n, e = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
    await page.mouse.move(x + dx * e, y + dy * e);
    await wait(16);
  }
  await page.mouse.up();
}
// the panel whose heading says h2
const panel = (h2) => page.evaluate((h2) => {
  const s = [...document.querySelectorAll("section.panel")].find((x) => x.querySelector("h2")?.textContent.includes(h2));
  s?.setAttribute("data-shot", h2.slice(0, 12)); return !!s;
}, h2);

const clips = {
  // the How it works tour from its dip stop, at its own pace (it moves on by itself)
  async "03"() {
    await go("/?tour=2", 4000);
    await record("03-problem", async () => { await wait(32000); });
  },
  // the map: the opening's stats, then into the map, a turn, and Tessa's own finds lit
  async "05"() {
    await go("/", 1500);
    const skip = await page.$(".o-skip");
    if (skip) await skip.click();
    await wait(2500);
    await record("05-map", async () => {
      await wait(3500);
      await (await page.$(".o-explore")).click();
      await wait(2200);
      await drag(-420, 60, 3200);
      await wait(800);
      await (await byText(".u-stats button", "found by Tessa")).click();
      await wait(5000);
    });
  },
  // 06 to 09 in one take: the whole pipeline on TIC 71503196, live on this machine; sped up in the edit
  async run() {
    await go("/run/71503196", 2500);
    await record("run", async () => {
      await wait(2500);
      await (await page.$(".run-go")).click();
      let opened = false;
      const end = Date.now() + 15 * 60000;
      while (Date.now() < end) {
        await follow(".run-log", 1400, 700);
        if (!opened) {  // open the first answer Glimmer gets back, to show the tool really ran
          const t = await page.$(".run-tool summary");
          if (t) { await wait(1500); await t.click(); opened = true; }
        }
        if (await page.$(".run-final, .run-log .err")) break;
      }
      await follow(".run-log", 2500, 600);
      await wait(4000);
    });
  },
  // join in: Ask Tessa starts a one-star search in the browser, then answers a question
  async "11"() {
    await go("/", 1500);
    const skip = await page.$(".o-skip");
    if (skip) await skip.click();
    await wait(2500);
    await record("11-join", async () => {
      await wait(1500);
      await page.click(".ask-line textarea");
      await wait(1200);
      await (await byText(".ask-starters button", "Search some stars")).click();
      await (await byText(".acts button", "Just one star", 90000)).click();
      await wait(9000);
      await page.click(".ask-line textarea");
      await page.keyboard.type("What is Tessa, in one minute?", { delay: 45 });
      await wait(400);
      await page.keyboard.press("Enter");
      await page.waitForFunction(() => document.querySelectorAll(".ask .msg-tessa").length >= 2, { timeout: 120000, polling: 300 }).catch(() => {});
      await wait(7000);
    });
  },
  // a candidate's case file, down to the prediction: the next transit, dated
  async "12"() {
    await go("/star/71503196", 2500);
    await panel("How to prove Tessa wrong");
    await record("12-prediction", async () => {
      await wait(3500);
      await scrollTo("[data-shot]", 4200, 90);
      await wait(7000);
    });
  },
  // the accuracy section: in from the method steps, the squares fill, then down to ExoMiner++
  async 10() {
    await go("/survey");
    // the section's top at the top of the frame (the cost line above stays off screen: the film doesn't mention prices)
    await scrollTo("#accuracy", 10, 30);
    await page.evaluate(() => document.querySelector("#accuracy").classList.remove("in"));
    await wait(1200);
    await record("10-accuracy", async () => {
      await wait(1500);
      await page.evaluate(() => document.querySelector("#accuracy").classList.add("in"));  // the squares fill, as on scroll
      await wait(8000);
      await scrollTo(".acc-note", 2200, 640);
      await wait(8000);
    });
  },
};

if (!clips[clip]) { console.log(`clips: ${Object.keys(clips).join(" ")}`); await browser.close(); process.exit(1); }
await clips[clip]();
await browser.close();
