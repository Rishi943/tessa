// Put the exported survey in public/data so Vite serves and ships it: from data/site (python -m pipeline.export) when
// this machine ran the pipeline, else downloaded from the live site, which serves the same export. The download takes
// the map, the dip list, every case file and the pool lookup (about 105 MB); not the visitor-search light curves
// (4,000 files, 3.6 GB), so "Search a star" works on the hosted site only.
import { cpSync, existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";

const from = new URL("../../data/site/", import.meta.url);
const to = new URL("../public/data/", import.meta.url);
const LIVE = "https://tessa.braveai.solutions/data/";

if (existsSync(from)) {
  rmSync(to, { recursive: true, force: true });
  cpSync(from, to, { recursive: true });
} else if (existsSync(new URL("summary.json", to)) && !process.argv.includes("--refresh")) {
  console.log("public/data is already there (npm run data -- --refresh downloads it again)");
} else {
  console.log(`no data/site here: downloading the export from ${LIVE}`);
  rmSync(to, { recursive: true, force: true });
  for (const d of ["stars", "pool", "search"]) mkdirSync(new URL(d, to), { recursive: true });
  const get = async (path) => {
    for (let attempt = 1; ; attempt++) {
      try {
        const r = await fetch(LIVE + path, { headers: { "user-agent": "tessa-site-local/1.0" } });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        writeFileSync(new URL(path, to), Buffer.from(await r.arrayBuffer()));
        return;
      } catch (e) {
        if (attempt === 4) throw new Error(`${path}: ${e.message}`);
        await new Promise((ok) => setTimeout(ok, 1000 * attempt));
      }
    }
  };
  const top = ["summary.json", "sky.json", "dips.json", "planets.json", "search/stars.json"];
  await Promise.all(top.map(get));
  const dips = (await import(new URL("dips.json", to), { with: { type: "json" } })).default;
  const rest = [...dips.map((d) => `stars/${d.tic}.json`), ...Array.from({ length: 100 }, (_, i) => `pool/${i}.json`)];
  let done = 0;
  const worker = async () => {
    while (rest.length) {
      await get(rest.pop());
      if (++done % 100 === 0) console.log(`${done} files`);
    }
  };
  await Promise.all(Array.from({ length: 16 }, worker));
  console.log(`${top.length + done} files in public/data`);
}
