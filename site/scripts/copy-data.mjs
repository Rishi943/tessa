// Copy the exported survey (python -m pipeline.export) into public/data so Vite serves and ships it.
import { cpSync, existsSync, rmSync } from "node:fs";

const from = new URL("../../data/site/", import.meta.url);
const to = new URL("../public/data/", import.meta.url);
if (!existsSync(from)) throw new Error("no data/site: run python -m pipeline.export first");
rmSync(to, { recursive: true, force: true });
cpSync(from, to, { recursive: true });
