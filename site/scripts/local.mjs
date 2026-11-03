// npm run local: the site with everything running on this machine. tools/local_run.py (the "Run a star" page's
// pipeline runs, keys from the repo's .env), the Worker (Ask Tessa, META_API_KEY in site/.dev.vars) and vite.
// Ctrl+C stops all three.
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..", "..");
const py = [resolve(root, ".venv/Scripts/python.exe"), resolve(root, ".venv/bin/python")].find(existsSync) ?? "python";
const kids = [
  spawn(py, [resolve(root, "tools/local_run.py")], { stdio: "inherit" }),
  spawn("npx", ["wrangler", "dev", "--port", "8787"], { stdio: "inherit", shell: true }),
  spawn("npx", ["vite"], { stdio: "inherit", shell: true }),
];
const stop = () => { for (const k of kids) k.kill(); process.exit(); };
process.on("SIGINT", stop);
process.on("SIGTERM", stop);
for (const k of kids) k.on("exit", (code) => { if (code) console.log(`a process stopped (exit ${code})`); });
