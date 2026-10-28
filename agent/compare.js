// Compare two models' dossier verdicts on the same stars: a confusion table, invented numbers,
// and the stars where they disagree.
// Run: node agent/compare.js VET_DIR [--prompt v5] [--kind dossiers|investigations] MODEL_A MODEL_B
import { readFileSync } from "node:fs";
import { join } from "node:path";

const args = process.argv.slice(2);
const opt = (name, fallback) => {
  const at = args.indexOf(name);
  return at >= 0 ? args.splice(at, 2)[1] : fallback;
};
const kind = opt("--kind", "dossiers");
const version = opt("--prompt", kind === "dossiers" ? "v5" : "inv1");
const [dir, a, b] = args;

function rows(model) {
  const [head, ...lines] = readFileSync(join(dir, `${kind}.${model}.${version}.csv`), "utf8").trim().split("\n");
  const cols = head.split(",");
  return new Map(lines.map((l) => {
    const cells = l.split(",");
    const r = Object.fromEntries(cols.map((c, i) => [c, cells[i]]));
    return [r.tic, r];
  }));
}

const A = rows(a), B = rows(b);
const tics = [...A.keys()].filter((t) => B.has(t));
const verdicts = ["gate2", "flag", "reject", "error"];
console.log(`${tics.length} stars in both; rows ${a}, columns ${b}`);
console.log(["", ...verdicts].map((v) => v.padStart(7)).join(""));
for (const va of verdicts) {
  const n = verdicts.map((vb) => tics.filter((t) => A.get(t).verdict === va && B.get(t).verdict === vb).length);
  if (n.some(Boolean)) console.log([va, ...n].map((v) => String(v).padStart(7)).join(""));
}
const same = tics.filter((t) => A.get(t).verdict === B.get(t).verdict).length;
const invented = (M) => tics.reduce((s, t) => s + Number(M.get(t).invented || 0), 0);
console.log(`agree ${same}/${tics.length}; invented numbers ${a} ${invented(A)}, ${b} ${invented(B)}`);
for (const t of tics.filter((t) => A.get(t).verdict !== B.get(t).verdict)) {
  console.log(`  TIC ${t}: ${A.get(t).verdict} vs ${B.get(t).verdict}`);
}
