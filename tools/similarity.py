"""How much code Tessa shares with the pre-hackathon research scripts (tess-hunt, not public). Read-only.

Two measures, over every code file of each (tess-hunt without virtual environments, third-party code, caches
and data; Tessa's working tree without node_modules and data):
1. Exact lines: normalised (whitespace collapsed), at least 30 characters, not imports. Lines both share.
2. Copied passages: token 12-grams (identifiers, numbers, operators). For each Tessa file, the share of its
   12-grams that also appear anywhere in tess-hunt; a copied function shows up as a run of them.
Result on 2026-10-28: 23 of 6,850 Tessa lines shared (0.3%), all one-line boilerplate; at most 8.9% of a
file's 12-grams (site/index.html: the standard HTML head; pipeline/fpp.py 6.1%: the same TRICERATOPS calls).

Run: python tools/similarity.py [OLD_DIR] [NEW_DIR]
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

OLD = Path(sys.argv[1] if len(sys.argv) > 1 else r"E:\AI\tess-hunt")
NEW = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent)
EXT = {".py", ".js", ".mjs", ".ts", ".html", ".css", ".sql", ".wgsl", ".cu"}
SKIP = {".venv", "node_modules", "reference-repo", "cache", "data", "__pycache__", "dist", ".git", "results",
        ".wrangler", "public", ".venv-tts", "vendor", ".hyperframes", "site-packages", "preprints"}
N = 12
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+\.?\d*|[^\sA-Za-z0-9_]")


def files(root: Path):
    for p in root.rglob("*"):
        if p.suffix in EXT and p.is_file() and not (set(p.relative_to(root).parts) & SKIP):
            yield p


def lines(text: str):
    for raw in text.splitlines():
        s = " ".join(raw.split())
        if len(s) >= 30 and not re.match(r"^(import |from \S+ import|#|//|\*|/\*)", s):
            yield s


def grams(text: str):
    t = TOKEN.findall(re.sub(r"#.*|//.*", "", text))
    return {tuple(t[i:i + N]) for i in range(len(t) - N + 1)}


def main() -> None:
    old_files, new_files = list(files(OLD)), list(files(NEW))
    old_lines, old_grams = defaultdict(set), set()
    for p in old_files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        for s in lines(text):
            old_lines[s].add(str(p.relative_to(OLD)))
        old_grams |= grams(text)
    print(f"old: {len(old_files)} code files; new: {len(new_files)} code files")
    shared, total, per_file = [], 0, []
    for p in new_files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        ls = list(lines(text))
        total += len(ls)
        shared += [(str(p.relative_to(NEW)), s, sorted(old_lines[s])[0]) for s in ls if s in old_lines]
        g = grams(text)
        if g:
            per_file.append((len(g & old_grams) / len(g), len(g & old_grams), str(p.relative_to(NEW))))
    print(f"\n1. Exact lines (>= 30 chars, not imports): {len(shared)} of {total} new lines also in old")
    for f, s, o in shared:
        print(f"   {f}  ==  {o}\n      {s[:150]}")
    print("\n2. Token 12-grams: files with the most overlap (share of the file's 12-grams found in old)")
    for share, n, f in sorted(per_file, reverse=True)[:12]:
        print(f"   {share:6.1%}  {n:5}  {f}")


if __name__ == "__main__":
    main()
