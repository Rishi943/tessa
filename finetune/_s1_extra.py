"""System 1 with extra inputs (as finetune.system1 --clues all, plus the named feature set), then the dev
comparison against the production version: train out-of-fold + val only, the test split untouched.
  python -m finetune._s1_extra train SET      SET: sector | diffimg | leo | sector+diffimg ...
  python -m finetune._s1_extra compare NAME [NAME ...]"""
import gzip
import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
import finetune.system1 as s1  # noqa: E402

NAN = float("nan")


def num(v):
    return NAN if v is None else float(v)


def packed(name):
    path = s1.FOLDER / "packed" / f"{name}_all.json.gz"
    return json.load(gzip.open(path, "rt", encoding="utf-8")) if path.exists() else {}


SECTOR = packed("sector")
LEO = packed("leo")
SECTOR_KEYS = ("sectors", "oe_max_sigma", "oe_sectors_over_3", "oe_same_sign_share", "sec_half_max_snr",
               "sec_half_sectors_over_3", "sec_half_positive_share")
LEO_KEYS = None  # every numeric metric, plus the 17 pass/fail tests


def extra(sets):
    def add(rows):
        for r in rows:
            x = r["x"]
            if "sector" in sets:
                m = SECTOR.get(r["toi"], {})
                for k in SECTOR_KEYS:
                    x[f"sector_{k}"] = num(m.get(k))
            if "diffimg" in sets:
                cen = (r.get("checks_centroid") or {})
                per = cen.get("sectors") or []
                snrs = [s["diff_snr"] for s in per if s.get("diff_snr") is not None]
                x["diffimg_sectors_used_share"] = len([s for s in per if s.get("used")]) / len(per) if per else NAN
                x["diffimg_median_snr"] = float(np.median(snrs)) if snrs else NAN
                x["diffimg_error_inflation"] = num(cen.get("error_inflation"))
                x["diffimg_err_arcsec"] = num(cen.get("err_arcsec"))
            if "leo" in sets:
                m = LEO.get(r["toi"], {})
                for k, v in (m.get("metrics") or {}).items():
                    if k not in ("tic", "planetno", "per", "epo", "dur") and not k.startswith("error_"):
                        x[f"leo_{k}"] = num(v)
                for k, v in (m.get("fails") or {}).items():
                    x[f"leo_fail_{k}"] = NAN if v is None else float(v)
        return rows
    return add


def load_with(sets):
    orig = s1.load

    def load(*a, **k):
        rows = orig(*a, **k)
        if "diffimg" in sets:  # the centroid's per-sector detail is in the vet files
            for r in rows:
                d = json.loads((s1.FOLDER / "vet" / f"{r['toi']}.json").read_text(encoding="utf-8"))
                r["checks_centroid"] = d["checks"].get("centroid")
        rows = extra(sets)(rows)
        names = sorted({k for r in rows for k in r["x"]})  # every row the same columns
        for r in rows:
            for n in names:
                r["x"].setdefault(n, NAN)
        return rows
    return load


if __name__ == "__main__":
    if sys.argv[1] == "train":
        sets = sys.argv[2].split("+")
        s1.load = load_with(sets)
        sys.argv = ["system1", "--clues", "all", "--name", "x-" + sys.argv[2]]
        s1.main()
    else:
        rows = {r["toi"]: r for r in s1.load()}
        for name in ["clues"] + sys.argv[2:]:
            p = json.loads((s1.OUT / f"preds-{name}.json").read_text(encoding="utf-8"))
            line = []
            for split in ("train", "val"):
                sel = [t for t, r in rows.items() if r["label"]["split"] == split and r["label"]["target"] in ("gate2", "reject")
                       and t in p and r["code"] == "agent"]
                y = np.array([rows[t]["label"]["target"] == "gate2" for t in sel])
                line.append(f"{split} AUC {roc_auc_score(y, [p[t] for t in sel]):.4f}")
            sel = [t for t, r in rows.items() if r["label"]["split"] in ("train", "val") and r["code"] == "agent"
                   and r["label"]["target"] in ("gate2", "reject") and t in p]
            y = np.array([rows[t]["label"]["target"] == "gate2" for t in sel])
            s = np.array([p[t] for t in sel])
            for lost in (0.02, 0.05):
                cut = np.quantile(s[y], lost)
                line.append(f"FPs caught at {lost:.0%} lost {(s[~y] < cut).sum()}/{(~y).sum()}")
            print(f"{name:>22}: " + " | ".join(line))
