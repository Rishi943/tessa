"""Novelty check B: is a signal that survived vetting already known?

Run on survivors just before gate 2, against lists fetched fresh: TOIs and CTOIs (this star or any
within the vetting's neighbour radius), confirmed planets and TESS eclipsing binaries, the AAVSO
variable-star index (VSX), Gaia DR3's eclipsing binaries among the stars bright enough to make the
dip, and SIMBAD's object types. Matches are by period or a simple harmonic, so a new planet around
a known host, or a transit on a known spotted star, still counts as new. NASA ADS is not searched:
it needs an API token.

Run: python -m pipeline.novelty TIC --results results.csv
Prints JSON.
"""
import argparse
import contextlib
import json
import sys
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord

from . import catalogs
from .vet import MAX_ECLIPSE_DEPTH, gaia_field, harmonic, known, signal_from_results, star_info

VSX_ARCSEC = 60  # about 3 TESS pixels
SIMBAD_ARCSEC = 30
ON_TARGET_ARCSEC = 5
SIMBAD_BINARY = {"EB*": "an eclipsing binary", "EB?": "a candidate eclipsing binary", "El*": "an ellipsoidal variable",
                 "SB*": "a spectroscopic binary"}
VSX_ECLIPSING = ("E", "EA", "EB", "EW", "ELL")
SIMBAD_SERVERS = ("simbad.cds.unistra.fr", "simbad.harvard.edu")
VIZIER_SERVERS = ("vizier.cds.unistra.fr", "vizier.idia.ac.za", "vizier.nao.ac.jp")


def toi_lists(tic: int, star: dict, sig: dict) -> tuple[list, list]:
    catalogs.toi(max_age_days=1), catalogs.ctoi(max_age_days=1), catalogs.confirmed(max_age_days=1)
    hits = known(tic, star, sig)
    notes = [f"{k['list']} {k['id']} ({'this star' if k['tic'] == tic else str(k['separation_arcsec']) + chr(34) + ' away'})"
             f" at P = {k['period']:.4f} d, not this period" for k in hits if k["harmonic"] is None]
    return [k for k in hits if k["harmonic"] is not None], notes


def mirrored(servers: tuple[str, ...], query):
    """query(server) on each mirror in turn until one answers (CDS itself was unreachable on 2026-10-29)."""
    for i, server in enumerate(servers):
        try:
            return query(server)
        except Exception:
            if i == len(servers) - 1:
                raise


def vsx(star: dict, sig: dict) -> tuple[list, list]:
    from astroquery.vizier import Vizier
    got = mirrored(VIZIER_SERVERS, lambda s: Vizier(columns=["Name", "Type", "Period", "_r"], row_limit=50, vizier_server=s)
                   .query_region(SkyCoord(star["ra"], star["dec"], unit="deg"), radius=VSX_ARCSEC * u.arcsec, catalog="B/vsx/vsx"))
    matches, notes = [], []
    for r in (got[0] if len(got) else []):
        p = float(r["Period"]) if r["Period"] is not None and str(r["Period"]) != "--" else None
        entry = {"list": "VSX", "id": str(r["Name"]), "type": str(r["Type"]), "period": p,
                 "separation_arcsec": round(float(r["_r"]), 1)}
        h = harmonic(p, sig["period"]) if p else None
        if h is not None:
            matches.append({**entry, "harmonic": h})
        elif p or str(r["Type"]).split("+")[0].split(":")[0] in VSX_ECLIPSING or entry["separation_arcsec"] < ON_TARGET_ARCSEC:
            notes.append(f"VSX {entry['id']} ({entry['type']}, {entry['separation_arcsec']}\" away)"
                         + (f" at P = {p:.4f} d, not this period" if p else ""))
    return matches, notes


def gaia_ebs(tic: int, star: dict, sig: dict) -> tuple[list, list]:
    field = gaia_field(tic, star)
    if field is None:
        return [], ["target not found in Gaia, so Gaia's eclipsing binaries were not checked"]
    df, me = field
    need = sig["depth_ppm"] * 1e-6 * 10 ** (0.4 * (df["mag"] - me["mag"]))
    able = df[need <= MAX_ECLIPSE_DEPTH]
    ids = ", ".join(str(int(s)) for s in able["source_id"])
    with contextlib.redirect_stdout(sys.stderr):
        from astroquery.gaia import Gaia
        rows = Gaia.launch_job(f"SELECT source_id, frequency FROM gaiadr3.vari_eclipsing_binary "
                               f"WHERE source_id IN ({ids})", verbose=False).get_results()
    sep = dict(zip(able["source_id"].astype("int64"), able["sep"]))
    matches, notes = [], []
    for r in rows:
        p = 1 / float(r["frequency"])
        entry = {"list": "Gaia DR3 EB", "id": str(int(r["source_id"])), "period": p,
                 "separation_arcsec": round(float(sep[int(r["source_id"])]), 1)}
        h = harmonic(p, sig["period"])
        if h is not None:
            matches.append({**entry, "harmonic": h})
        else:
            notes.append(f"Gaia DR3 EB {entry['id']} ({entry['separation_arcsec']}\" away) at P = {p:.4f} d, not this period")
    return matches, notes


def simbad(star: dict) -> tuple[list, list]:
    from astroquery.simbad import Simbad

    def query(server):
        sb = Simbad()
        sb.server = server
        sb.add_votable_fields("otype")
        return sb.query_region(SkyCoord(star["ra"], star["dec"], unit="deg"), radius=SIMBAD_ARCSEC * u.arcsec)
    got = mirrored(SIMBAD_SERVERS, query)
    notes = []
    target = SkyCoord(star["ra"], star["dec"], unit="deg")
    for r in (got if got is not None else []):
        kind = SIMBAD_BINARY.get(str(r["otype"]))
        if kind:
            sep = target.separation(SkyCoord(r["ra"], r["dec"], unit="deg")).arcsec
            notes.append(f"SIMBAD lists {r['main_id']} ({sep:.0f}\" away) as {kind}")
    return [], notes


def novelty(tic: int, sig: dict) -> dict:
    star = star_info(tic)
    matches, notes, errors = [], [], []
    for name, check in [("TOI/CTOI/confirmed/TESS EB", lambda: toi_lists(tic, star, sig)), ("VSX", lambda: vsx(star, sig)),
                        ("Gaia DR3 EB", lambda: gaia_ebs(tic, star, sig)), ("SIMBAD", lambda: simbad(star))]:
        try:
            m, n = check()
            matches += m
            notes += n
        except Exception as e:
            errors.append(f"{name}: {e.__class__.__name__}: {e}"[:200])
    return {"tic": tic, "new": not matches and not errors, "complete": not errors,
            "matches": matches, "notes": notes, "errors": errors}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tic", type=int)
    ap.add_argument("--results", required=True, help="search results.csv with this star's signal")
    args = ap.parse_args()
    print(json.dumps(novelty(args.tic, signal_from_results(args.tic, Path(args.results))), indent=1, default=float))


if __name__ == "__main__":
    main()
