// The checks without what names the star: the fine-tuned vetter learns from TOIs whose outcomes are public,
// so an ID, a position or an exact period would let it recall the answer instead of reading the checks.
// Used for its training data and for every dossier it writes, so it never sees the two differently.
const sig3 = (v) => (typeof v === "number" ? Number(v.toPrecision(3)) : v);

export function anonymize(checks) {
  const c = structuredClone(checks);
  delete c.tic;
  c.known = []; // survey stars have none by construction, and TOIs are vetted blind
  for (const k of ["tic", "source_id", "ra", "dec", "upcoming_sectors"]) delete c.star?.[k];
  for (const part of [c.signal, c.refit]) {
    if (!part) continue;
    delete part.t0;
    part.period = sig3(part.period);
  }
  for (const d of c.system1?.drivers ?? []) d.value = sig3(d.value); // a driver can be the period
  for (const n of c.neighbours?.closest_that_could ?? []) delete n.source_id;
  if (c.centroid) {
    for (const s of c.centroid.suspects ?? []) delete s.source_id;
    if (c.centroid.on_star) delete c.centroid.on_star.source_id;
    for (const s of c.centroid.sectors ?? []) delete s.mid_btjd;
    if (typeof c.centroid.why === "string") c.centroid.why = c.centroid.why.replace(/Gaia \d+/g, "a Gaia star");
  }
  c.reasons = (c.reasons ?? []).map((r) => r.replace(/Gaia \d+/g, "a Gaia star").replace(/TIC \d+/g, "a nearby star"));
  return c;
}
