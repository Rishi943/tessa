// Visitors' searches. The Worker hands each visitor a held-back star nobody has searched yet and records what
// their browser found. Results live in KV: s:<tic> is the first finished search of a star (with the searcher's
// handle, if they gave one); c:<tic> is a short claim, so two visitors searching at once get different stars.
// A browser can report anything, so a star that passes is searched again by the pipeline before it counts.

interface RateLimit { limit(o: { key: string }): Promise<{ success: boolean }> }
export interface SearchEnv {
  ASSETS: Fetcher;
  SEARCHES: KVNamespace;
  SEARCH_LIMIT: RateLimit;
}
interface Star { tic: number; ra: number; dec: number; dist: number; tmag: number; teff: number | null; rad: number | null; mass: number | null }

const CLAIM_S = 900;  // a search takes seconds; a claim lasts long enough for a slow laptop
let stars: Star[] | null = null;

async function list(env: SearchEnv, url: string): Promise<Star[]> {
  stars ??= await (await env.ASSETS.fetch(new Request(new URL("/data/search/stars.json", url)))).json() as Star[];
  return stars;
}
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } });

export async function search(request: Request, env: SearchEnv): Promise<Response> {
  const url = new URL(request.url);
  const all = await list(env, request.url);
  if (url.pathname === "/api/search/stats") return json(await stats(env, all.length));
  const ip = request.headers.get("cf-connecting-ip") ?? "anon";
  if (!(await env.SEARCH_LIMIT.limit({ key: ip })).success) return json({ error: "slow down" }, 429);

  if (url.pathname === "/api/search/next" && request.method === "GET") {
    // a few random tries finds a free star while most are free; then walk the list in order
    const order = [...Array(12)].map(() => all[Math.floor(Math.random() * all.length)]).concat(all);
    for (const s of order) {
      const [done, claimed] = await Promise.all([env.SEARCHES.get(`s:${s.tic}`), env.SEARCHES.get(`c:${s.tic}`)]);
      if (done || claimed) continue;
      await env.SEARCHES.put(`c:${s.tic}`, "1", { expirationTtl: CLAIM_S });
      return json({ star: s, left: (await stats(env, all.length)).left });
    }
    return json({ error: "every held-back star has been searched" }, 404);
  }

  if (url.pathname === "/api/search/result" && request.method === "POST") {
    let b: Record<string, unknown>;
    try { b = await request.json(); } catch { return json({ error: "bad json" }, 400); }
    const tic = Number(b.tic);
    if (!all.some((s) => s.tic === tic)) return json({ error: "not a held-back star" }, 400);
    const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? Math.round(v * 1e5) / 1e5 : null);
    const dip = b.dip && typeof b.dip === "object" ? b.dip as Record<string, unknown> : null;
    const handle = typeof b.handle === "string" ? b.handle.replace(/[^\w@.\- ]/g, "").trim().slice(0, 24) : "";
    const row = {
      tic, pass: b.pass === true, handle, at: new Date().toISOString(),
      dip: dip ? { snr: num(dip.snr), period: num(dip.period), t0: num(dip.t0), dur_h: num(dip.dur_h), depth_ppm: num(dip.depth_ppm), transits: num(dip.transits) } : null,
      top_bump: num(b.topBump), n_freq: num(b.nFreq), seconds: num(b.seconds),
    };
    const first = !(await env.SEARCHES.get(`s:${tic}`));
    if (first) await env.SEARCHES.put(`s:${tic}`, JSON.stringify(row), { metadata: { pass: row.pass ? 1 : 0, handle } });
    return json({ ok: true, first, ...(await stats(env, all.length)) });
  }
  return json({ error: "not found" }, 404);
}

// how many held-back stars visitors have searched, and how many cleared the bar
async function stats(env: SearchEnv, total: number) {
  let searched = 0, passed = 0, cursor: string | undefined;
  do {
    const page = await env.SEARCHES.list<{ pass: number }>({ prefix: "s:", cursor });
    searched += page.keys.length;
    passed += page.keys.filter((k) => k.metadata?.pass).length;
    cursor = page.list_complete ? undefined : page.cursor;
  } while (cursor);
  return { total, searched, passed, left: total - searched };
}
