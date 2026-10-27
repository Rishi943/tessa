// Visitors' searches. The Worker hands each visitor a star nobody has searched yet and records what their browser
// found, in D1 (migrations/0001_searches.sql): results has the first finished search of each star, with the
// searcher's handle if they gave one; claims holds a star for 15 minutes, so two visitors searching at once get
// different stars; totals keeps the counts. A browser can report anything, so a star that passes is searched
// again by the pipeline before it counts.

interface RateLimit { limit(o: { key: string }): Promise<{ success: boolean }> }
export interface SearchEnv {
  ASSETS: Fetcher;
  DB: D1Database;
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
    const now = Math.floor(Date.now() / 1000);
    const taken = async (tics: number[] | null) => {
      const where = tics ? ` AND tic IN (${tics.join(",")})` : "";
      const r = await env.DB.prepare(`SELECT tic FROM results WHERE 1${where} UNION SELECT tic FROM claims WHERE until > ?${where}`)
        .bind(now).all<{ tic: number }>();
      return new Set(r.results.map((x) => x.tic));
    };
    // 32 random stars almost always include a free one; when most are gone, read every taken one instead
    const sample = [...new Set([...Array(32)].map(() => all[Math.floor(Math.random() * all.length)]))];
    const busy = await taken(sample.map((s) => s.tic));
    let free = sample.filter((s) => !busy.has(s.tic));
    if (!free.length) {
      const t = await taken(null);
      free = all.filter((s) => !t.has(s.tic));
    }
    for (const s of free) {
      // the claim only lands if nobody holds the star: one statement, so two visitors can't both win it
      const r = await env.DB.prepare("INSERT INTO claims (tic, until) VALUES (?1, ?2) ON CONFLICT (tic) DO UPDATE SET until = ?2 WHERE until <= ?3")
        .bind(s.tic, now + CLAIM_S, now).run();
      if (r.meta.changes) return json({ star: s, left: (await stats(env, all.length)).left });
    }
    return json({ error: "every star has been searched" }, 404);
  }

  if (url.pathname === "/api/search/result" && request.method === "POST") {
    let b: Record<string, unknown>;
    try { b = await request.json(); } catch { return json({ error: "bad json" }, 400); }
    const tic = Number(b.tic);
    if (!all.some((s) => s.tic === tic)) return json({ error: "not a star in the visitors' pool" }, 400);
    const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? Math.round(v * 1e5) / 1e5 : null);
    const dip = b.dip && typeof b.dip === "object" ? b.dip as Record<string, unknown> : null;
    const handle = typeof b.handle === "string" ? b.handle.replace(/[^\w@.\- ]/g, "").trim().slice(0, 24) : "";
    const row = {
      tic, pass: b.pass === true, handle, at: new Date().toISOString(),
      dip: dip ? { snr: num(dip.snr), period: num(dip.period), t0: num(dip.t0), dur_h: num(dip.dur_h), depth_ppm: num(dip.depth_ppm), transits: num(dip.transits) } : null,
      top_bump: num(b.topBump), n_freq: num(b.nFreq), seconds: num(b.seconds),
    };
    const r = await env.DB.prepare("INSERT OR IGNORE INTO results (tic, pass, handle, at, row) VALUES (?, ?, ?, ?, ?)")
      .bind(tic, row.pass ? 1 : 0, handle, row.at, JSON.stringify(row)).run();
    const first = r.meta.changes > 0;
    await env.DB.batch([
      ...(first ? [env.DB.prepare("UPDATE totals SET searched = searched + 1, passed = passed + ? WHERE id = 1").bind(row.pass ? 1 : 0)] : []),
      env.DB.prepare("DELETE FROM claims WHERE tic = ?").bind(tic),
    ]);
    return json({ ok: true, first, ...(await stats(env, all.length)) });
  }
  return json({ error: "not found" }, 404);
}

// how many of the pool's stars visitors have searched, and how many cleared the bar
async function stats(env: SearchEnv, total: number) {
  const t = await env.DB.prepare("SELECT searched, passed FROM totals WHERE id = 1").first<{ searched: number; passed: number }>();
  const searched = t?.searched ?? 0, passed = t?.passed ?? 0;
  return { total, searched, passed, left: total - searched };
}
