// /api/glimmer: the fine-tuned Glimmer vetter on Modal, for the repo's demo (tools/demo.py), so a judge needs no key.
// The Worker holds the Modal URL and key as secrets and forwards OpenAI-style chat completions for one model only:
//   npx wrangler secret put GLIMMER_VETTER_URL   (the vetter endpoint `modal deploy serve/glimmer.py` printed)
//   npx wrangler secret put GLIMMER_API_KEY      (the Modal secret glimmer-key)
// GET  /api/glimmer/health                 asks vLLM's /health with a short timeout: {ready}. A cold container takes
//                                          3-5 minutes to start; the call that finds it asleep is what wakes it.
// POST /api/glimmer/v1/chat/completions    one model turn (a dossier takes up to six: five tool calls and the verdict)

interface RateLimit { limit(o: { key: string }): Promise<{ success: boolean }> }
export interface GlimmerEnv {
  GLIMMER_VETTER_URL?: string;
  GLIMMER_API_KEY?: string;
  GLIMMER_LIMIT: RateLimit;   // per visitor
  GLIMMER_GLOBAL: RateLimit;  // everyone together, to keep the GPU bill bounded
}

const MODEL = "glimmer-vetter-v10";
const MAX_BODY = 200_000;  // a dossier's messages with five tool answers are ~40 kB
const MAX_TOKENS = 16384;

const json = (v: unknown, status = 200) =>
  new Response(JSON.stringify(v), { status, headers: { "content-type": "application/json" } });

export async function glimmer(request: Request, env: GlimmerEnv): Promise<Response> {
  if (!env.GLIMMER_VETTER_URL || !env.GLIMMER_API_KEY) return json({ error: "Glimmer is not configured on this deployment" }, 503);
  const base = env.GLIMMER_VETTER_URL.replace(/\/$/, "");
  const path = new URL(request.url).pathname;

  if (path === "/api/glimmer/health" && request.method === "GET") {
    try {
      const r = await fetch(`${base}/health`, { signal: AbortSignal.timeout(8000) });
      return json({ ready: r.ok });
    } catch {
      return json({ ready: false });  // asleep or starting: this request has woken it
    }
  }

  if (path !== "/api/glimmer/v1/chat/completions" || request.method !== "POST") return json({ error: "not found" }, 404);
  const ip = request.headers.get("cf-connecting-ip") ?? "anon";
  const [mine, all] = await Promise.all([env.GLIMMER_LIMIT.limit({ key: ip }), env.GLIMMER_GLOBAL.limit({ key: "all" })]);
  if (!mine.success || !all.success) return json({ error: "slow down" }, 429);

  const text = await request.text();
  if (text.length > MAX_BODY) return json({ error: "request too large" }, 413);
  let body: Record<string, unknown>;
  try {
    body = JSON.parse(text);
  } catch {
    return json({ error: "bad json" }, 400);
  }
  if (body.model !== MODEL) return json({ error: `only ${MODEL} is served here` }, 400);
  if (body.stream) return json({ error: "streaming is not supported here" }, 400);
  body.max_tokens = Math.min(Number(body.max_tokens) || MAX_TOKENS, MAX_TOKENS);

  // Modal answers a request still running after 150 s with a 303 to the same URL; fetch follows it
  const res = await fetch(`${base}/v1/chat/completions`, {
    method: "POST",
    headers: { authorization: `Bearer ${env.GLIMMER_API_KEY}`, "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  return new Response(res.body, { status: res.status, headers: { "content-type": res.headers.get("content-type") ?? "application/json" } });
}
