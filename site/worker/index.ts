// tessa.braveai.solutions: the static site comes straight from the assets; this Worker only answers
// /api/ask, where Muse Spark answers questions about the survey with tools over the exported files, and
// /api/search/*, which hands visitors a star to search in their browser and records what they found.
import { search } from "./search";
import type { SearchEnv } from "./search";
import { SYSTEM, TOOLS, runTool, stepText } from "./tools";

interface RateLimit { limit(o: { key: string }): Promise<{ success: boolean }> }
export interface Env extends SearchEnv {
  ASSETS: Fetcher;
  META_API_KEY: string;
  ASK_LIMIT: RateLimit;  // per visitor
  ASK_GLOBAL: RateLimit;  // everyone together, to keep the credit safe
}

const META_URL = "https://api.meta.ai/v1/chat/completions";
const MODEL = "muse-spark-1.3-contributor";
const MAX_ROUNDS = 6;  // model calls per question
const MAX_QUESTION = 1200;

type Msg = { role: string; content: string | null; tool_calls?: ToolCall[]; tool_call_id?: string };
type ToolCall = { id: string; type: "function"; function: { name: string; arguments: string } };

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/api/ask" && request.method === "POST") return ask(request, env, ctx);
    if (url.pathname.startsWith("/api/search/")) return search(request, env);
    if (url.pathname.startsWith("/api/")) return new Response("not found", { status: 404 });
    return env.ASSETS.fetch(request);
  },
};

async function ask(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
  const ip = request.headers.get("cf-connecting-ip") ?? "anon";
  const [mine, all] = await Promise.all([env.ASK_LIMIT.limit({ key: ip }), env.ASK_GLOBAL.limit({ key: "all" })]);
  if (!mine.success || !all.success) return new Response("slow down", { status: 429 });

  let body: { messages?: { role: string; content: string }[]; page?: string };
  try {
    body = await request.json();
  } catch {
    return new Response("bad json", { status: 400 });
  }
  const turns = (body.messages ?? []).slice(-8)
    .filter((m) => (m.role === "user" || m.role === "assistant") && typeof m.content === "string")
    .map((m) => ({ role: m.role, content: m.content.slice(0, MAX_QUESTION * 2) }));
  if (!turns.length || turns[turns.length - 1].role !== "user" || turns[turns.length - 1].content.length > MAX_QUESTION) {
    return new Response("bad question", { status: 400 });
  }
  const page = typeof body.page === "string" && /^\/[\w/#-]{0,40}$/.test(body.page) ? body.page : "/";

  const { readable, writable } = new TransformStream();
  const out = writable.getWriter();
  const enc = new TextEncoder();
  const emit = (type: string, text: string) => out.write(enc.encode(JSON.stringify({ type, text }) + "\n"));

  ctx.waitUntil((async () => {
    try {
      const messages: Msg[] = [{ role: "system", content: `${SYSTEM}\n\nThe reader is on the page ${page}.` }, ...turns];
      for (let round = 0; round < MAX_ROUNDS; round++) {
        const last = round === MAX_ROUNDS - 1;
        const res = await fetch(META_URL, {
          method: "POST",
          headers: { authorization: `Bearer ${env.META_API_KEY}`, "content-type": "application/json" },
          body: JSON.stringify({ model: MODEL, messages, tools: last ? undefined : TOOLS, temperature: 0.3, max_tokens: 4096 }),
        });
        if (!res.ok) {
          console.log("meta api", res.status, (await res.text()).slice(0, 300));
          await emit("error", "The model is not answering right now. Try again in a minute.");
          return;
        }
        const data = (await res.json()) as { choices: { message: Msg }[] };
        const msg = data.choices?.[0]?.message;
        if (!msg) throw new Error("empty reply");
        if (msg.tool_calls?.length) {
          messages.push({ role: "assistant", content: msg.content ?? null, tool_calls: msg.tool_calls });
          for (const call of msg.tool_calls) {
            let args: Record<string, unknown> = {};
            try { args = JSON.parse(call.function.arguments || "{}"); } catch { /* the tool reports the bad call */ }
            await emit("step", stepText(call.function.name, args));
            const result = await runTool(call.function.name, args, (path) => env.ASSETS.fetch(new Request(new URL(path, request.url))));
            if (call.function.name === "show" && result.includes('"shown":true')) {
              await out.write(enc.encode(JSON.stringify({ type: "show", tic: Math.trunc(Number(args.tic)), view: args.view }) + "\n"));
            }
            messages.push({ role: "tool", tool_call_id: call.id, content: result });
          }
          continue;
        }
        await emit("answer", (msg.content ?? "").trim() || "I could not find an answer to that in Tessa's files.");
        return;
      }
      await emit("error", "That took too many steps. Try a narrower question.");
    } catch (e) {
      console.log("ask failed", String(e));
      await emit("error", "Something went wrong while answering. Try again.");
    } finally {
      await out.close();
    }
  })());

  return new Response(readable, { headers: { "content-type": "application/x-ndjson; charset=utf-8", "cache-control": "no-store" } });
}
