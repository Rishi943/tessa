// Smoke test: can each Meta model call a tool and use its result? Prints tokens and cost.
// Run: node agent/smoke.js [model ...]   (defaults to Spark 1.3, contributor mode)
import { Type, complete } from "@earendil-works/pi-ai";
import { MODELS, apiKey } from "./models.js";

const tools = [{
  name: "odd_even",
  description: "Compare the transit depth of odd and even transits of a star's signal.",
  parameters: Type.Object({ tic: Type.Integer({ description: "TESS Input Catalog ID" }) }),
}];

function runTool(call) {
  if (call.name !== "odd_even") return { text: `unknown tool ${call.name}`, isError: true };
  return { text: JSON.stringify({ tic: call.arguments.tic, depth_odd_ppm: 667, depth_even_ppm: 594, diff_sigma: 1.1 }), isError: false };
}

async function smoke(id) {
  const model = MODELS[id];
  const context = {
    systemPrompt: "You vet transit signals. Use tools for every number; never invent values.",
    messages: [{ role: "user", content: "Do the odd and even transits of TIC 458191673 agree? Answer in one sentence.", timestamp: Date.now() }],
    tools,
  };
  const t0 = Date.now();
  let cost = 0, input = 0, output = 0;
  for (let turn = 0; turn < 4; turn++) {
    const msg = await complete(model, context, { apiKey: apiKey(id) });
    context.messages.push(msg);
    cost += msg.usage.cost.total; input += msg.usage.input; output += msg.usage.output;
    if (msg.stopReason === "error") return console.log(`${id}: ERROR ${msg.errorMessage}`);
    const calls = msg.content.filter((b) => b.type === "toolCall");
    if (!calls.length) {
      const text = msg.content.filter((b) => b.type === "text").map((b) => b.text).join("").trim();
      return console.log(`${id}: ${turn} tool turn(s), ${input} in / ${output} out tokens, $${cost.toFixed(5)}, ${Date.now() - t0} ms\n  ${text}`);
    }
    for (const call of calls) {
      const r = runTool(call);
      context.messages.push({ role: "toolResult", toolCallId: call.id, toolName: call.name,
        content: [{ type: "text", text: r.text }], isError: r.isError, timestamp: Date.now() });
    }
  }
  console.log(`${id}: no final answer after 4 turns`);
}

for (const id of process.argv.slice(2).length ? process.argv.slice(2) : ["muse-spark-1.3-contributor"]) {
  await smoke(id);
}
