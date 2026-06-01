import { appendFileSync, mkdirSync } from "fs";
import { join } from "path";

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "write_ledger",
    label: "Write Agent Ledger",
    description:
      "Saves the results of a generate + validate cycle to runs/agent/{model}.jsonl. " +
      "Call this after validate_kernel to persist the result. " +
      "Each call appends one row — the file grows across runs.",
    parameters: Type.Object({
      entry_file: Type.String({ description: "TritonBench task filename, e.g. tanh.py" }),
      model: Type.String({ description: "Model name used for generation, e.g. google/gemma-4-E4B-it" }),
      content: Type.String({ description: "The generated Triton kernel code" }),
      finish_reason: Type.String({ description: "LLM finish reason: stop, length, etc." }),
      call_at_1: Type.Boolean({ description: "Whether the kernel ran without errors" }),
      exe_at_1: Type.Union([Type.Boolean(), Type.Null()], {
        description: "Whether the kernel outputs matched the reference",
      }),
      mismatches: Type.Array(Type.String(), { description: "List of mismatch descriptions" }),
      semantic_warnings: Type.Array(Type.String(), { description: "Semantic checker warnings" }),
      attempts: Type.Number({ description: "How many generation attempts the agent made" }),
      prompt_tokens: Type.Union([Type.Number(), Type.Null()], { description: "Input tokens used" }),
      completion_tokens: Type.Union([Type.Number(), Type.Null()], { description: "Output tokens used" }),
    }),
    async execute(_toolCallId, params) {
      const slug = params.model.toLowerCase().replace(/\//g, "-").replace(/[^a-z0-9-]/g, "-");
      const outDir = join(process.cwd(), "runs", "agent");
      const outFile = join(outDir, `${slug}.jsonl`);

      mkdirSync(outDir, { recursive: true });

      const row = {
        timestamp: new Date().toISOString(),
        entry_file: params.entry_file,
        model: params.model,
        content: params.content,
        finish_reason: params.finish_reason,
        "call@1": params.call_at_1,
        "exe@1": params.exe_at_1,
        mismatches: params.mismatches,
        semantic_warnings: params.semantic_warnings,
        execution_time: null,
        attempts: params.attempts,
        usage: {
          prompt_tokens: params.prompt_tokens,
          completion_tokens: params.completion_tokens,
        },
      };

      appendFileSync(outFile, JSON.stringify(row) + "\n", "utf-8");

      return {
        content: [{ type: "text", text: `Saved to ${outFile}\nentry: ${params.entry_file} | model: ${params.model} | call@1: ${params.call_at_1} | exe@1: ${params.exe_at_1} | attempts: ${params.attempts}` }],
        details: { file: outFile, row },
      };
    },
  });
}
