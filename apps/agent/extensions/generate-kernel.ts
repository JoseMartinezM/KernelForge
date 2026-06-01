import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "generate_kernel",
    label: "Generate Triton Kernel",
    description:
      "Generates a Triton kernel for a specific TritonBench task using a configured model. " +
      "Returns the generated Python code. Call validate_kernel after this to check correctness.",
    parameters: Type.Object({
      entry_file: Type.String({
        description: "TritonBench task filename, e.g. tanh.py or softmax.py.",
      }),
      model: Type.String({
        description:
          "Model name from llm_models.json. Options: google/gemma-4-E4B-it, " +
          "lightning-ai/deepseek-v4-pro, openai/gpt-5.4-2026-03-05, lightning-ai/gemma-4-31B-it.",
      }),
    }),
    async execute(toolCallId, params, signal) {
      const result = await pi.exec(
        "uv",
        [
          "run", "python",
          "scripts/generate_kernel.py",
          "--entry-file", params.entry_file,
          "--model", params.model,
        ],
        { signal },
      );

      if (result.code !== 0) {
        return {
          content: [{ type: "text", text: `Generation failed:\n${result.stderr}` }],
          details: { error: result.stderr },
        };
      }

      const jsonLine = result.stdout
        .trim()
        .split("\n")
        .reverse()
        .find((line) => line.startsWith("{"));

      if (!jsonLine) {
        return {
          content: [{ type: "text", text: `No JSON result:\n${result.stdout}` }],
          details: { error: "no json result" },
        };
      }

      const genResult = JSON.parse(jsonLine);

      if (genResult.error) {
        return {
          content: [{ type: "text", text: `Error: ${genResult.error}` }],
          details: genResult,
        };
      }

      const summary =
        `entry_file: ${genResult.entry_file}\n` +
        `model: ${genResult.model}\n` +
        `finish_reason: ${genResult.finish_reason}\n` +
        `tokens used: ${genResult.usage?.completion_tokens ?? "unknown"}\n\n` +
        `generated code:\n${genResult.content}`;

      return {
        content: [{ type: "text", text: summary }],
        details: genResult,
      };
    },
  });
}
