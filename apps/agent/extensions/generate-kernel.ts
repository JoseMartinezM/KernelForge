import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

function grammarBackendParameter(description: string) {
  return Type.Optional(Type.Union([
    Type.Literal("xgrammar"),
    Type.Literal("llama-cpp"),
  ], { description }));
}

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
          "Model name from llm_models.json. Options: google/gemma-4-31b-it, " +
          "google/gemma-4-26b-it, google/gemma-4-E4B-it, " +
          "lightning-ai/deepseek-v4-pro, openai/gpt-5.4-2026-03-05, lightning-ai/gemma-4-31B-it.",
      }),
      use_grammar: Type.Optional(Type.Boolean({
        description:
          "When true, use constrained generation with grammar/triton.gbnf unless grammar_file is set.",
      })),
      grammar_file: Type.Optional(Type.String({
        description:
          "Optional GBNF grammar path. Defaults to grammar/triton.gbnf when grammar is enabled.",
      })),
      grammar_backend: grammarBackendParameter(
        "Grammar backend to use with grammar_file. Options: xgrammar, llama-cpp. Defaults to xgrammar.",
      ),
      guided_decoding_backend: grammarBackendParameter(
        "Deprecated alias for grammar_backend. Defaults to xgrammar.",
      ),
    }),
    async execute(toolCallId, params, signal) {
      const args = [
        "run", "python",
        "scripts/generate_kernel.py",
        "--entry-file", params.entry_file,
        "--model", params.model,
      ];

      const shouldUseGrammar =
        params.use_grammar === true ||
        params.grammar_file !== undefined ||
        params.grammar_backend !== undefined ||
        params.guided_decoding_backend !== undefined;

      if (shouldUseGrammar) {
        const grammarBackend =
          params.grammar_backend ?? params.guided_decoding_backend ?? "xgrammar";
        args.push("--grammar-file", params.grammar_file ?? "grammar/triton.gbnf");
        args.push("--grammar-backend", grammarBackend);
      }

      const result = await pi.exec(
        "uv",
        args,
        { signal },
      );

      const jsonLine = result.stdout
        .trim()
        .split("\n")
        .reverse()
        .find((line) => line.trim().startsWith("{"));

      let genResult: any | undefined;
      if (jsonLine) {
        try {
          genResult = JSON.parse(jsonLine);
        } catch {
          return {
            content: [{ type: "text", text: `Invalid JSON result:\n${jsonLine}` }],
            details: {
              error: "invalid json result",
              stdout: result.stdout,
              stderr: result.stderr,
              code: result.code,
            },
          };
        }
      }

      if (genResult?.error) {
        return {
          content: [{ type: "text", text: `Error: ${genResult.error}` }],
          details: genResult,
        };
      }

      if (result.code !== 0) {
        const output = [result.stderr, result.stdout]
          .map((text) => text.trim())
          .filter(Boolean)
          .join("\n");

        return {
          content: [{
            type: "text",
            text: `Generation failed:\n${output || `process exited with code ${result.code}`}`,
          }],
          details: {
            error: output,
            stdout: result.stdout,
            stderr: result.stderr,
            code: result.code,
          },
        };
      }

      if (!genResult) {
        return {
          content: [{ type: "text", text: `No JSON result:\n${result.stdout}` }],
          details: {
            error: "no json result",
            stdout: result.stdout,
            stderr: result.stderr,
            code: result.code,
          },
        };
      }

      const grammarSummary = genResult.grammar
        ? `${genResult.grammar.source_path} (` +
          `${genResult.grammar.backend ?? genResult.grammar.guided_decoding_backend})`
        : "none";
      const summary =
        `entry_file: ${genResult.entry_file}\n` +
        `model: ${genResult.model}\n` +
        `grammar: ${grammarSummary}\n` +
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
