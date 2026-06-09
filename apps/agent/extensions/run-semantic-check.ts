import { unlinkSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const SemanticCheckParameters = Type.Object({
  kernel_code: Type.String({
    description: "The generated Triton kernel Python code to check.",
  }),
});

type SemanticCheckDetails = {
  error: string | null;
  warnings: string[];
};

export default function (pi: ExtensionAPI) {
  pi.registerTool<typeof SemanticCheckParameters, SemanticCheckDetails>({
    name: "run_semantic_check",
    label: "Run Semantic Check",
    description:
      "Runs a fast CPU-only semantic check on a Triton kernel before sending it to GPU validation. " +
      "Detects common Triton anti-patterns: missing @triton.jit, missing mask on vectorized tl.load/tl.store, " +
      "missing tl.program_id, and BLOCK_SIZE used without tl.constexpr. " +
      "Call this after generate_kernel and before validate_kernel to catch obvious errors cheaply.",
    parameters: SemanticCheckParameters,
    async execute(toolCallId, params, signal) {
      const tmpFile = join(tmpdir(), `kernelforge_check_${toolCallId}.py`);

      try {
        writeFileSync(tmpFile, params.kernel_code, "utf-8");

        const result = await pi.exec(
          "uv",
          ["run", "python", "scripts/run_semantic_check.py", "--kernel-file", tmpFile],
          { signal },
        );

        if (result.code !== 0) {
          return {
            content: [{ type: "text", text: `Semantic check failed to run:\n${result.stderr}` }],
            details: { error: result.stderr, warnings: [] },
          };
        }

        const parsed = JSON.parse(result.stdout.trim());
        const warnings: string[] = parsed.warnings ?? [];

        const summary =
          warnings.length === 0
            ? "No semantic warnings — kernel looks clean, safe to send to GPU."
            : `${warnings.length} warning(s) found — fix before sending to GPU:\n  - ${warnings.join("\n  - ")}`;

        return {
          content: [{ type: "text", text: summary }],
          details: { error: null, warnings },
        };
      } finally {
        try {
          unlinkSync(tmpFile);
        } catch {
          // temp file already gone
        }
      }
    },
  });
}
