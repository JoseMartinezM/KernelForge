import { unlinkSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "validate_kernel",
    label: "Validate Kernel on GPU",
    description:
      "Validates a generated Triton kernel against the TritonBench reference on a Modal T4 GPU. " +
      "Returns call@1 (kernel ran without errors), exe@1 (outputs matched reference), and mismatch details. " +
      "Use this after generate_kernel to check whether the generated code is correct.",
    parameters: Type.Object({
      kernel_code: Type.String({
        description: "The generated Triton kernel Python code to validate.",
      }),
      entry_file: Type.String({
        description: "TritonBench task filename to validate against, e.g. tanh.py or softmax.py.",
      }),
    }),
    async execute(toolCallId, params, signal) {
      const tmpFile = join(tmpdir(), `kernelforge_${toolCallId}.py`);

      try {
        writeFileSync(tmpFile, params.kernel_code, "utf-8");

        const result = await pi.exec(
          "uv",
          [
            "run", "modal", "run",
            "scripts/modal_eval.py",
            "--entry-file", params.entry_file,
            "--kernel-file", tmpFile,
          ],
          { signal },
        );

        if (result.code !== 0) {
          return {
            content: [{ type: "text", text: `Modal eval failed:\n${result.stderr}` }],
            details: { error: result.stderr },
          };
        }

        // Modal prints progress lines before the JSON result — grab the last JSON line
        const jsonLine = result.stdout
          .trim()
          .split("\n")
          .reverse()
          .find((line) => line.startsWith("{"));

        if (!jsonLine) {
          return {
            content: [{ type: "text", text: `No JSON result in output:\n${result.stdout}` }],
            details: { error: "no json result" },
          };
        }

        const evalResult = JSON.parse(jsonLine);

        const lines: string[] = [
          `entry_file: ${evalResult.file}`,
          `call@1: ${evalResult["call@1"]}`,
          `exe@1: ${evalResult["exe@1"]}`,
          `mismatches: ${evalResult.mismatches?.length ? evalResult.mismatches.join(", ") : "none"}`,
        ];

        // Include stderr so the agent can read the exact error and fix the kernel
        if (evalResult.pred?.stderr) {
          lines.push(`\npred stderr:\n${evalResult.pred.stderr}`);
        }
        if (evalResult.ref?.stderr) {
          lines.push(`\nref stderr:\n${evalResult.ref.stderr}`);
        }

        return {
          content: [{ type: "text", text: lines.join("\n") }],
          details: evalResult,
        };
      } finally {
        try {
          unlinkSync(tmpFile);
        } catch {
          // temp file already gone, nothing to do
        }
      }
    },
  });
}
