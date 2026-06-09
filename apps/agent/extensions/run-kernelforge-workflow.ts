import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "run_kernelforge_workflow",
    label: "Run KernelForge Workflow",
    description:
      "Runs the final KernelForge teacher-plan + Gemma/xgrammar workflow for one KAGBench task. " +
      "Returns ledger paths and a visualization-friendly summary row.",
    parameters: Type.Object({
      task_id: Type.Optional(Type.String({ description: "KAGBench task id, e.g. tritonbench_g/vector_addition." })),
      entry_file: Type.Optional(Type.String({ description: "KAGBench entry file, e.g. vector_addition.py." })),
      teacher_model: Type.Optional(Type.String({ description: "Teacher/planner model from llm_models.json." })),
      implementer_model: Type.Optional(Type.String({ description: "Gemma implementer model from llm_models.json." })),
      candidates: Type.Optional(Type.Number({ description: "Candidates per attempt." })),
      max_repairs: Type.Optional(Type.Number({ description: "Maximum public-test repair rounds." })),
      eval_backend: Type.Optional(Type.Union([Type.Literal("none"), Type.Literal("local"), Type.Literal("modal")], {
        description: "Evaluation backend. Use none for generation-only previews, local for local GPU tests, or modal for Modal GPU tests.",
      })),
      out: Type.Optional(Type.String({ description: "Output run directory under runs/agent or an explicit path." })),
      dry_run: Type.Optional(Type.Boolean({ description: "Build prompts and redacted config without calling models." })),
      no_grammar: Type.Optional(Type.Boolean({ description: "Disable xgrammar extra_body for debugging." })),
      grammar_backend: Type.Optional(Type.Union([Type.Literal("xgrammar"), Type.Literal("llama-cpp")], {
        description: "Grammar request format. Use llama-cpp for local llama-server native GBNF.",
      })),
    }),
    async execute(_toolCallId, params, signal) {
      if (!params.task_id && !params.entry_file) {
        return {
          content: [{ type: "text", text: "Provide either task_id or entry_file." }],
          details: { error: "missing task selector" },
        };
      }

      const args = ["run", "python", "-m", "kernelforge.agent", "run"];
      if (params.task_id) args.push("--task-id", params.task_id);
      if (params.entry_file) args.push("--entry-file", params.entry_file);
      if (params.teacher_model) args.push("--teacher-model", params.teacher_model);
      if (params.implementer_model) args.push("--implementer-model", params.implementer_model);
      if (params.candidates !== undefined) args.push("--candidates", String(params.candidates));
      if (params.max_repairs !== undefined) args.push("--max-repairs", String(params.max_repairs));
      if (params.eval_backend) args.push("--eval-backend", params.eval_backend);
      if (params.out) args.push("--out", params.out);
      if (params.dry_run) args.push("--dry-run");
      if (params.no_grammar) args.push("--no-grammar");
      if (params.grammar_backend) args.push("--grammar-backend", params.grammar_backend);

      const result = await pi.exec("uv", args, { signal });
      if (result.code !== 0) {
        return {
          content: [{ type: "text", text: `KernelForge workflow failed:\n${result.stderr || result.stdout}` }],
          details: { error: result.stderr || result.stdout },
        };
      }

      let parsed: unknown;
      try {
        parsed = JSON.parse(result.stdout.trim());
      } catch {
        return {
          content: [{ type: "text", text: result.stdout }],
          details: { raw_stdout: result.stdout },
        };
      }

      const data = parsed as any;
      if (params.dry_run) {
        return {
          content: [{ type: "text", text: `Dry run built prompts for ${data.tasks?.length ?? 0} task(s).` }],
          details: data,
        };
      }

      const summaries = data.summaries ?? [];
      const lines = [
        `run_id: ${data.run_id}`,
        `out_dir: ${data.out_dir}`,
        `tasks: ${summaries.length}`,
        ...summaries.map((row: any) =>
          `${row.task_id}: ${row.status} | public=${row.public_passed} | hidden=${row.hidden_passed} | final=${row.final_candidate_id ?? "none"}`,
        ),
      ];

      return {
        content: [{ type: "text", text: lines.join("\n") }],
        details: data,
      };
    },
  });
}
