import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

type TritonDocSnippet = {
  title: string;
  keywords: string[];
  body: string;
};

const DOC_SNIPPETS: TritonDocSnippet[] = [
  {
    title: "Kernel shape",
    keywords: ["jit", "kernel", "wrapper", "launch", "grid"],
    body:
      "Define Triton kernels with @triton.jit and launch them from the public Python wrapper. " +
      "The wrapper should allocate outputs, compute a grid, pass pointer arguments, and return " +
      "the same structure as the PyTorch reference.",
  },
  {
    title: "Program ids and offsets",
    keywords: ["program_id", "tl.program_id", "offsets", "tl.arange", "block"],
    body:
      "Use tl.program_id to identify the current program instance and tl.arange to build " +
      "vector offsets inside a tile. A common 1D pattern is pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).",
  },
  {
    title: "Masked memory access",
    keywords: ["mask", "load", "store", "tl.load", "tl.store", "bounds"],
    body:
      "Use tl.load and tl.store for global memory access. When vector offsets can exceed the " +
      "valid range, pass mask=... to both operations and use other=... on loads when needed.",
  },
  {
    title: "Compile-time constants",
    keywords: ["constexpr", "tl.constexpr", "BLOCK_SIZE", "meta", "parameters"],
    body:
      "Annotate tile sizes and other compile-time meta-parameters as tl.constexpr in @triton.jit " +
      "kernel signatures. Pass values like BLOCK_SIZE=1024 from the wrapper launch.",
  },
  {
    title: "Reductions",
    keywords: ["reduction", "sum", "max", "softmax", "tl.sum", "tl.max"],
    body:
      "For reductions inside a tile, use Triton operations such as tl.sum, tl.max, and " +
      "broadcasted arithmetic over tl.arange offsets. For softmax, subtract the tile max before exp.",
  },
  {
    title: "Matrix multiply",
    keywords: ["matmul", "dot", "tl.dot", "stride", "2d", "matrix"],
    body:
      "Use tl.dot for block matrix multiplication. Build row and column offsets with separate " +
      "tl.arange ranges, load A and B tiles with masks, accumulate in fp32 when appropriate, " +
      "and store the output tile with bounds masks.",
  },
  {
    title: "Avoid PyTorch fallback",
    keywords: ["torch", "fallback", "reference", "unsupported"],
    body:
      "Do not use PyTorch for core computation in the generated implementation. PyTorch is " +
      "acceptable for allocation helpers like torch.empty_like in the public wrapper.",
  },
];

function scoreSnippet(snippet: TritonDocSnippet, queryWords: string[]): number {
  const haystack = `${snippet.title} ${snippet.keywords.join(" ")} ${snippet.body}`.toLowerCase();
  return queryWords.reduce((score, word) => score + (haystack.includes(word) ? 1 : 0), 0);
}

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "search_triton_docs",
    label: "Search Triton Context",
    description:
      "Returns concise local Triton guidance for kernel generation. " +
      "Call this before generate_kernel when the task needs Triton syntax or pattern context.",
    parameters: Type.Object({
      query: Type.String({
        description: "Triton topic or task detail to look up, e.g. masked loads, softmax, tl.dot.",
      }),
      limit: Type.Optional(Type.Number({
        description: "Maximum number of snippets to return. Defaults to 3.",
      })),
    }),
    async execute(_toolCallId, params) {
      const queryWords = params.query
        .toLowerCase()
        .split(/[^a-z0-9_.]+/)
        .filter(Boolean);
      const limit = Math.max(1, Math.min(params.limit ?? 3, DOC_SNIPPETS.length));

      const matches = DOC_SNIPPETS
        .map((snippet) => ({ snippet, score: scoreSnippet(snippet, queryWords) }))
        .filter((match) => match.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, limit)
        .map((match) => match.snippet);

      const snippets = matches.length > 0 ? matches : DOC_SNIPPETS.slice(0, limit);
      const text = snippets
        .map((snippet) => `${snippet.title}\n${snippet.body}`)
        .join("\n\n");

      return {
        content: [{ type: "text", text }],
        details: { query: params.query, snippets },
      };
    },
  });
}
