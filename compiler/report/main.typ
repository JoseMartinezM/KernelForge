#import "lib.typ": report-template
#import "@preview/finite:0.5.1": automaton
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

#set document(title: "Triton GPU Kernel Lexical Analyzer and Parser")
#set heading(numbering: "1.")
#set par(justify: true, spacing: 0.65em)
#set figure(numbering: "1")

#let tec-blue = rgb("#162773")
#let ink = rgb("#252a31")
#let muted = rgb("#5f6770")
#let soft-blue = rgb("#eef3fb")
#let rule = rgb("#cfd8e3")

#show heading.where(level: 1): it => {
  pagebreak(weak: true)
  block(above: 0.4em, below: 0.65em)[
    #text(fill: tec-blue, weight: "bold")[#it]
    #v(-0.35em)
    #line(length: 100%, stroke: 0.7pt + tec-blue)
  ]
}

#show heading.where(level: 2): it => {
  block(above: 0.85em, below: 0.35em)[#text(fill: ink, weight: "bold")[#it]]
}

#show raw.where(block: true): it => block(
  width: 100%,
  inset: 7pt,
  radius: 3pt,
  fill: rgb("#f6f8fb"),
  stroke: 0.45pt + rule,
)[#text(size: 8.6pt)[#it]]

#let artifact(title, body) = block(
  width: 100%,
  inset: 8pt,
  radius: 4pt,
  fill: soft-blue,
  stroke: 0.6pt + rule,
)[
  #text(size: 9pt, weight: "bold", fill: tec-blue)[#title]
  #v(4pt)
  #body
]

#let section-note(body) = block(
  width: 100%,
  inset: 7pt,
  radius: 3pt,
  fill: rgb("#fbfcff"),
  stroke: (left: 2pt + tec-blue, rest: 0.4pt + rule),
)[#text(size: 9.4pt)[#body]]

#let mini-source(title, body) = figure(
  artifact(title, body),
  kind: image,
  caption: title,
)

#show: report-template.with(
  title: "Triton GPU Kernel Lexical Analyzer and Parser",
  subtitle: "Flex/Bison compiler front end",
  authors: (
    "Imanol Armando González Solís",
    "José Manuel Martínez Morales",
    "Pablo Esteban Reyes Herrera",
    "Sebastian Gerritsen Ortiz",
    "Victor Javier Quintana Cisneros",
  ),
  abstract: [
    The project implements and validates a Flex/Bison scanner and parser for a reduced Triton GPU kernel language. The compiler recognizes top-level `@triton.jit` kernel functions, emits a token sequence, builds symbol tables, validates the supported syntax, and reports lexical, syntax, and semantic errors. The implementation uses Flex and Bison directly; it does not use Python's `ast` module or native regular-expression libraries.
  ],
)

#set table(
  fill: (x, y) => if y == 0 { tec-blue } else if calc.odd(y) { rgb("#f7f9fc") } else { none },
  stroke: 0.5pt + rule,
)

#show table.cell.where(y: 0): set text(fill: white, weight: "bold")
#show figure.caption: set text(size: 9pt, fill: muted)

= Introduction <introduction>

== Executive summary <executive-summary>

The front end is implemented as a Flex/Bison toolchain for a reduced Triton GPU-kernel language and provides two executables:

- `triton_scanner`, a standalone Flex scanner that prints the token stream for Triton JIT blocks and then prints the scanner symbol table.
- `triton_compiler`, a Flex/Bison validator that parses the same token stream and prints a structured report of accepted kernels.

The implementation accepts realistic Python/Triton source files while formally processing only top-level `@triton.jit` functions. Host-side Python code is skipped by the scanner, so imports, launch wrappers, allocation code, and tests can remain in the input file without requiring the project to become a full Python compiler.

The parser recognizes representative Triton kernels such as vector addition, softmax, dropout, matrix multiplication, flash attention, and layer normalization fixtures. On successful input it reports kernel names, formal parameters, `tl.constexpr` parameters, local identifiers, and called `tl.*` APIs. On failure it exits nonzero and reports lexical, syntax, semantic, or missing-kernel diagnostics.

== Notation and development model <notation-and-development-model>

The lexical specification is written as regular expressions in Flex. Each expression denotes the set of lexemes accepted for a token class. Flex compiles these expressions into deterministic finite automata and applies longest-match selection before running the associated action.

The syntax specification is written as context-free productions in Bison. Bison builds an LALR parser from those productions. Scanner actions pass tokens and lexeme text to the parser; parser actions update the kernel report directly. The design deliberately avoids Python's `ast` module and native regular-expression libraries.

The work followed an iterative waterfall model. The stable language boundary was first captured as a reduced Triton kernel subset; the scanner was then implemented and tested; parser coverage was added over the validated token stream; and regression tests were expanded as real fixtures exposed missing syntax. This model fit the project scope because the high-level language boundary was known early, while the exact grammar details benefited from short validation cycles against concrete Triton examples.

Flex and Bison are used as the implementation tools because they map directly to the front-end responsibilities: regular-language recognition for lexemes, context-free validation for statements and expressions, and deterministic command-line behavior suitable for automated tests.

== Scope and assumptions <scope-and-assumptions>

Triton kernels are Python functions decorated with `@triton.jit`. A complete source file commonly contains Python host code around those kernels. The front end therefore distinguishes between the input file and the formal language it validates:

- Outside top-level JIT blocks, the scanner remains in host mode and skips text.
- At a top-level `@triton.jit` decorator, the scanner enters JIT mode and emits tokens.
- After the JIT function dedents back to column zero, the scanner returns to host mode.
- Indented `@triton.jit` decorators are reported as out-of-scope nested JIT blocks.

The supported kernel subset includes decorators, function definitions, parameters, `tl.constexpr` annotations, assignments, augmented assignments, returns, assertions, `if`/`elif`/`else`, `for`, `while`, `with`, function calls, attributes, indexing, slicing, numeric literals, string literals, and Python-like layout tokens. Constructs unrelated to the target kernel subset, such as class declarations, exception handling, lambda expressions, comprehensions, and asynchronous functions, are intentionally not modeled.

== IEEE 830 and assignment-deliverable alignment <ieee-830-alignment>

The evaluation presentation requires the report to follow an IEEE 830-style development structure and to include the scanner deliverables listed in @tab-deliverables. The report therefore separates requirements and lexical definition in Analysis, construction details in Design and Implementation, and objective evidence in Verification and Validation.

#figure(
  table(
    columns: (2fr, 2fr),
    inset: 5pt,
    align: left,
    [*Required deliverable*], [*Where it is covered*],
    [Informal description of required lexemes], [Section @lexical-inventory-and-omissions],
    [Regular expression for each token kind], [@tab-regex-inventory],
    [Automata that recognize required lexemes], [@fig-host-jit-automaton, @fig-identifier-dfa, and @fig-number-dfa],
    [Token IDs], [@tab-token-ids],
    [Transition table], [@tab-transition-table],
    [Symbol table design], [@tab-data-model],
    [Scanner implemented using UNIX `lex`], [Section @source-modules and @sec-lex-printout],
    [Scanner output examples], [Section @execution-snapshots],
  ),
  kind: table,
  caption: [Assignment deliverables from the evaluation presentation and report coverage.],
) <tab-deliverables>

= Analysis <analysis>

== Language boundary <language-boundary>

The formal input recognized by the parser is one or more top-level Triton JIT blocks:

#mini-source([Accepted JIT block shape], [
```text
@triton.jit
def kernel_name(parameters):
    statements
```
]) <fig-jit-block-shape>

The scanner treats the surrounding file as transport context, not as part of the validated language. This is the reason the implementation can run against practical fixtures that include imports and test harness code while still keeping the compiler front end small, auditable, and focused on GPU kernels.

== Delivered capability traceability <delivered-capability-traceability>

#section-note[
  Delivered capabilities are linked to their implementation artifacts and validation evidence to support auditability and verification.
]

#figure(
  table(
    columns: (auto, 1.2fr, 1.45fr, 1.2fr),
    inset: 5pt,
    align: left,
    [*Trace*], [*Delivered capability*], [*Implementation path*], [*Validation evidence*],
    [C1], [Input can be read from a source file or standard input.], [`main` in `triton.l` and `triton.y` opens `argv[1]` when supplied and otherwise reads from `yyin`.], [`test_c_compiler.py` runs binaries on fixture files; manual runs also cover command-line input.],
    [C2], [Triton JIT lexemes are recognized without parsing unrelated host Python.], [`INITIAL` and `JIT` start conditions in `triton.l`; top-level `@triton.jit` switches modes.], [Valid fixtures contain imports and comments before kernels; scanner output begins at the decorator.],
    [C3], [The scanner can emit a human-readable token sequence.], [`emit_token` and `queue_pop` print token name, lexeme, and source line in standalone mode.], [`triton_scanner vector_add.py` snapshot in the verification section.],
    [C4], [The scanner records unique lexical entries.], [`add_symbol` stores `(token, lexeme, first line)` in a fixed table and prints it at end of scan.], [`TestScanner` checks concrete tokens, numeric forms, string forms, and INDENT/DEDENT balance.],
    [C5], [Lexical, syntax, semantic, and missing-kernel failures are visible to users.], [Lexical counters in `triton.l`; `yyerror` and semantic duplicate checks in `triton.y`; nonzero exit in parser `main`.], [Invalid fixtures: `bad_syntax.py`, `duplicate_kernel.py`, and `missing_jit.py`.],
    [C6], [The parser validates the supported kernel structure and expression subset.], [Bison grammar in `triton.y` for decorators, signatures, suites, statements, calls, attributes, indexing, slicing, and expression precedence.], [Valid fixtures and end-to-end generated kernels exercise common syntax families.],
    [C7], [Successful parsing produces a compact technical kernel report.], [`KernelInfo`, `ParameterInfo`, `add_parameter`, `add_local`, `add_tl_call`, and `print_report` in `triton.y`.], [Reports for `vector_add.py` and `layer_norm.py` list kernels, parameters, locals, and Triton calls.],
  ),
  kind: table,
  caption: [Delivered capability traceability matrix.],
) <tab-traceability>

== Lexical inventory and omissions <lexical-inventory-and-omissions>

The scanner includes lexemes needed by Triton kernels rather than the whole Python language. Identifiers cover kernel names, variables, parameter names, and qualified names through the separate `DOT` token. Numeric literals cover decimal, binary, hexadecimal, and common floating-point forms. Octal literals are intentionally omitted because they do not appear in the target Triton kernels and would add an unused token family. String literals cover quoted, triple-quoted, raw, byte, Unicode, and formatted prefixes because real kernels may call APIs such as `float("inf")` or pass string-like arguments. Operators cover arithmetic, pointer-offset arithmetic, masks, comparisons, matrix multiplication, and augmented assignment. Delimiters cover function calls, indexing, slicing, dictionaries or brace-delimited forms, annotations, and statement separators. Layout tokens model Python block structure.

Broad Python-only constructs are not given keyword tokens unless they are useful to the accepted kernel subset. For example, `class`, `try`, `except`, `lambda`, `async`, `await`, and `raise` remain outside the modeled grammar. This keeps scanner and parser behavior aligned with the targeted GPU-kernel subset instead of overfitting a partial general Python grammar.

== Formal regular expressions <formal-regular-expressions>

#figure(
  table(
    columns: (1.1fr, 2.2fr, 1.55fr),
    inset: 5pt,
    align: left,
    [*Token or group*], [*Flex expression or literal family*], [*Accepted examples*],
    [`NAME`], [`[A-Za-z_][A-Za-z0-9_]*`], [`add_kernel`, `tl`, `BLOCK_SIZE`],
    [`NUMBER_INT`], [`[0-9]+`], [`0`, `128`],
    [`NUMBER_HEX`], [`0[xX][0-9a-fA-F]+`], [`0xFF`],
    [`NUMBER_BIN`], [`0[bB][01]+`], [`0b1010`],
    [`NUMBER_FLOAT`], [`(([0-9]+\.[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+)`], [`0.0`, `.5`, `1.5e-3`],
    [`STRING`], [`[fFrRbBuU]*"""([^"\\]|\\.|"[^"\\]|""[^"\\])*"""`], [Triple double-quoted string],
    [`STRING`], [`[fFrRbBuU]*'''([^'\\]|\\.|'[^'\\]|''[^'\\])*'''`], [Triple single-quoted string],
    [`STRING`], [`[fFrRbBuU]*"([^"\\\n]|\\.)*"`], [`"inf"`, `f"x"`],
    [`STRING`], [`[fFrRbBuU]*'([^'\\\n]|\\.)*'`], [`'raw'`, `r'raw'`],
    [Keywords], [Recognized after `NAME` by exact comparison], [`def`, `if`, `for`, `return`, `True`, `None`],
    [Compound operators], [`**=`, `//=`, `<<=`, `>>=`, `+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`], [Augmented assignment],
    [Comparisons and multi-character operators], [`==`, `!=`, `<=`, `>=`, `**`, `//`, `<<`, `>>`, `->`, `...`], [Precedence-sensitive expressions],
    [Single-character operators], [`+`, `-`, `*`, `/`, `%`, `&`, `|`, `^`, `~`, `@`, `<`, `>`, `=`], [Arithmetic, masks, assignment],
    [Delimiters], [`(`, `)`, `[`, `]`, `{`, `}`, `:`, `,`, `.`, `;`], [Calls, indexing, suites, attributes],
    [JIT activator], [`@triton[ \t]*\.[ \t]*jit` in host mode], [Switches from host scanning to token emission],
    [Layout], [`\n[ \t]*` plus indentation comparison], [`NEWLINE`, `INDENT`, `DEDENT`],
  ),
  kind: table,
  caption: [Formal lexical inventory and Flex regular-expression families.],
) <tab-regex-inventory>

== Token ID catalogue <token-id-catalogue>

The public token numbers are shared by standalone scanner mode and the Bison parser. The generated `triton.tab.h` assigns the following values:

#figure(
  table(
    columns: (auto, 1fr, auto, 1fr, auto, 1fr, auto, 1fr),
    inset: 4pt,
    align: left,
    [*ID*], [*Token*], [*ID*], [*Token*], [*ID*], [*Token*], [*ID*], [*Token*],
    [258], [`NAME`], [259], [`STRING`], [260], [`NUMBER_INT`], [261], [`NUMBER_FLOAT`],
    [262], [`NUMBER_HEX`], [263], [`NUMBER_BIN`], [264], [`DEF`], [265], [`IF`],
    [266], [`ELIF`], [267], [`ELSE`], [268], [`FOR`], [269], [`IN`],
    [270], [`WHILE`], [271], [`WITH`], [272], [`AS`], [273], [`RETURN`],
    [274], [`ASSERT`], [275], [`PASS`], [276], [`AND`], [277], [`OR`],
    [278], [`NOT`], [279], [`IS`], [280], [`TRUE`], [281], [`FALSE`],
    [282], [`NONE`], [283], [`NEWLINE`], [284], [`INDENT`], [285], [`DEDENT`],
    [286], [`DOUBLESTAREQ`], [287], [`DOUBLESLASHEQ`], [288], [`LSHIFTEQ`], [289], [`RSHIFTEQ`],
    [290], [`PLUSEQ`], [291], [`MINUSEQ`], [292], [`STAREQ`], [293], [`SLASHEQ`],
    [294], [`PERCENTEQ`], [295], [`AMPEQ`], [296], [`PIPEEQ`], [297], [`CARETEQ`],
    [298], [`EQEQ`], [299], [`NOTEQ`], [300], [`LTEQ`], [301], [`GTEQ`],
    [302], [`DOUBLESTAR`], [303], [`DOUBLESLASH`], [304], [`LSHIFT`], [305], [`RSHIFT`],
    [306], [`ARROW`], [307], [`ELLIPSIS`], [308], [`PLUS`], [309], [`MINUS`],
    [310], [`STAR`], [311], [`SLASH`], [312], [`PERCENT`], [313], [`AMP`],
    [314], [`PIPE`], [315], [`CARET`], [316], [`TILDE`], [317], [`AT`],
    [318], [`LT`], [319], [`GT`], [320], [`EQ`], [321], [`LPAREN`],
    [322], [`RPAREN`], [323], [`LBRACKET`], [324], [`RBRACKET`], [325], [`LBRACE`],
    [326], [`RBRACE`], [327], [`COLON`], [328], [`COMMA`], [329], [`DOT`],
    [330], [`SEMI`], [], [], [], [], [], [],
  ),
  kind: table,
  caption: [Public token IDs emitted by the scanner and consumed by the parser.],
) <tab-token-ids>

`YYEOF`, `YYerror`, and `YYUNDEF` remain Bison-internal control tokens. Parser-only precedence symbols (`IFX`, `UPLUS`, `UMINUS`, and `UTILDE`) are not emitted by the scanner.

= Design <design>

== Front-end architecture <front-end-architecture>

#figure(
  diagram(
    node-fill: soft-blue,
    node-stroke: 0.8pt + tec-blue,
    edge-stroke: 0.8pt + tec-blue,
    node((0, 0), [`source.py` / stdin], width: 36mm, height: 9mm, corner-radius: 2pt),
    edge("d", "-|>", [`read input`], label-side: left),
    node((0, 1), align(center)[
      *Flex scanner*\
      `triton.l`\
      HOST/JIT states, regexes, layout queue, scanner symbols
    ], width: 68mm, height: 20mm, corner-radius: 3pt),
    edge("d", "-|>", [`tokens + semantic text`], label-side: left),
    node((0, 2), align(center)[
      *Bison parser*\
      `triton.y`\
      kernel grammar, precedence, semantic checks, report aggregation
    ], width: 68mm, height: 20mm, corner-radius: 3pt),
    edge("d", "-|>", [`emit result`], label-side: left),
    node((0, 3), [token listing or parser report], width: 46mm, height: 9mm, corner-radius: 2pt),
  ),
  caption: [Scanner/parser ownership boundary and data flow.],
) <fig-front-end-architecture>

As shown in @fig-front-end-architecture, the scanner owns character-level recognition and layout handling. The parser owns grammar validation and higher-level reporting. The executables are built from the same scanner source, with `WITH_PARSER` selecting whether tokens are printed immediately or returned to Bison.

== Scanner automata <scanner-automata>

The scanner has a top-level deterministic mode automaton. `HOST` corresponds to Flex `INITIAL`; `JIT` is an exclusive Flex start condition.

#figure(
  automaton(
    (
      HOST: (JIT: [`@triton.jit` at column 0]),
      JIT: (HOST: [top-level dedent]),
    ),
    initial: "HOST",
    final: ("HOST",),
  ),
  caption: [Top-level scanner mode automaton for host skipping and JIT token emission.],
) <fig-host-jit-automaton>

@fig-host-jit-automaton summarizes the `HOST`/`JIT` start-condition behavior. The detailed transition rules are listed in @tab-transition-table.

Representative token DFAs are summarized below. Flex builds equivalent deterministic automata from the regular expressions in the previous section.

#figure(
  automaton(
    (
      I0: (I1: [letter or `_`]),
      I1: (I1: [letter, digit, or `_`]),
    ),
    initial: "I0",
    final: ("I1",),
  ),
  caption: [Identifier and keyword-candidate DFA; accepted lexemes are relabeled as keywords by exact lookup.],
) <fig-identifier-dfa>

#figure(
  diagram(
    node-stroke: 0.8pt + ink,
    edge-stroke: 0.8pt + ink,
    node((-0.9, 0), [Start], stroke: none),
    edge((-0.9, 0), (0, 0), "-|>"),
    node((0, 0), [$N_0$], radius: 6.5mm),
    edge((0, 0), (1.15, 0), "-|>", [`0`]),
    node((1.15, 0), [NZ], radius: 6.5mm, stroke: 1.4pt + ink),
    edge((1.15, 0), (2.45, -0.72), "-|>", [`x/X`], label-pos: 0.42),
    node((2.45, -0.72), [HX], radius: 6.5mm),
    edge((2.45, -0.72), (3.65, -0.72), "-|>", [hex]),
    node((3.65, -0.72), [$H_1$], name: <dfa-h1>, radius: 6.5mm, stroke: 1.4pt + ink),
    edge(<dfa-h1>, "-|>", <dfa-h1>, [hex], bend: 125deg, loop-angle: 90deg),
    edge((1.15, 0), (2.45, 0.72), "-|>", [`b/B`], label-pos: 0.42),
    node((2.45, 0.72), [BX], radius: 6.5mm),
    edge((2.45, 0.72), (3.65, 0.72), "-|>", [`0/1`]),
    node((3.65, 0.72), [$B_1$], name: <dfa-b1>, radius: 6.5mm, stroke: 1.4pt + ink),
    edge(<dfa-b1>, "-|>", <dfa-b1>, [`0/1`], bend: 125deg, loop-angle: 90deg),
  ),
  caption: [Base-prefixed numeric DFA for hexadecimal and binary integer literals.],
) <fig-number-dfa>

#figure(
  table(
    columns: (1.2fr, 2.4fr, 1.7fr),
    inset: 5pt,
    align: left,
    [*Automaton family*], [*Key transitions*], [*Acceptance rule*],
    [Identifier/keyword], [`I0 -- letter/_ --> I1`; `I1 -- letter/digit/_ --> I1`], [`I1` accepts a `NAME` candidate; exact lexeme lookup may relabel it as a keyword.],
    [Base-prefixed integers], [`N0 -- 0 --> NZ`; `NZ -- x/X --> HX -- hex --> H1`; `NZ -- b/B --> BX -- 0/1 --> B1`], [`NZ` accepts `0`; `H1` accepts `NUMBER_HEX`; `B1` accepts `NUMBER_BIN`.],
    [Decimal and float], [`N0 -- digit --> D1`; `D1 -- . --> F1`; `D1/F1 -- e/E --> ES`; `ES -- +/- --> ED`; `ES/ED -- digit --> E1`], [`D1` accepts `NUMBER_INT`; `F1` and `E1` accept `NUMBER_FLOAT`.],
    [Operator trie], [`*`, `/`, `<`, `-`, and `.` branch to the longest valid compound token before accepting.], [Longest-match selection returns tokens such as `DOUBLESTAR`, `DOUBLESTAREQ`, `DOUBLESLASH`, `LSHIFT`, `LTEQ`, `ARROW`, or `ELLIPSIS`.],
  ),
  kind: table,
  caption: [Textual summary of token-recognition automata not fully expanded as separate figures.],
) <tab-dfa-summary>

String recognition is handled by four regular-expression DFAs: single-quoted, double-quoted, triple-single-quoted, and triple-double-quoted. Each allows an optional prefix in `[fFrRbBuU]*`, consumes escaped characters as `\\.`, rejects unescaped physical newlines in single-line strings, and accepts only at the matching closing delimiter.

== Finite transition table <finite-transition-table>

#figure(
  table(
    columns: (auto, 1.45fr, 1.35fr, 1.75fr),
    inset: 5pt,
    align: left,
    [*State*], [*Input class*], [*Next state*], [*Action*],
    [`HOST`], [Top-level `@triton.jit`], [`JIT`], [Reset layout state, re-read decorator so `AT NAME DOT NAME` are emitted.],
    [`HOST`], [Indented `@triton.jit` on whitespace-only line], [`HOST`], [Increment lexical-error count and skip nested declaration.],
    [`HOST`], [Other non-newline text], [`HOST`], [Advance host column and skip.],
    [`HOST`], [Newline], [`HOST`], [Reset host column to zero.],
    [`JIT`], [Identifier regex], [`JIT`], [Emit keyword token when lexeme matches keyword set; otherwise emit `NAME`.],
    [`JIT`], [Number regex], [`JIT`], [Emit one of `NUMBER_INT`, `NUMBER_FLOAT`, `NUMBER_HEX`, or `NUMBER_BIN`.],
    [`JIT`], [String regex], [`JIT`], [Emit `STRING`.],
    [`JIT`], [Operator or delimiter trie], [`JIT`], [Emit longest matching token.],
    [`JIT`], [`(`, `[`, `{`], [`JIT`], [Increment `paren_depth`; emit delimiter token.],
    [`JIT`], [`)`, `]`, `}`], [`JIT`], [Decrement `paren_depth` when positive; emit delimiter token.],
    [`JIT`], [Whitespace or comment], [`JIT`], [Skip unless newline handling applies.],
    [`JIT`], [Newline with `paren_depth > 0`], [`JIT`], [Skip implicit line join.],
    [`JIT`], [Newline with same indentation], [`JIT`], [Queue `NEWLINE`.],
    [`JIT`], [Newline with larger indentation], [`JIT`], [Push indentation width; queue `NEWLINE`, `INDENT`.],
    [`JIT`], [Newline with smaller indentation], [`JIT` or `HOST`], [Pop indentation stack; queue one or more `DEDENT`; return to `HOST` after top-level dedent.],
    [`JIT`], [Unsupported character], [`JIT`], [Emit lexical error and continue scanning.],
    [`JIT`], [EOF], [accept], [Flush pending `NEWLINE` and `DEDENT` tokens before ending.],
  ),
  kind: table,
  caption: [Finite transition table for scanner start conditions, layout handling, and EOF cleanup.],
) <tab-transition-table>

== Parser grammar model <parser-grammar-model>

The Bison grammar validates a sequence of JIT blocks. Each block consists of a decorator, a `def` signature, an optional return annotation, and an indented suite. The statement layer supports simple statements separated by semicolons and compound statements with nested suites. The expression layer models Triton-style arithmetic and masks with Bison precedence declarations.

#figure(
  table(
    columns: (1.15fr, 2.85fr),
    inset: 5pt,
    align: left,
    [*Nonterminal*], [*Production summary*],
    [`program`], [`jit_blocks`],
    [`jit_block`], [`decorator DEF NAME "(" parameters ")" return_annotation ":" suite`],
    [`suite`], [`NEWLINE+ INDENT block DEDENT`],
    [`block`], [`block_item+`],
    [`block_item`], [`NEWLINE | simple_stmt NEWLINE | compound_stmt`],
    [`simple_stmt`], [`expr | assignment | annotated_assignment | return | assert | pass`],
    [`compound`], [`if | for | while | with`],
    [`primary`], [`atom | primary "." NAME | primary "(" arguments ")" | primary "[" slices "]"`],
  ),
  kind: table,
  caption: [Parser grammar summary for the supported Triton JIT subset.],
) <tab-parser-grammar>

Parser actions keep the report lightweight. When a kernel name is accepted, `begin_kernel` creates a `KernelInfo` record. Parameter actions record names and `tl.constexpr` flags. Assignment actions record local identifiers. Call actions record only qualified `tl.*` calls, so the final report emphasizes Triton APIs rather than every helper function.

== Processing pseudocode <processing-pseudocode>

#figure(
  table(
    columns: (auto, 1.25fr, 2.7fr),
    inset: 5pt,
    align: left,
    [*Step*], [*Component*], [*Algorithmic action*],
    [1], [Scanner], [Start in `HOST`; skip ordinary host text until a top-level `@triton.jit` decorator is found.],
    [2], [Scanner], [On a top-level JIT decorator, reset indentation and parenthesis state, enter `JIT`, and reprocess the decorator as normal tokens.],
    [3], [Scanner], [In `JIT`, emit tokens for regex matches; for physical newlines at parenthesis depth zero, compare indentation and queue `NEWLINE`, `INDENT`, or `DEDENT`.],
    [4], [Scanner], [Report lexical errors for unsupported characters and flush final layout tokens at EOF.],
    [5], [Parser], [For each accepted JIT block, create a kernel record and record parameters, `tl.constexpr` flags, assigned locals, and qualified `tl.*` calls.],
    [6], [Parser], [Print the final report and exit nonzero when lexical, syntax, semantic, or missing-kernel errors are present.],
  ),
  kind: table,
  caption: [Algorithmic description of scanner and parser processing.],
) <tab-processing-algorithm>

== Symbol and report data model <symbol-and-report-data-model>

#figure(
  table(
    columns: (1fr, 1.25fr, 2fr),
    inset: 5pt,
    align: left,
    [*Structure*], [*Owner*], [*Purpose*],
    [`Symbol`], [`triton.l`], [Stores scanner symbol-table rows: integer ID, token name, lexeme, and first source line.],
    [`PendingToken`], [`triton.l`], [Buffers synthetic `NEWLINE`, `INDENT`, and `DEDENT` tokens so layout can be returned one token at a time.],
    [`ParameterInfo`], [`triton.y`], [Stores a kernel parameter name and whether its annotation was exactly `tl.constexpr`.],
    [`KernelInfo`], [`triton.y`], [Stores kernel name, line, parameters, locals, and called Triton APIs.],
  ),
  kind: table,
  caption: [Runtime data structures used by scanner symbol-table output and parser reports.],
) <tab-data-model>

= Implementation Status <implementation-status>

== Source modules <source-modules>

The scanner implementation is in `compiler/triton.l`. Its definition section declares token numbers for standalone mode, imports Bison token definitions for parser mode, defines the symbol table, and implements the token queue used by indentation handling. Its rules section contains host-skipping rules, JIT token rules, string and numeric literal rules, longest-match operator rules, newline and indentation logic, and lexical diagnostics. Its user-code section provides the standalone scanner `main` and the `triton_next_token` wrapper used by Bison.

The parser implementation is in `compiler/triton.y`. Its definition section declares semantic values, token names, precedence, locations, and report structures. Its grammar section recognizes JIT blocks, signatures, parameters, suites, statements, compound control flow, expressions, calls, attributes, and slices. Its user-code section reports syntax errors, prints the final parser report, and sets the process exit status.

The build automation is in `compiler/Makefile`. The `scanner` target builds `triton_scanner`; the `compiler` target runs Bison and Flex and links `triton_compiler`; `clean` removes generated C files, headers, binaries, and parser output.

== Lex source printout <sec-lex-printout>

The evaluation presentation requires a complete scanner source printout. The listing below is kept as source code because it is implementation evidence, not explanatory prose.

#figure(
  block(width: 100%, inset: 6pt, radius: 3pt, fill: rgb("#f6f8fb"), stroke: 0.45pt + rule)[
    #text(size: 6.2pt)[#raw(read("../triton.l"), lang: "c")]
  ],
  kind: raw,
  caption: [Complete Flex scanner source file (`compiler/triton.l`).],
) <fig-lex-source>

== Runtime behavior <runtime-behavior>

The scanner emits tokens only for JIT islands. This behavior is visible in the vector-add fixture: the file header and imports are skipped, and the first emitted token is the decorator's `AT` token. Layout is represented with synthetic `NEWLINE`, `INDENT`, and `DEDENT` tokens, including final dedents at end-of-file.

The parser accepts multiple kernels in one file. Duplicate kernel names are treated as semantic errors. Files with no top-level JIT kernels produce a report stating that no kernels were found and exit with status 1. This makes the command-line tool useful both for positive validation and for CI-style failure checks.

== Known boundaries <known-boundaries>

The implementation is intentionally a compiler front end for the reduced Triton kernel subset, not a Python interpreter or full Python parser. Host Python is ignored unless it contains a top-level JIT decorator. General Python constructs outside the modeled subset may be skipped in host mode or rejected in JIT mode. The parser report is also intentionally structural: it does not execute kernels, infer types, build an AST, or check numerical correctness.

= Verification and Validation <verification-and-validation>

== Test set <test-set>

The automated tests compile and run the scanner and parser against fixtures under `tests/compiler/fixtures`. Valid fixtures cover vector addition, softmax, dropout, matrix multiplication, flash attention, and layer normalization. Invalid fixtures cover malformed syntax, duplicate kernel names, and missing top-level JIT decorators. Additional in-memory end-to-end tests exercise comments at the beginning of kernel bodies, embedded `tl.arange` calls, `tl.dot` in augmented assignment, return annotations, nested suites, bitwise and shift expressions, multiple decorators, numeric forms, and string forms.

== Execution snapshots <execution-snapshots>

Scanner output for `tests/compiler/fixtures/valid/vector_add.py` starts at the JIT decorator, demonstrating host-code skipping and token emission:

```text
=== Token sequence (Triton JIT blocks only) ===
< AT               , @                        , line 13 >
< NAME             , triton                   , line 13 >
< DOT              , .                        , line 13 >
< NAME             , jit                      , line 13 >
< NEWLINE          , \n                       , line 14 >
< DEF              , def                      , line 14 >
< NAME             , add_kernel               , line 14 >
< LPAREN           , (                        , line 14 >
...
< INDENT           ,                          , line 21 >
< NAME             , pid                      , line 21 >
< EQ               , =                        , line 21 >
< NAME             , tl                       , line 21 >
< DOT              , .                        , line 21 >
< NAME             , program_id               , line 21 >
```

Parser output for the same fixture shows the accepted kernel and its extracted Triton API calls:

```text
=== Triton JIT parser report ===
Kernels parsed: 1

[KERNEL] add_kernel (line 14)
  Parameters: x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
  Local identifiers: pid, block_start, offsets, mask, x, y, output
  Triton calls: tl.program_id, tl.arange, tl.load, tl.store

Errors: lexical=0, syntax=0, semantic=0
```

The layer-normalization fixture demonstrates multi-kernel support:

```text
=== Triton JIT parser report ===
Kernels parsed: 2

[KERNEL] layer_norm_kernel (line 17)
  Parameters: X, W, B, Y, Mean, Rstd, stride, N, eps, BLOCK_SIZE: tl.constexpr
  Triton calls: tl.program_id, tl.arange, tl.load, tl.sum, tl.where, tl.sqrt, tl.store

[KERNEL] layer_norm_bwd_dx_fused (line 57)
  Parameters: DX, DY, DW, DB, X, W, Mean, Rstd, Lock, stride, N, GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
  Triton calls: tl.program_id, tl.arange, tl.load, tl.sum, tl.store

Errors: lexical=0, syntax=0, semantic=0
```

Malformed syntax is reported through Bison diagnostics and a nonzero process status:

```text
SYNTAX ERROR line 12: syntax error
```

Duplicate kernel names are detected after parsing the second declaration header:

```text
SEMANTIC ERROR line 20: duplicate Triton kernel 'add_kernel'

=== Triton JIT parser report ===
Kernels parsed: 1
...
Errors: lexical=0, syntax=0, semantic=1
exit=1
```

A file without a top-level JIT island also exits nonzero while explaining what was missing from the parsed subset:

```text
=== Triton JIT parser report ===
Kernels parsed: 0

No top-level @triton.jit kernels were found.

Errors: lexical=0, syntax=0, semantic=0
exit=1
```

== Validation command <validation-command>

Validation was performed with the following command sequence:

#figure(
  table(
    columns: (auto, 2.8fr),
    inset: 5pt,
    align: left,
    [*Step*], [*Command*],
    [1], [`make -C compiler compiler scanner`],
    [2], [`uv run pytest tests/compiler -q`],
    [3], [`make -C compiler clean`],
  ),
  kind: table,
  caption: [Reproducible validation command sequence.],
) <tab-validation-command>

The compiler test suite passes with 41 tests. The command rebuilds generated Flex/Bison artifacts before test execution and removes generated binaries afterwards.

= References <references>

[1] Free Software Foundation, _Bison: The Yacc-compatible Parser Generator_. GNU Project. Accessed June 10, 2026. Available: https://www.gnu.org/software/bison/manual/.

[2] Free Software Foundation, _Flex: The Fast Lexical Analyzer_. GNU Project. Accessed June 10, 2026. Available: https://westes.github.io/flex/manual/.

[3] Python Software Foundation, _The Python Language Reference_. Accessed June 10, 2026. Available: https://docs.python.org/3/reference/.

[4] Triton Contributors, _Triton Documentation: Language and Programming Guide_. Accessed June 10, 2026. Available: https://triton-lang.org/main/.

[5] IEEE Computer Society, _IEEE Recommended Practice for Software Requirements Specifications_, IEEE Std. 830-1998. New York, NY, USA: IEEE, 1998.
