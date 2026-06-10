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

#let numbered-source(source, start: 1) = block(
  width: 100%,
  inset: 6pt,
  radius: 3pt,
  fill: rgb("#f6f8fb"),
  stroke: 0.45pt + rule,
)[
  #set par(justify: false, leading: 0.48em)
  #set text(size: 6.2pt)
  #for (line, i) in source.split("\n").zip(range(source.split("\n").len())) [
    #box(width: 9mm, align(right)[#text(fill: muted)[#str(start + i)]])
    #h(4pt)
    #if line == "" { h(0pt) } else { raw(line, lang: "c") }
    #linebreak()
  ]
]

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
    The project implements and validates a lexical analyzer for a reduced Triton GPU-kernel language, together with the parser that consumes the resulting token stream. The scanner recognizes the lexemes required by top-level Triton JIT kernels, and the design maps those lexemes to deterministic automata, transition tables, token identifiers, algorithms, and data structures.

    The implemented scanner recognizes top-level `@triton.jit` kernel functions inside practical Python/Triton files, emits the ordered token sequence, constructs a scanner symbol table, and reports invalid symbols or unsupported lexical situations. The implementation uses Flex directly as the UNIX Lex-compatible scanner generator; it does not use Python's `ast` module and does not use native regular-expression APIs as a substitute for the scanner.
  ],
)

#set table(
  fill: (x, y) => if y == 0 { tec-blue } else if calc.odd(y) { rgb("#f7f9fc") } else { none },
  stroke: 0.5pt + rule,
)

#show table.cell.where(y: 0): set text(fill: white, weight: "bold")
#show figure.caption: set text(size: 9pt, fill: muted)

= Introduction <introduction>

== Summary <summary>

The front end is implemented as a Flex/Bison toolchain for a reduced Triton GPU-kernel language and provides two executables:

- `triton_scanner`, a standalone Flex scanner that prints the token stream for Triton JIT blocks and then prints the scanner symbol table.
- `triton_compiler`, a Flex/Bison validator that parses the same token stream and prints a structured report of accepted kernels.

The implementation accepts realistic Python/Triton source files while formally processing only top-level `@triton.jit` functions. Host-side Python code is skipped by the scanner, so imports, launch wrappers, allocation code, and tests can remain in the input file without requiring the project to become a full Python compiler.

The parser recognizes representative Triton kernels such as vector addition, softmax, dropout, matrix multiplication, flash attention, and layer normalization fixtures. On successful input it reports kernel names, formal parameters, `tl.constexpr` parameters, local identifiers, and called `tl.*` APIs. On failure it exits nonzero and reports lexical, syntax, semantic, or missing-kernel diagnostics.

== Notation <notation>

=== Lexeme

A lexeme is a concrete character sequence in the input source file. Examples in the supported Triton subset include `def`, `BLOCK_SIZE`, `tl`, `0xFF`, `+=`, `(`, and the physical newline that separates two statements. The scanner reads lexemes from left to right and never rewrites the input source.

=== Token and token ID

A token is the symbolic class assigned to a lexeme. For example, the lexeme `BLOCK_SIZE` is emitted as `NAME`, while the lexeme `def` is emitted as `DEF`. A token ID is the integer value shared by the scanner and parser for that class. Token IDs are part of the construction blueprint that the implementation follows.

=== Regular expression

A regular expression is the formal notation used to describe a family of valid lexemes. For example, `[A-Za-z_][A-Za-z0-9_]*` defines the identifier family before keyword relabeling. The expressions specify the language to be recognized; the Flex file implements those expressions.

=== Finite state machine and DFA

A finite state machine is a recognition model formed by states and labeled transitions. A deterministic finite automaton (DFA) has at most one next state for each pair of current state and input class. The scanner design is expressed as separate DFAs for mode control, identifiers, numbers, strings, operators, delimiters, and indentation/layout so that each automaton remains readable while the combined behavior still covers the complete token set.

=== Transition table

A transition table is the tabular form of a DFA. Each row names a state, each input-class column describes the next state for that class, and accepting states identify the token returned when no longer input transition can be taken. Compact transition tables define token recognition and layout handling.

=== Symbol table

The scanner symbol table stores the unique token/lexeme pairs that require reporting, together with their first source line. The parser has separate report structures for kernels, parameters, local identifiers, and Triton API calls.

=== Lex/Flex and Bison

Flex is used as the UNIX Lex-compatible scanner generator. It turns the regular-expression specification and attached C actions into the executable scanner. Bison is used only for the parser that consumes scanner tokens and validates the supported grammar. Neither Python's `ast` module nor native regular-expression libraries are used to perform lexical analysis.

=== Development model

The work followed an iterative waterfall model. The language boundary and lexical requirements were analyzed first; the scanner design then translated those requirements into automata, transition tables, token IDs, and symbol-table structures; implementation encoded that design in Flex; and validation checked the produced scanner against realistic Triton fixtures. The model fit the project because the lexical-analysis boundary was well defined, while short iterations against real kernels were still useful for correcting omissions before final validation.

== Scope and assumptions <scope-and-assumptions>

Triton kernels are Python functions decorated with `@triton.jit`. A complete source file commonly contains Python host code around those kernels. The front end therefore distinguishes between the input file and the formal language it validates:

- Outside top-level JIT blocks, the scanner remains in host mode and skips text.
- At a top-level `@triton.jit` decorator, the scanner enters JIT mode and emits tokens.
- After the JIT function dedents back to column zero, the scanner returns to host mode.
- Indented `@triton.jit` decorators are reported as out-of-scope nested JIT blocks.

The supported kernel subset includes decorators, function definitions, parameters, `tl.constexpr` annotations, assignments, augmented assignments, returns, assertions, `if`/`elif`/`else`, `for`, `while`, `with`, function calls, attributes, indexing, slicing, numeric literals, string literals, and Python-like layout tokens. Constructs unrelated to the target kernel subset, such as class declarations, exception handling, lambda expressions, comprehensions, and asynchronous functions, are intentionally not modeled.

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

== Scanner requirements <scanner-requirements>

The scanner requirements state what the scanner shall do, not how Flex encodes it. Each requirement is verifiable during testing and traceable to a design element and implementation location.

#figure(
  table(
    columns: (auto, 2.4fr, 1.3fr),
    inset: 5pt,
    align: left,
    [*Req.*], [*Required scanner behavior*], [*Validation target*],
    [R1], [The scanner shall read a source file or standard input and process characters in source order.], [CLI fixture and stdin tests.],
    [R2], [The scanner shall ignore ordinary host Python text and shall emit tokens only for top-level `@triton.jit` blocks.], [Valid fixtures with imports and host code.],
    [R3], [The scanner shall reject indented `@triton.jit` declarations as nested JIT blocks outside the supported subset.], [Lexical-error path.],
    [R4], [The scanner shall recognize the supported keyword, identifier, literal, operator, delimiter, decorator, and layout lexeme families.], [Token sequence tests and regex inventory.],
    [R5], [The scanner shall apply longest-match behavior for operators and literals so compound tokens such as `**=`, `//`, `->`, and `...` are not split incorrectly.], [Operator-focused token tests.],
    [R6], [The scanner shall convert Python-like physical layout into `NEWLINE`, `INDENT`, and `DEDENT` tokens for JIT blocks.], [Indent/dedent balance tests.],
    [R7], [The scanner shall construct a symbol table containing unique emitted token/lexeme pairs that require reporting, with first source line.], [Standalone scanner symbol-table output.],
    [R8], [The scanner shall report unsupported characters, inconsistent indentation, excessive indentation nesting, and nested JIT declarations as lexical errors.], [Invalid fixture and focused error cases.],
    [R9], [The scanner shall not use Python's `ast` module or native regular-expression libraries as the recognizer.], [Source inspection and build path.],
  ),
  kind: table,
  caption: [Validated scanner requirements for the Triton kernel lexical analyzer.],
) <tab-scanner-requirements>

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
    [C3], [The scanner can emit a human-readable token sequence.], [`emit_token` and `queue_pop` print token name, lexeme, and source line in standalone mode.], [Recorded `triton_scanner vector_add.py` output.],
    [C4], [The scanner records unique lexical entries.], [`add_symbol` stores `(token, lexeme, first line)` in a fixed table and prints it at end of scan.], [`TestScanner` checks concrete tokens, numeric forms, string forms, and INDENT/DEDENT balance.],
    [C5], [Lexical, syntax, semantic, and missing-kernel failures are visible to users.], [Lexical counters in `triton.l`; `yyerror` and semantic duplicate checks in `triton.y`; nonzero exit in parser `main`.], [Invalid fixtures: `bad_syntax.py`, `duplicate_kernel.py`, and `missing_jit.py`.],
    [C6], [The parser validates the supported kernel structure and expression subset.], [Bison grammar in `triton.y` for decorators, signatures, suites, statements, calls, attributes, indexing, slicing, and expression precedence.], [Valid fixtures and end-to-end generated kernels exercise common syntax families.],
    [C7], [Successful parsing produces a compact technical kernel report.], [`KernelInfo`, `ParameterInfo`, `add_parameter`, `add_local`, `add_tl_call`, and `print_report` in `triton.y`.], [Reports for `vector_add.py` and `layer_norm.py` list kernels, parameters, locals, and Triton calls.],
  ),
  kind: table,
  caption: [Delivered capability traceability matrix.],
) <tab-traceability>

== Lexical inventory and omissions <lexical-inventory-and-omissions>

The lexical inventory is the source for the design. In other words, the DFAs and transition tables implement these lexical families; the implementation-specific Flex rules do not define the analysis retroactively. Each included family is necessary for the target Triton kernels or for the scanner's observable outputs.

*Decorators and mode boundary.* A top-level `@triton.jit` decorator marks the start of the formal language island inside a larger Python file. Before this decorator the scanner is in host mode and skips text. Inside the JIT island, the same decorator is emitted as ordinary tokens (`AT NAME DOT NAME`) so the parser can validate the expected block shape.

*Identifiers and keywords.* Identifiers cover kernel names, variables, parameters, aliases such as `tl`, symbolic constants such as `BLOCK_SIZE`, and attribute components. Keywords are not a separate character pattern; they are exact lexeme classifications after the identifier DFA accepts a candidate. This avoids duplicate automata while still producing tokens such as `DEF`, `IF`, `FOR`, `RETURN`, `TRUE`, and `NONE`.

*Numeric literals.* Integer literals cover decimal, hexadecimal, and binary forms used in offsets, masks, dimensions, and constants. Floating literals cover decimal-point and scientific-notation forms used in numerical expressions such as epsilons, scales, and sentinels. Octal literals are intentionally omitted because they are outside the supported Triton kernel subset and would add a token family with no validation target.

*String literals.* Single-line, triple-quoted, raw, byte, Unicode, and formatted-prefix strings are accepted because practical kernels and wrappers may contain arguments such as `float("inf")` or debug names. Single-line string DFAs reject unescaped physical newlines; triple-string DFAs continue until the matching triple delimiter.

*Operators.* Arithmetic, bitwise, comparison, pointer-offset, matrix-multiplication, and augmented-assignment operators are required because Triton kernels heavily use vectorized expressions, masks, and in-place accumulators. Longest-match behavior is mandatory: `**=` must not be emitted as `DOUBLESTAR` followed by `EQ`, and `...` must not be emitted as three `DOT` tokens.

*Delimiters.* Parentheses, brackets, braces, colon, comma, dot, and semicolon delimit calls, parameter lists, indexing, slicing, dictionaries or brace-delimited expressions, attribute access, annotations, suites, and same-line simple statements.

*Layout.* Physical newlines and indentation are lexical components in the supported Python-like syntax. The scanner must emit `NEWLINE`, `INDENT`, and `DEDENT` tokens while suppressing layout inside open parentheses, brackets, or braces.

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


== Scanner automata <scanner-automata>

The automata below are standalone scanner designs derived from the lexical inventory in Analysis. They are not diagrams of Flex's generated internal tables. Large token families are divided into readable DFAs, and tokens with identical structure are grouped when the only difference is the accepted character set or emitted token name.

=== Mode and JIT-island DFA

The scanner has a top-level deterministic mode automaton. `HOST` corresponds to non-emitting transport text; `JIT` corresponds to the formal Triton kernel island where tokens are emitted.

#figure(
  automaton(
    (
      HOST: (JIT: [`@triton.jit` at column 0]),
      JIT: (HOST: [top-level dedent after body]),
    ),
    initial: "HOST",
    final: ("HOST",),
  ),
  caption: [Top-level scanner mode automaton for host skipping and JIT token emission.],
) <fig-host-jit-automaton>

In `HOST`, non-newline text is consumed without token emission. A top-level decorator resets the indentation stack, enters `JIT`, and re-reads the decorator so the formal token stream starts with `AT NAME DOT NAME`. An indented decorator is not a transition to `JIT`; it remains in `HOST` and reports the nested-JIT lexical error required by R3.

=== Identifier and keyword DFA

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

The DFA accepts one identifier candidate. The accepting action performs exact comparison against the keyword set (`def`, `if`, `elif`, `else`, `for`, `in`, `while`, `with`, `as`, `return`, `assert`, `pass`, `and`, `or`, `not`, `is`, `True`, `False`, and `None`). A match emits the keyword token; otherwise the action emits `NAME`.

=== Numeric-literal DFAs

The numeric design is split into decimal/floating forms and base-prefixed integer forms. This keeps exponent and prefix behavior readable while preserving the longest-match rule.

#figure(
  diagram(
    node-stroke: 0.8pt + ink,
    edge-stroke: 0.8pt + ink,
    node((-0.75, 0), [Start], stroke: none),
    edge((-0.75, 0), (0, 0), "-|>"),
    node((0, 0), [$D_0$], radius: 6.5mm),
    edge((0, 0), (1.1, 0), "-|>", [digit]),
    node((1.1, 0), [$D_1$], name: <dec-d1>, radius: 6.5mm, stroke: 1.4pt + ink),
    edge(<dec-d1>, "-|>", <dec-d1>, [digit], bend: 125deg, loop-angle: 90deg),
    edge((1.1, 0), (2.2, -0.75), "-|>", [`.`]),
    node((2.2, -0.75), [$F_1$], name: <dec-f1>, radius: 6.5mm, stroke: 1.4pt + ink),
    edge(<dec-f1>, "-|>", <dec-f1>, [digit], bend: 125deg, loop-angle: 90deg),
    edge((0, 0), (1.1, -1.35), "-|>", [`.`]),
    node((1.1, -1.35), [$P$], radius: 6.5mm),
    edge((1.1, -1.35), (2.2, -1.35), "-|>", [digit]),
    node((2.2, -1.35), [$F_2$], name: <dec-f2>, radius: 6.5mm, stroke: 1.4pt + ink),
    edge(<dec-f2>, "-|>", <dec-f2>, [digit], bend: 125deg, loop-angle: 90deg),
    edge((1.1, 0), (2.2, 0.75), "-|>", [`e/E`]),
    edge((2.2, -0.75), (2.95, 0.15), "-|>", [`e/E`]),
    edge((2.2, -1.35), (2.95, -0.55), "-|>", [`e/E`]),
    node((3.35, 0.15), [$E_s$], radius: 6.5mm),
    edge((3.35, 0.15), (4.35, 0.75), "-|>", [`+/-`]),
    node((4.35, 0.75), [$E_±$], radius: 6.5mm),
    edge((3.35, 0.15), (4.35, -0.15), "-|>", [digit]),
    edge((4.35, 0.75), (4.35, -0.15), "-|>", [digit]),
    node((4.35, -0.15), [$E_1$], name: <dec-e1>, radius: 6.5mm, stroke: 1.4pt + ink),
    edge(<dec-e1>, "-|>", <dec-e1>, [digit], bend: 125deg, loop-angle: 90deg),
  ),
  caption: [Decimal integer and floating-literal DFA. `D_1` accepts `NUMBER_INT`; `F_1`, `F_2`, and `E_1` accept `NUMBER_FLOAT`.],
) <fig-decimal-float-dfa>

#figure(
  diagram(
    node-stroke: 0.8pt + ink,
    edge-stroke: 0.8pt + ink,
    node((-0.9, 0), [Start], stroke: none),
    edge((-0.9, 0), (0, 0), "-|>"),
    node((0, 0), [$N_0$], radius: 6.5mm),
    edge((0, 0), (1.15, 0), "-|>", [`0`]),
    node((1.15, 0), [$Z$], radius: 6.5mm, stroke: 1.4pt + ink),
    edge((1.15, 0), (2.45, -0.72), "-|>", [`x/X`], label-pos: 0.42),
    node((2.45, -0.72), [$H_x$], radius: 6.5mm),
    edge((2.45, -0.72), (3.65, -0.72), "-|>", [hex]),
    node((3.65, -0.72), [$H_1$], name: <dfa-h1>, radius: 6.5mm, stroke: 1.4pt + ink),
    edge(<dfa-h1>, "-|>", <dfa-h1>, [hex], bend: 125deg, loop-angle: 90deg),
    edge((1.15, 0), (2.45, 0.72), "-|>", [`b/B`], label-pos: 0.42),
    node((2.45, 0.72), [$B_x$], radius: 6.5mm),
    edge((2.45, 0.72), (3.65, 0.72), "-|>", [`0/1`]),
    node((3.65, 0.72), [$B_1$], name: <dfa-b1>, radius: 6.5mm, stroke: 1.4pt + ink),
    edge(<dfa-b1>, "-|>", <dfa-b1>, [`0/1`], bend: 125deg, loop-angle: 90deg),
  ),
  caption: [Base-prefixed numeric DFA for hexadecimal and binary integer literals. `Z` accepts decimal zero, `H_1` accepts `NUMBER_HEX`, and `B_1` accepts `NUMBER_BIN`.],
) <fig-number-dfa>

=== String-literal DFA family

The four string DFAs share the same structure and differ only by delimiter length and delimiter character. Prefixes are consumed before the opening delimiter. Escaped characters are consumed as a two-character unit so an escaped quote does not close the string.

#figure(
  diagram(
    node-stroke: 0.8pt + ink,
    edge-stroke: 0.8pt + ink,
    node((-0.75, 0), [Start], stroke: none),
    edge((-0.75, 0), (0, 0), "-|>"),
    node((0, 0), [$S_0$], name: <str-s0>, radius: 6.5mm),
    edge(<str-s0>, "-|>", <str-s0>, [`f/r/b/u` prefix], bend: 125deg, loop-angle: 90deg),
    edge((0, 0), (1.25, 0), "-|>", [opening delimiter]),
    node((1.25, 0), [$S_1$], name: <str-s1>, radius: 6.5mm),
    edge(<str-s1>, "-|>", <str-s1>, [non-delimiter char], bend: 125deg, loop-angle: 90deg),
    edge(<str-s1>, "-|>", <str-s1>, [`\\.` escape], bend: 125deg, loop-angle: 250deg),
    edge((1.25, 0), (2.6, 0), "-|>", [matching delimiter]),
    node((2.6, 0), [$S_2$], radius: 6.5mm, stroke: 1.4pt + ink),
  ),
  caption: [Grouped string-literal DFA. Single-line variants reject unescaped physical newlines; triple-delimited variants use the same body loop until the matching triple delimiter.],
) <fig-string-dfa>

=== Operator and delimiter DFAs

Operators are designed as a trie so the scanner can take the longest valid path before accepting a token. The same structure applies to operator families that differ only by the first character.

#figure(
  diagram(
    node-stroke: 0.8pt + ink,
    edge-stroke: 0.8pt + ink,
    node((-0.7, 0), [Start], stroke: none),
    edge((-0.7, 0), (0, 0), "-|>"),
    node((0, 0), [$O_0$], radius: 6.5mm),
    edge((0, 0), (1.1, -0.9), "-|>", [`*` or `/`]),
    node((1.1, -0.9), [$O_1$], radius: 6.5mm, stroke: 1.4pt + ink),
    edge((1.1, -0.9), (2.25, -0.9), "-|>", [same char]),
    node((2.25, -0.9), [$O_2$], radius: 6.5mm, stroke: 1.4pt + ink),
    edge((2.25, -0.9), (3.35, -0.9), "-|>", [`=`]),
    node((3.35, -0.9), [$O_3$], radius: 6.5mm, stroke: 1.4pt + ink),
    edge((1.1, -0.9), (2.25, -1.65), "-|>", [`=`]),
    node((2.25, -1.65), [$O_4$], radius: 6.5mm, stroke: 1.4pt + ink),
    edge((0, 0), (1.1, 0.15), "-|>", [`<`, `>`, `!`, `=`]),
    node((1.1, 0.15), [$C_1$], radius: 6.5mm, stroke: 1.4pt + ink),
    edge((1.1, 0.15), (2.25, 0.15), "-|>", [`=`, or same shift char]),
    node((2.25, 0.15), [$C_2$], radius: 6.5mm, stroke: 1.4pt + ink),
    edge((2.25, 0.15), (3.35, 0.15), "-|>", [`=` for shifts]),
    node((3.35, 0.15), [$C_3$], radius: 6.5mm, stroke: 1.4pt + ink),
    edge((0, 0), (1.1, 1.1), "-|>", [single delimiter]),
    node((1.1, 1.1), [$D_1$], radius: 6.5mm, stroke: 1.4pt + ink),
  ),
  caption: [Grouped operator/delimiter DFA. Accepting states map to single operators, compound operators, comparisons, shifts, augmented assignments, and one-character delimiters.],
) <fig-operator-dfa>

The delimiter branch accepts `(`, `)`, `[`, `]`, `{`, `}`, `:`, `,`, `.`, and `;`. Open delimiters increment `paren_depth`; close delimiters decrement it when positive. That side effect is part of the layout design because newlines inside a nonzero delimiter depth do not generate `NEWLINE`, `INDENT`, or `DEDENT`.

=== Layout DFA

The layout automaton operates only in `JIT` mode and only on physical newlines. It treats indentation width as a stack value rather than as a fixed finite alphabet, so the table in @tab-layout-transition-table gives the precise transition rule.

#figure(
  automaton(
    (
      L0: (L0: [newline inside open delimiter], L1: [newline at depth 0]),
      L1: (L0: [same indentation], L2: [larger indentation], L3: [smaller indentation], ERR: [inconsistent indentation]),
      L2: (L0: [queue `INDENT`]),
      L3: (L0: [queue one or more `DEDENT`], HOST: [dedent to column 0 after JIT body]),
    ),
    initial: "L0",
    final: ("L0", "HOST"),
  ),
  caption: [Layout DFA for `NEWLINE`, `INDENT`, `DEDENT`, and top-level dedent back to host mode.],
) <fig-layout-dfa>

== Finite transition table <finite-transition-table>

The transition tables use character classes rather than individual characters where those characters have identical behavior. This is the efficient DFA form that a developer can use to implement the scanner without relying on Flex-specific internal table names.

#figure(
  table(
    columns: (auto, 1.1fr, 1fr, 1fr, 1fr, 1fr, 1.2fr),
    inset: 4pt,
    align: left,
    [*State*], [*letter or `_`*], [*digit*], [*`.`*], [*quote*], [*operator/delimiter*], [*Accepting action*],
    [`J0`], [`I1`], [`D1`], [`P` or `DOT` path], [`S0`], [`O1` or delimiter accept], [None],
    [`I1`], [`I1`], [`I1`], [stop], [stop], [stop], [`NAME` or keyword token],
    [`D1`], [stop], [`D1`], [`F1`], [stop], [stop or exponent path], [`NUMBER_INT` unless continued to float],
    [`F1`], [stop], [`F1`], [stop], [stop], [exponent path on `e/E`], [`NUMBER_FLOAT`],
    [`P`], [error/stop], [`F2`], [error/stop], [error/stop], [error/stop], [None; requires following digit],
    [`F2`], [stop], [`F2`], [stop], [stop], [exponent path on `e/E`], [`NUMBER_FLOAT`],
    [`E_s`], [error/stop], [`E1`], [error/stop], [error/stop], [`E_±` on `+/-`], [None; requires exponent digit],
    [`E_±`], [error/stop], [`E1`], [error/stop], [error/stop], [error/stop], [None; requires exponent digit],
    [`E1`], [stop], [`E1`], [stop], [stop], [stop], [`NUMBER_FLOAT`],
    [`Z`], [prefix path on `x/X` or `b/B`], [decimal path], [`F1`], [stop], [stop], [`NUMBER_INT` for single `0`],
    [`H_x`], [hex to `H1`], [hex to `H1`], [error/stop], [error/stop], [error/stop], [None; requires hex digit],
    [`H1`], [hex to `H1`], [hex to `H1`], [stop], [stop], [stop], [`NUMBER_HEX`],
    [`B_x`], [error/stop], [`B1` on `0/1`], [error/stop], [error/stop], [error/stop], [None; requires binary digit],
    [`B1`], [stop], [`B1` on `0/1`], [stop], [stop], [stop], [`NUMBER_BIN`],
    [`S0`], [prefix loop], [prefix loop when prefix char], [error/stop], [`S1` on opening delimiter], [error/stop], [None],
    [`S1`], [`S1`], [`S1`], [`S1`], [accept on matching delimiter], [`S1`, escape subpath on `\\`], [`STRING` at closing delimiter],
    [`O1`], [stop], [stop], [ellipsis path when first char was `.`], [stop], [compound path or accept], [single or compound operator token by longest match],
  ),
  kind: table,
  caption: [Compact token-recognition transition table for the scanner's standalone DFA design.],
) <tab-token-transition-table>

#figure(
  table(
    columns: (auto, 1.4fr, 1.45fr, 1.7fr),
    inset: 5pt,
    align: left,
    [*State*], [*Input condition*], [*Next state*], [*Action*],
    [`HOST`], [Top-level `@triton.jit`], [`JIT`], [Reset indentation and parenthesis state; re-read decorator for normal token emission.],
    [`HOST`], [Indented `@triton.jit` on whitespace-only line], [`HOST`], [Report nested-JIT lexical error and continue skipping host text.],
    [`HOST`], [Other non-newline text], [`HOST`], [Update column and skip.],
    [`HOST`], [Newline], [`HOST`], [Reset column and whitespace-only-line flag.],
    [`JIT`], [Token DFA accepts lexeme], [`JIT`], [Emit token, add symbol-table entry when applicable, and return token.],
    [`JIT`], [Whitespace or comment not including significant newline], [`JIT`], [Skip.],
    [`JIT`], [Newline while `paren_depth > 0`], [`JIT`], [Skip implicit continuation.],
    [`JIT`], [Newline with width equal to stack top], [`JIT`], [Queue `NEWLINE`.],
    [`JIT`], [Newline with width greater than stack top], [`JIT`], [Push width; queue `NEWLINE` and `INDENT`.],
    [`JIT`], [Newline with width less than stack top and matches lower stack level], [`JIT` or `HOST`], [Queue `NEWLINE`, pop levels, queue one `DEDENT` per pop; enter `HOST` when body dedents to column zero.],
    [`JIT`], [Newline with width not present in stack], [`JIT`], [Queue `NEWLINE`; report inconsistent indentation.],
    [`JIT`], [Unsupported character], [`JIT`], [Report lexical error and continue.],
    [`JIT`], [EOF with pending layout], [accept after queue flush], [Queue final `NEWLINE` and `DEDENT` tokens, then return `0`.],
  ),
  kind: table,
  caption: [Mode, layout, and EOF transition table for the scanner.],
) <tab-layout-transition-table>

@tab-token-transition-table and @tab-layout-transition-table together replace the informal transition summary: they identify states, input classes, next states, and accepting actions. The two-table split is deliberate because token recognition is regular over character classes, while layout recognition additionally depends on an indentation stack and delimiter depth.

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

= Implementation <implementation>

== Source modules <source-modules>

The scanner implementation is in `compiler/triton.l`. Its definition section declares token numbers for standalone mode, imports Bison token definitions for parser mode, defines the symbol table, and implements the token queue used by indentation handling. Its rules section contains host-skipping rules, JIT token rules, string and numeric literal rules, longest-match operator rules, newline and indentation logic, and lexical diagnostics. Its user-code section provides the standalone scanner `main` and the `triton_next_token` wrapper used by Bison.

The parser implementation is in `compiler/triton.y`. Its definition section declares semantic values, token names, precedence, locations, and report structures. Its grammar section recognizes JIT blocks, signatures, parameters, suites, statements, compound control flow, expressions, calls, attributes, and slices. Its user-code section reports syntax errors, prints the final parser report, and sets the process exit status.

The build automation is in `compiler/Makefile`. The `scanner` target builds `triton_scanner`; the `compiler` target runs Bison and Flex and links `triton_compiler`; `clean` removes generated C files, headers, binaries, and parser output.

== Flex implementation sections <flex-implementation-sections>

The Flex implementation is organized into the three standard Lex source sections: definitions, rules, and user code.

#figure(
  table(
    columns: (1fr, 1.4fr, 1.6fr),
    inset: 5pt,
    align: left,
    [*Flex section*], [*Design responsibility*], [*Implementation approach*],
    [Definition section], [Declares scanner state, token interface, symbols, helper functions, Flex options, start conditions, and named regular-expression fragments.], [Defines standalone token IDs when Bison is absent, imports `triton.tab.h` when the parser is present, stores unique symbol rows, buffers synthetic layout tokens, tracks indentation and delimiter depth, declares `%x JIT`, and names reusable regex fragments such as `NAME_RE`, `HEX`, `BIN`, `FLOAT`, and `SPACE`.],
    [Rules section], [Implements the DFA transitions and accepting actions from @scanner-automata and @finite-transition-table.], [Uses `INITIAL` rules for host skipping and top-level JIT activation, `JIT` rules for strings, numbers, identifiers, keywords, operators, delimiters, comments, layout, EOF cleanup, and unsupported-character diagnostics. Rule order preserves longest-match behavior for compound operators and literals.],
    [User code section], [Provides executable entry points around the scanner core.], [Defines `triton_next_token` for Bison builds, `next_token` for standalone queue-aware scanning, and `main` for command-line scanner output and final symbol-table printout.],
  ),
  kind: table,
  caption: [Implementation explanation of the three Lex source sections.],
) <tab-flex-section-implementation>

The scanner's implementation follows the design directly: the `INITIAL`/`JIT` start-condition rules implement the mode DFA; named regular expressions implement the token DFAs; rule actions map accepted lexemes to the token IDs in @tab-token-ids; the layout action implements the indentation transition table; and `add_symbol` implements the scanner symbol table required by R7.

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

= Appendix A: Complete Flex scanner source <appendix-flex-source>

The appendix provides the complete `compiler/triton.l` printout split into the three Lex file sections. Line numbers are the original source-file line numbers.

#let triton-l-lines = read("../triton.l").split("\n")
#let flex-definition-source = triton-l-lines.slice(0, 287).join("\n")
#let flex-rules-source = triton-l-lines.slice(289, 468).join("\n")
#let flex-user-source = triton-l-lines.slice(470).join("\n")

== Definition section <appendix-flex-definition>

#figure(
  align(left)[#numbered-source(flex-definition-source, start: 1)],
  kind: raw,
  caption: [Definition section of `compiler/triton.l` with original line numbers.],
) <fig-flex-definition-source>

== Rules section <appendix-flex-rules>

#figure(
  align(left)[#numbered-source(flex-rules-source, start: 290)],
  kind: raw,
  caption: [Rules section of `compiler/triton.l` with original line numbers.],
) <fig-flex-rules-source>

== User code section <appendix-flex-user-code>

#figure(
  align(left)[#numbered-source(flex-user-source, start: 471)],
  kind: raw,
  caption: [User code section of `compiler/triton.l` with original line numbers.],
) <fig-flex-user-source>


= References <references>

[1] Free Software Foundation, _Bison: The Yacc-compatible Parser Generator_. GNU Project. Accessed June 10, 2026. Available: https://www.gnu.org/software/bison/manual/.

[2] Free Software Foundation, _Flex: The Fast Lexical Analyzer_. GNU Project. Accessed June 10, 2026. Available: https://westes.github.io/flex/manual/.

[3] Python Software Foundation, _The Python Language Reference_. Accessed June 10, 2026. Available: https://docs.python.org/3/reference/.

[4] Triton Contributors, _Triton Documentation: Language and Programming Guide_. Accessed June 10, 2026. Available: https://triton-lang.org/main/.
