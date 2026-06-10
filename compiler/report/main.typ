#import "lib.typ": report-template

#set document(title: "Triton GPU Kernel Lexical Analyzer and Parser")
#set heading(numbering: "1.")
#set par(justify: true, spacing: 0.65em)

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
    This report documents the analysis, design, implementation, and testing of a Flex/Bison scanner and parser for a reduced Triton GPU kernel language. The compiler recognizes top-level `@triton.jit` kernel functions, emits a token sequence, builds symbol tables, validates the supported syntax, and reports lexical, syntax, and semantic errors. The implementation uses Flex and Bison directly; it does not use Python's `ast` module or native regular-expression libraries.
  ],
)

#set table(
  fill: (x, y) => if y == 0 { rgb("#1a3c5e") } else { none },
  stroke: 0.8pt + rgb("#cccccc"),
)

#show table.cell.where(y: 0): set text(fill: white, weight: "bold")

= Introduction <introduction>

== Executive summary <executive-summary>

The commissioned front end is implemented as a Flex/Bison toolchain for a reduced Triton GPU-kernel language. The current delivery has two executables:

- `triton_scanner`, a standalone Flex scanner that prints the token stream for Triton JIT blocks and then prints the scanner symbol table.
- `triton_compiler`, a Flex/Bison validator that parses the same token stream and prints a structured report of accepted kernels.

The implementation accepts realistic Python/Triton source files while formally processing only top-level `@triton.jit` functions. Host-side Python code is skipped by the scanner, so imports, launch wrappers, allocation code, and tests can remain in the input file without requiring the project to become a full Python compiler.

The current parser recognizes representative Triton kernels such as vector addition, softmax, dropout, matrix multiplication, flash attention, and layer normalization fixtures. On successful input it reports kernel names, formal parameters, `tl.constexpr` parameters, local identifiers, and called `tl.*` APIs. On failure it exits nonzero and reports lexical, syntax, semantic, or missing-kernel diagnostics.

== Notation and development model <notation-and-development-model>

The lexical specification is written as regular expressions in Flex. Each expression denotes the set of lexemes accepted for a token class. Flex compiles these expressions into deterministic finite automata and applies longest-match selection before running the associated action.

The syntax specification is written as context-free productions in Bison. Bison builds an LALR parser from those productions. Scanner actions pass tokens and lexeme text to the parser; parser actions update the kernel report directly. The design deliberately avoids Python's `ast` module and native regular-expression libraries.

The work followed an iterative waterfall model. The stable language boundary was first captured as a reduced Triton kernel subset; the scanner was then implemented and tested; parser coverage was added over the validated token stream; and regression tests were expanded as real fixtures exposed missing syntax. This model is appropriate for the commission because the high-level scope was known early, while the exact grammar details benefited from short validation cycles against concrete Triton examples.

Flex and Bison are used as the implementation tools because they map directly to the front-end responsibilities: regular-language recognition for lexemes, context-free validation for statements and expressions, and deterministic command-line behavior suitable for automated tests.

== Scope and assumptions <scope-and-assumptions>

Triton kernels are Python functions decorated with `@triton.jit`. A complete source file commonly contains Python host code around those kernels. The front end therefore distinguishes between the input file and the formal language it validates:

- Outside top-level JIT blocks, the scanner remains in host mode and skips text.
- At a top-level `@triton.jit` decorator, the scanner enters JIT mode and emits tokens.
- After the JIT function dedents back to column zero, the scanner returns to host mode.
- Indented `@triton.jit` decorators are reported as out-of-scope nested JIT blocks.

The supported kernel subset includes decorators, function definitions, parameters, `tl.constexpr` annotations, assignments, augmented assignments, returns, assertions, `if`/`elif`/`else`, `for`, `while`, `with`, function calls, attributes, indexing, slicing, numeric literals, string literals, and Python-like layout tokens. Constructs unrelated to the current kernel target, such as class declarations, exception handling, lambda expressions, comprehensions, and asynchronous functions, are intentionally not modeled.

= Analysis <analysis>

== Language boundary <language-boundary>

The formal input recognized by the parser is one or more top-level Triton JIT blocks:

```text
@triton.jit
def kernel_name(parameters):
    statements
```

The scanner treats the surrounding file as transport context, not as part of the validated language. This is the reason the implementation can run against practical fixtures that include imports and test harness code while still keeping the compiler front end small, auditable, and focused on GPU kernels.

== Delivered capability traceability <delivered-capability-traceability>

#align(center)[
  #table(
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
  )
]

== Lexical inventory and omissions <lexical-inventory-and-omissions>

The scanner includes lexemes needed by Triton kernels rather than the whole Python language. Identifiers cover kernel names, variables, parameter names, and qualified names through the separate `DOT` token. Numeric literals cover decimal, binary, octal, hexadecimal, and common floating-point forms. String literals cover quoted, triple-quoted, raw, byte, Unicode, and formatted prefixes because real kernels may call APIs such as `float("inf")` or pass string-like arguments. Operators cover arithmetic, pointer-offset arithmetic, masks, comparisons, matrix multiplication, and augmented assignment. Delimiters cover function calls, indexing, slicing, dictionaries or brace-delimited forms, annotations, and statement separators. Layout tokens model Python block structure.

Broad Python-only constructs are not given keyword tokens unless they are useful to the accepted kernel subset. For example, `class`, `try`, `except`, `lambda`, `async`, `await`, and `raise` remain outside the modeled grammar. This keeps scanner and parser behavior aligned with the current GPU-kernel commission instead of overfitting a partial general Python grammar.

== Formal regular expressions <formal-regular-expressions>

#align(center)[
  #table(
    columns: (1.1fr, 2.2fr, 1.55fr),
    inset: 5pt,
    align: left,
    [*Token or group*], [*Flex expression or literal family*], [*Accepted examples*],
    [`NAME`], [`[A-Za-z_][A-Za-z0-9_]*`], [`add_kernel`, `tl`, `BLOCK_SIZE`],
    [`NUMBER_INT`], [`[0-9]+`], [`0`, `128`],
    [`NUMBER_HEX`], [`0[xX][0-9a-fA-F]+`], [`0xFF`],
    [`NUMBER_BIN`], [`0[bB][01]+`], [`0b1010`],
    [`NUMBER_OCT`], [`0[oO][0-7]+`], [`0o755`],
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
  )
]

== Token ID catalogue <token-id-catalogue>

The public token numbers are shared by standalone scanner mode and the Bison parser. The generated `triton.tab.h` currently assigns the following values:

```text
258 NAME             259 STRING           260 NUMBER_INT       261 NUMBER_FLOAT
262 NUMBER_HEX       263 NUMBER_BIN       264 NUMBER_OCT       265 DEF
266 IF               267 ELIF             268 ELSE             269 FOR
270 IN               271 WHILE            272 WITH             273 AS
274 RETURN           275 ASSERT           276 PASS             277 AND
278 OR               279 NOT              280 IS               281 TRUE
282 FALSE            283 NONE             284 NEWLINE          285 INDENT
286 DEDENT           287 DOUBLESTAREQ     288 DOUBLESLASHEQ    289 LSHIFTEQ
290 RSHIFTEQ         291 PLUSEQ           292 MINUSEQ          293 STAREQ
294 SLASHEQ          295 PERCENTEQ        296 AMPEQ            297 PIPEEQ
298 CARETEQ          299 EQEQ             300 NOTEQ            301 LTEQ
302 GTEQ             303 DOUBLESTAR       304 DOUBLESLASH      305 LSHIFT
306 RSHIFT           307 ARROW            308 ELLIPSIS         309 PLUS
310 MINUS            311 STAR             312 SLASH            313 PERCENT
314 AMP              315 PIPE             316 CARET            317 TILDE
318 AT               319 LT               320 GT               321 EQ
322 LPAREN           323 RPAREN           324 LBRACKET         325 RBRACKET
326 LBRACE           327 RBRACE           328 COLON            329 COMMA
330 DOT              331 SEMI
```

`YYEOF`, `YYerror`, and `YYUNDEF` remain Bison-internal control tokens. Parser-only precedence symbols (`IFX`, `UPLUS`, `UMINUS`, and `UTILDE`) are not emitted by the scanner.

= Design <design>

== Front-end architecture <front-end-architecture>

```text
                 source .py / stdin
                         |
                         v
        +--------------------------------+
        | Flex scanner: triton.l         |
        | - HOST/JIT start conditions    |
        | - token regexes                |
        | - indentation token queue      |
        | - standalone symbol table      |
        +----------------+---------------+
                         |
             token stream| + semantic text
                         v
        +--------------------------------+
        | Bison parser: triton.y         |
        | - kernel grammar               |
        | - expression precedence        |
        | - semantic duplicate checks    |
        | - kernel report aggregation    |
        +----------------+---------------+
                         |
                         v
          token listing or parser report
```

The scanner owns character-level recognition and layout handling. The parser owns grammar validation and higher-level reporting. The executables are built from the same scanner source, with `WITH_PARSER` selecting whether tokens are printed immediately or returned to Bison.

== Scanner automata <scanner-automata>

The scanner has a top-level deterministic mode automaton. `HOST` corresponds to Flex `INITIAL`; `JIT` is an exclusive Flex start condition.

```text
HOST
  -- column 0 and @triton.jit --> JIT, re-read decorator as tokens
  -- indented @triton.jit -----> HOST, emit lexical error
  -- any other host text ------> HOST, skip
  -- EOF ----------------------> accept end of input

JIT
  -- token regex --------------> JIT, emit token
  -- opening delimiter --------> JIT, paren_depth++ and emit token
  -- closing delimiter --------> JIT, paren_depth-- and emit token
  -- newline while depth > 0 --> JIT, skip implicit line join
  -- newline at depth 0 -------> JIT, enqueue NEWLINE and layout tokens
  -- dedent to column 0 -------> HOST after queued DEDENT/NEWLINE tokens
  -- EOF ----------------------> flush final DEDENT tokens, accept
```

Representative token DFAs are summarized below. Flex builds equivalent deterministic automata from the regular expressions in the previous section.

```text
Identifier and keyword DFA

I0 -- letter/_ ------------> I1 accepting NAME candidate
I1 -- letter/digit/_ ------> I1
I1 -- other ---------------> accept; exact lexeme lookup may relabel as keyword

Base-prefixed numeric DFA

N0 -- 0 -------------------> NZ accepting NUMBER_INT
NZ -- x/X -----------------> HX
HX -- hex digit -----------> H1 accepting NUMBER_HEX
H1 -- hex digit -----------> H1
NZ -- b/B -----------------> BX
BX -- 0/1 -----------------> B1 accepting NUMBER_BIN
B1 -- 0/1 -----------------> B1
NZ -- o/O -----------------> OX
OX -- 0..7 ----------------> O1 accepting NUMBER_OCT
O1 -- 0..7 ----------------> O1

Decimal and float DFA

N0 -- digit ---------------> D1 accepting NUMBER_INT
D1 -- digit ---------------> D1
D1 -- '.' -----------------> F1 accepting NUMBER_FLOAT
N0 -- '.' -----------------> FD
FD -- digit ---------------> F1 accepting NUMBER_FLOAT
D1/F1 -- e/E --------------> ES
ES -- + / - ---------------> ED
ES/ED -- digit ------------> E1 accepting NUMBER_FLOAT
E1 -- digit ---------------> E1

Operator trie DFA examples

'*'  accepts STAR; next '*' accepts DOUBLESTAR; next '=' accepts DOUBLESTAREQ
'/'  accepts SLASH; next '/' accepts DOUBLESLASH; next '=' accepts DOUBLESLASHEQ
'<'  accepts LT;    next '<' accepts LSHIFT;     next '=' accepts LSHIFTEQ or LTEQ
'-'  accepts MINUS; next '>' accepts ARROW;       next '=' accepts MINUSEQ
'.'  accepts DOT;   next '..' accepts ELLIPSIS
```

String recognition is handled by four regular-expression DFAs: single-quoted, double-quoted, triple-single-quoted, and triple-double-quoted. Each allows an optional prefix in `[fFrRbBuU]*`, consumes escaped characters as `\\.`, rejects unescaped physical newlines in single-line strings, and accepts only at the matching closing delimiter.

== Finite transition table <finite-transition-table>

#align(center)[
  #table(
    columns: (auto, 1.45fr, 1.35fr, 1.75fr),
    inset: 5pt,
    align: left,
    [*State*], [*Input class*], [*Next state*], [*Action*],
    [`HOST`], [Top-level `@triton.jit`], [`JIT`], [Reset layout state, re-read decorator so `AT NAME DOT NAME` are emitted.],
    [`HOST`], [Indented `@triton.jit` on whitespace-only line], [`HOST`], [Increment lexical-error count and skip nested declaration.],
    [`HOST`], [Other non-newline text], [`HOST`], [Advance host column and skip.],
    [`HOST`], [Newline], [`HOST`], [Reset host column to zero.],
    [`JIT`], [Identifier regex], [`JIT`], [Emit keyword token when lexeme matches keyword set; otherwise emit `NAME`.],
    [`JIT`], [Number regex], [`JIT`], [Emit one of `NUMBER_INT`, `NUMBER_FLOAT`, `NUMBER_HEX`, `NUMBER_BIN`, or `NUMBER_OCT`.],
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
  )
]

== Parser grammar model <parser-grammar-model>

The Bison grammar validates a sequence of JIT blocks. Each block consists of a decorator, a `def` signature, an optional return annotation, and an indented suite. The statement layer supports simple statements separated by semicolons and compound statements with nested suites. The expression layer models Triton-style arithmetic and masks with Bison precedence declarations.

```text
program     -> jit_blocks
jit_block   -> decorator DEF NAME '(' parameters ')' return_annotation ':' suite
suite       -> NEWLINE+ INDENT block DEDENT
block       -> block_item+
block_item  -> NEWLINE | simple_stmt NEWLINE | compound_stmt
simple_stmt -> expr | assignment | annotated_assignment | return | assert | pass
compound    -> if | for | while | with
primary     -> atom | primary '.' NAME | primary '(' arguments ')' | primary '[' slices ']'
```

Parser actions keep the report lightweight. When a kernel name is accepted, `begin_kernel` creates a `KernelInfo` record. Parameter actions record names and `tl.constexpr` flags. Assignment actions record local identifiers. Call actions record only qualified `tl.*` calls, so the final report emphasizes Triton APIs rather than every helper function.

== Processing pseudocode <processing-pseudocode>

```text
scan(input):
  state = HOST
  for each matched lexeme:
    if state == HOST:
      skip ordinary host text
      if top_level_jit_decorator:
        reset indentation and parenthesis state
        state = JIT
        reprocess decorator as normal JIT tokens
    else:
      if lexeme is newline and paren_depth == 0:
        compare indentation against indent_stack
        queue NEWLINE / INDENT / DEDENT tokens
      else if lexeme matches a token expression:
        emit token and optional semantic text
      else:
        report lexical error

parse(tokens):
  for each accepted jit_block:
    create kernel record
    record parameters and constexpr flags
    record assigned local identifiers
    record tl.* calls seen in primary-call productions
  after parse:
    print report
    exit nonzero if any error counter is nonzero or no kernels were found
```

== Symbol and report data model <symbol-and-report-data-model>

#align(center)[
  #table(
    columns: (1fr, 1.25fr, 2fr),
    inset: 5pt,
    align: left,
    [*Structure*], [*Owner*], [*Purpose*],
    [`Symbol`], [`triton.l`], [Stores scanner symbol-table rows: integer ID, token name, lexeme, and first source line.],
    [`PendingToken`], [`triton.l`], [Buffers synthetic `NEWLINE`, `INDENT`, and `DEDENT` tokens so layout can be returned one token at a time.],
    [`ParameterInfo`], [`triton.y`], [Stores a kernel parameter name and whether its annotation was exactly `tl.constexpr`.],
    [`KernelInfo`], [`triton.y`], [Stores kernel name, line, parameters, locals, and called Triton APIs.],
  )
]

= Implementation Status <implementation-status>

== Source modules <source-modules>

The scanner implementation is in `compiler/triton.l`. Its definition section declares token numbers for standalone mode, imports Bison token definitions for parser mode, defines the symbol table, and implements the token queue used by indentation handling. Its rules section contains host-skipping rules, JIT token rules, string and numeric literal rules, longest-match operator rules, newline and indentation logic, and lexical diagnostics. Its user-code section provides the standalone scanner `main` and the `triton_next_token` wrapper used by Bison.

The parser implementation is in `compiler/triton.y`. Its definition section declares semantic values, token names, precedence, locations, and report structures. Its grammar section recognizes JIT blocks, signatures, parameters, suites, statements, compound control flow, expressions, calls, attributes, and slices. Its user-code section reports syntax errors, prints the final parser report, and sets the process exit status.

The build automation is in `compiler/Makefile`. The `scanner` target builds `triton_scanner`; the `compiler` target runs Bison and Flex and links `triton_compiler`; `clean` removes generated C files, headers, binaries, and parser output.

== Current behavior <current-behavior>

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

The current validation sequence is:

```bash
make -C compiler compiler scanner
uv run pytest tests/compiler -q
make -C compiler clean
```

The compiler test suite currently passes with 40 tests. The command rebuilds generated Flex/Bison artifacts before test execution and removes generated binaries afterwards.

= Work Plan <work-plan>

#align(center)[
  #table(
    columns: (1.25fr, 1fr, 2fr),
    inset: 5pt,
    align: left,
    [*Stage*], [*Status*], [*Result or next action*],
    [Language analysis], [Complete], [Reduced the target to top-level Triton JIT kernels embedded in Python files; excluded unrelated Python host constructs.],
    [Lexical design], [Complete], [Defined token inventory, regular expressions, mode automaton, layout handling, token IDs, and scanner symbol table.],
    [Scanner implementation], [Complete], [Implemented Flex `HOST`/`JIT` states, token emission, indentation queue, lexical diagnostics, and standalone scanner mode.],
    [Parser design], [Complete], [Defined Bison grammar for kernel declarations, suites, statements, expressions, calls, attributes, and slices.],
    [Parser implementation], [Complete], [Implemented report structures, duplicate-kernel checks, `tl.constexpr` detection, local collection, and `tl.*` call collection.],
    [Fixture validation], [Complete], [Validated accepted and rejected examples through the automated compiler test suite.],
    [Report packaging], [In progress], [Consolidate formal analysis, automata, transition tables, snapshots, and references into the client-facing technical report.],
    [Final handoff], [Pending], [Regenerate the report artifact, rerun validation from a clean compiler directory, and deliver source plus generated report.],
  )
]

= References <references>

Free Software Foundation. _Bison: The Yacc-compatible Parser Generator_. GNU Project. Accessed June 10, 2026. https://www.gnu.org/software/bison/manual/.

Free Software Foundation. _Flex: The Fast Lexical Analyzer_. GNU Project. Accessed June 10, 2026. https://westes.github.io/flex/manual/.

IEEE Computer Society. _IEEE Recommended Practice for Software Requirements Specifications_. IEEE Std 830-1998. New York: Institute of Electrical and Electronics Engineers, 1998.

Python Software Foundation. _The Python Language Reference_. Accessed June 10, 2026. https://docs.python.org/3/reference/.

Triton Contributors. _Triton Documentation: Language and Programming Guide_. Accessed June 10, 2026. https://triton-lang.org/main/.
