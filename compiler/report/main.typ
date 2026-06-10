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

== Summary <summary>

This project implements a compiler front end for a reduced Triton GPU kernel language. The current software has two executable modes:

- `triton_scanner`, a standalone Flex scanner that prints the sequence of tokens and the scanner symbol table.
- `triton_compiler`, a Flex/Bison validator that parses the supported Triton JIT kernel subset and prints a structured kernel report.

The implementation intentionally focuses on the language elements needed to recognize Triton kernels. It does not attempt to parse complete Python programs.

== Notation <notation>

The lexical specification is written with regular expressions. Each regular expression denotes the set of lexemes accepted for a token class, such as identifiers, numeric literals, strings, operators, and delimiters. Flex compiles these regular expressions into deterministic automata that choose the longest matching lexeme and then execute the associated rule action.

The parser is written with context-free grammar rules in Bison. Bison builds an LALR parser from those productions. The scanner sends tokens to the parser, and parser actions build the kernel report while parsing. No abstract syntax tree is constructed.

== Scope and assumptions <scope-and-assumptions>

Triton kernels are written as Python functions decorated with `@triton.jit`. A complete source file normally contains Python host code around those kernels, such as imports, wrapper functions, memory allocation, launch grids, and tests. This compiler accepts such files as input but formally recognizes only the Triton JIT kernel blocks.

The supported language is intentionally smaller than full Python. It includes the lexical and syntactic forms needed for common Triton kernels: decorators, function definitions, parameters, `tl.constexpr` annotations, assignments, returns, assertions, `if`/`elif`/`else`, `for`, `while`, `with`, expressions, function calls, attributes, indexing, slicing, numeric literals, string literals, and indentation-based suites.

General-purpose Python constructs that are not necessary for this kernel subset are excluded. Examples include class declarations, exception handling, lambda expressions, comprehensions, and asynchronous functions.

= Analysis <analysis>

== Supported language subset <supported-language-subset>

The input file may be a full Python/Triton source file containing imports, wrappers, helper functions, launch code, and tests. The formal language recognized by the compiler is narrower: one or more top-level Triton JIT function blocks.

A supported Triton JIT block has this shape:

```text
@triton.jit
def kernel_name(parameters):
    statements
```

The scanner skips host Python outside top-level `@triton.jit` blocks. This allows the compiler to process realistic Python/Triton files without using Python's `ast` module as a preprocessing step.

Nested JIT declarations are not accepted in the current subset. If an indented `@triton.jit` decorator is found, the scanner reports that nested JIT blocks are outside the supported subset. This choice keeps the required language focused on Triton GPU kernels rather than the full Python host language.

== Functional requirements <functional-requirements>

#align(center)[
  #table(
    columns: (auto, 1.4fr, 1.5fr),
    inset: 6pt,
    align: left,
    [*ID*], [*Requirement*], [*Implementation evidence*],
    [FR1], [Read a source file containing Python/Triton code.], [`main` in `triton.l` and `triton.y` opens the provided input file or reads standard input.],
    [FR2], [Recognize Triton JIT kernel lexemes.], [Flex rules in `triton.l` activate JIT mode at top-level `@triton.jit` and emit tokens inside the block.],
    [FR3], [Print the token sequence.], [`triton_scanner` prints each token as it is recognized.],
    [FR4], [Build a symbol table for tokens that require it.], [`triton.l` stores unique lexemes with token type and first line.],
    [FR5], [Report lexical errors.], [Unsupported characters and nested JIT decorators produce lexical error messages.],
    [FR6], [Validate the supported kernel structure.], [`triton.y` parses decorators, function headers, suites, statements, and expressions.],
    [FR7], [Report parsed kernels.], [`triton_compiler` prints kernel names, parameters, local identifiers, and `tl.*` calls.],
  )
]

== Lexical components <lexical-components>

#align(center)[
  #table(
    columns: (1.3fr, 1.8fr, 2fr),
    inset: 6pt,
    align: left,
    [*Token group*], [*Regular expression or literal form*], [*Purpose*],
    [Identifiers], [`[A-Za-z_][A-Za-z0-9_]*`], [Kernel names, parameters, local variables, and qualified names such as `tl.load`.],
    [Integers], [`[0-9]+`, `0x...`, `0b...`, `0o...`], [Numeric constants used in offsets, block sizes, and masks.],
    [Floats], [`123.`, `123.4`, `.4`, `1e-3`], [Floating-point constants such as `0.0` and `-float("inf")`.],
    [Strings], [Single, double, and triple quoted strings with optional prefixes], [String constants used in calls such as `float("inf")`.],
    [Keywords], [`def`, `if`, `elif`, `else`, `for`, `in`, `while`, `with`, `as`, `return`, `assert`, `pass`, `and`, `or`, `not`, `is`, `True`, `False`, `None`], [Necessary Python-like words for the supported kernel subset.],
    [Operators], [`+ - * / // % ** & | ^ ~ << >>`, comparisons, and assignment variants], [Triton arithmetic, masks, pointer arithmetic, and assignment statements.],
    [Delimiters], [`(` `)` `[` `]` `{` `}` `:` `,` `.` `;`], [Calls, indexing, slicing, blocks, attributes, and statement separators.],
    [Layout tokens], [`NEWLINE`, `INDENT`, `DEDENT`], [Python-style block structure for function bodies and nested statements.],
  )
]

The scanner intentionally does not include broad Python-only keywords such as `class`, `try`, `except`, `lambda`, or `raise`, because they are not necessary for the supported Triton kernel subset.

= Design <design>

== Scanner architecture <scanner-architecture>

The scanner is implemented as a small state machine with two Flex start conditions:

```text
HOST state:
  skip host Python text
  if top-level @triton.jit is found: enter JIT state
  if indented @triton.jit is found: report nested-JIT lexical error

JIT state:
  emit Triton tokens
  track parentheses for implicit line joining
  emit NEWLINE, INDENT, and DEDENT tokens
  return to HOST after the JIT block dedents to top level
```

The `HOST` state is a lexical cleanup mechanism implemented directly in Flex. It is not a Python parser and does not use Python AST processing.

== Symbol table design <symbol-table-design>

The standalone scanner symbol table stores unique `(token, lexeme)` pairs with an integer ID and first line number. The parser report stores higher-level kernel information: kernel names, parameter names, `tl.constexpr` flags, local identifiers assigned inside the kernel, and called Triton API functions.

= Implementation <implementation>

The scanner source file is `compiler/triton.l`. Its definition section declares token IDs for standalone scanner mode, includes Bison token definitions for parser mode, defines the symbol table, and implements the token queue used for indentation. Its rules section contains host-skipping rules, JIT token rules, newline/indentation handling, and lexical error handling. Its user-code section defines the standalone scanner `main` and the `triton_next_token` wrapper used by Bison.

The parser source file is `compiler/triton.y`. Its definition section declares semantic value types, token names, precedence, and report data structures. Its grammar section recognizes top-level JIT blocks, function signatures, parameters, suites, simple statements, compound statements, expressions, calls, attributes, and slices. Its user-code section reports syntax errors and returns a nonzero status when lexical, syntax, semantic, or no-kernel errors are present.

= Verification and Validation <verification-and-validation>

The project tests compile the scanner and parser and run them against valid and invalid fixtures under `tests/compiler/fixtures`. Valid fixtures include vector addition, softmax, dropout, matmul, flash attention, and layer normalization kernels. Invalid fixtures include malformed syntax, duplicate kernel names, and a file with no top-level JIT kernel.

Current validation command:

```bash
make -C compiler compiler scanner
uv run pytest tests/compiler -q
make -C compiler clean
```

The automated compiler test suite passes with 40 tests.

= References <references>

- Python Software Foundation, _The Python Language Reference_, Lexical Analysis and Expressions sections.
- The GNU Project, _Flex: The Fast Lexical Analyzer_ manual.
- The GNU Project, _Bison: The Yacc-compatible Parser Generator_ manual.
- Triton project documentation, _Triton Language_ and _Programming Guide_ sections.
