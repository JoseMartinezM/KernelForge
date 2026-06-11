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

#let accepting-state(pos, label, ..args) = node(
  pos,
  label,
  radius: 6mm,
  stroke: 0.8pt + ink,
  extrude: (-2.5, 0),
  ..args,
)

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

#let listing-counter = counter("listing")

#let numbered-source(source, start: 1) = block(
  width: 100%,
  breakable: true,
  inset: (x: 4pt, y: 3pt),
  fill: rgb("#f6f8fb"),
  stroke: 0.45pt + rule,
)[
  #set par(justify: false, leading: 0.48em)
  #set text(size: 7.8pt)
  #for (line, i) in source.split("\n").zip(range(source.split("\n").len())) [
    #box(width: 0pt)[#move(dx: -9mm)[#box(width: 7mm)[#align(right)[#text(fill: muted)[#str(start + i)]]]]]
    #if line == "" { h(0pt) } else { raw(line, lang: "c") }
    #linebreak()
  ]
]

#let source-listing(source, start: 1, caption: none) = {
  numbered-source(source, start: start)
  v(0.25em)
  listing-counter.step()
  align(center)[#text(size: 11pt, fill: muted)[Listing #context listing-counter.display("1"): #caption]]
}

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

    The implemented scanner recognizes top-level `@triton.jit` kernel functions inside practical Python/Triton files, emits the ordered token sequence, constructs a scanner symbol table, and reports invalid symbols or unsupported lexical situations.
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

=== Host-side Python

Host-side Python means ordinary Python code in the input file that runs on the CPU/Python side around Triton kernels, such as imports, helper functions, allocation code, launch wrappers, and tests. It is not part of the formal kernel language validated by the parser; the scanner skips it until a top-level `@triton.jit` decorator enters JIT mode.

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

Flex is used as the UNIX Lex-compatible scanner generator. It turns the regular-expression specification and attached C actions into the executable scanner [2]. Bison is used only for the parser that consumes scanner tokens and validates the supported grammar [1]. Neither Python's `ast` module nor native regular-expression libraries are used to perform lexical analysis.

== Scope and assumptions <scope-and-assumptions>

Triton kernels are Python functions decorated with `@triton.jit` [4]. A complete source file commonly contains Python host code around those kernels. The front end therefore distinguishes between the input file and the formal language it validates:

- Outside top-level JIT blocks, the scanner remains in host mode and skips text.
- At a top-level `@triton.jit` decorator, the scanner enters JIT mode and emits tokens.
- After the JIT function dedents back to column zero, the scanner returns to host mode.
- Indented `@triton.jit` decorators are reported as out-of-scope nested JIT blocks.

The supported kernel subset includes decorators, function definitions, parameters, `tl.constexpr` annotations, assignments, augmented assignments, returns, assertions, `if`/`elif`/`else`, `for`, `while`, `with`, function calls, attributes, indexing, slicing, numeric literals, string literals, and Python-like layout tokens. The identifier, string, operator, delimiter, and indentation conventions are intentionally Python-like [3], while the recognized JIT boundary is Triton-specific [4]. Constructs unrelated to the target kernel subset, such as class declarations, exception handling, lambda expressions, comprehensions, and asynchronous functions, are intentionally not modeled.

= Analysis <analysis>

== Purpose of the analysis model <analysis-purpose>

The analysis model defines what the lexical analyzer must recognize, ignore, report, and expose before any implementation mechanism is selected. It is the bridge between the system-level goal -- a Lex/Flex scanner for a reduced Triton language -- and the design model that will later define automata, transition tables, token IDs, queues, and data structures.


== Scanner scope and interface <scanner-contract>

An initial analysis was carried out to determine the scope of the lexical scanner and its general interface, ensuring
best possible compatibility with the functionality of Lex and Yacc. Subsequently, a formal list of requirements
would be created based on these items.

#figure(
  table(
    columns: (1.1fr, 3.2fr),
    inset: 5pt,
    align: left,
    [*Interface item*], [*Specification*],
    [Input], [A text source program supplied from a named file or standard input. The program may contain ordinary Python host code and one or more Triton JIT kernel declarations.],
    [Recognized language region], [Only top-level `@triton.jit` function blocks are part of the formal language scanned for tokens.],
    [Ignored transport region], [Ordinary host Python outside top-level JIT blocks is skipped as context. This preserves the distinction between complete practical input files and the smaller formal language validated by the compiler front end.],
    [Primary output], [An ordered stream of tokens representing the lexical components of top-level JIT blocks.],
    [Secondary output], [A scanner symbol table containing unique emitted token/lexeme pairs that require reporting, together with the first source line where each pair appears.],
    [Diagnostics], [Lexical error messages for unsupported nested JIT blocks, inconsistent indentation, excessive indentation nesting, and unsupported characters.],
    [Non-goals], [The scanner is not a full Python lexer, Python parser, Triton semantic analyzer, type checker, optimizer, or kernel executor.],
    [Required technology constraint], [The lexical analyzer must be specified for Lex/Flex. Python's `ast` module and native regular-expression APIs are not acceptable substitutes for the scanner-generator recognizer.],
  ),
  kind: table,
  caption: [Scope and external interface for the Triton JIT lexical analyzer.],
) <tab-scanner-contract>

== Language boundary and supported subset <language-boundary>

The formal language begins at a top-level Triton JIT decorator and continues through the decorated function body:

```python
@triton.jit
def kernel_name(parameters):
    statements
```

The source file around this block is transport context. Imports, helper functions, launch wrappers, allocation code, tests, and comments outside top-level JIT blocks may appear in the same input file, but they are not part of the formal scanner language. In this report, it will be referred to as Host-side or Host Python code. This boundary lets the scanner process realistic Triton files while keeping the lexical definition focused on GPU kernels.

The supported kernel subset includes decorators, function definitions, parameters, `tl.constexpr` annotations, assignments, augmented assignments, returns, assertions, `if`/`elif`/`else`, `for`, `while`, `with`, function calls, attribute access, indexing, slicing, numeric literals, string literals, comments, and Python-like layout. Constructs unrelated to the target kernel subset--such as class declarations, exception handling, lambda expressions, comprehensions, asynchronous functions, and nested or class-contained JIT declarations--are outside the formal lexical boundary.

The scanner shall treat an indented `@triton.jit` decorator as an unsupported nested JIT declaration rather than silently accepting it. The constraint is deliberate: accepting such a decorator would require host-language block tracking before the formal JIT island begins.

== Lexical specification process <lexical-specification-process>

The formal lexical specification follows five steps:

#figure(
  table(
    columns: (auto, 1.35fr, 2.7fr),
    inset: 5pt,
    align: left,
    [*Step*], [*What is required*], [*Justification*],
    [1], [Define the language boundary.], [The scanner must know which parts of a practical source file are formal Triton input and which parts are host-code context to be ignored.],
    [2], [Enumerate lexical component families.], [Every token emitted by the scanner must belong to an identified family: decorator marker, identifier, keyword, literal, operator, delimiter, layout token, or skipped comment/whitespace.],
    [3], [Assign abstract token names.], [The parser and scanner report need stable symbolic identifiers such as `NAME`, `DEF`, `NUMBER_FLOAT`, `INDENT`, and `DOUBLESTAREQ`. Numeric token IDs are a design concern, but token names are part of the analysis contract.],
    [4], [Specify each family formally with regular expressions or a deterministic lexical rule.], [Regular expressions define the accepted lexeme sets; layout requires an additional deterministic rule because indentation depends on surrounding lines.],
    [5], [Define errors and validation expectations.], [The scanner must be testable: every unsupported lexical condition must have a diagnostic, and every requirement must have observable validation evidence.],
  ),
  kind: table,
  caption: [Process used to derive the scanner's formal lexical specification.],
) <tab-specification-process>

== Verifiable scanner requirements <scanner-requirement-contract>

The scanner requirements turn the scope and interface in @tab-scanner-contract into obligations that can be checked against the implementation. They describe what the lexical analyzer must accept, reject, produce, and expose, while still avoiding design mechanisms such as scanner modes, automata, transition tables, queues, and data structures.

#table(
  columns: (auto, 2.1fr, 2.0fr, 1.7fr),
  inset: 5pt,
  align: left,
  [*Req.*], [*Description*], [*Justification*], [*Observable validation*],
  [R1], [The scanner shall read a source program from a named file or from standard input and process characters in source order without rewriting the input.], [The lexical analyzer is the first compiler-front-end stage, so all later behavior depends on preserving the source order of lexemes.], [A fixture file and an equivalent stdin input produce the same token sequence.],
  [R2], [The scanner shall treat ordinary host Python text as transport context and emit tokens only for top-level `@triton.jit` function blocks.], [Practical Triton files include imports, launch wrappers, allocation code, and tests; these are outside the formal language being scanned.], [An input with imports before a kernel emits its first token at the JIT decorator, not at the import statements.],
  [R3], [The scanner shall reject indented `@triton.jit` declarations as nested JIT blocks outside the supported subset.], [Supporting nested or class-contained kernels would require tracking host-language block structure, which is outside the scanner's formal boundary.], [An indented JIT decorator produces a lexical diagnostic and is not treated as the beginning of a token-emitting island.],
  [R4], [The scanner shall recognize every lexical family required by the Triton kernel subset: decorators, identifiers, keywords, numeric literals, string literals, operators, delimiters, comments, and layout tokens.], [These families are the complete lexical vocabulary needed by the parser to validate kernel declarations, suites, statements, expressions, calls, indexing, and annotations.], [Representative kernels produce the expected token classes for names, literals, control-flow keywords, `tl.*` calls, indexing, arithmetic, masks, and statement layout.],
  [R5], [The scanner shall classify lexemes with shared prefixes by choosing the longest valid token at the current input position.], [Several valid lexemes begin with the same characters; for example `**=`, `**`, and `*`, or `.5`, `...`, and `.`. Splitting too early would change the parser input.], [Ambiguous-prefix inputs show outcomes such as `**=` -> `DOUBLESTAREQ` rather than `DOUBLESTAR EQ`, `.5` -> `NUMBER_FLOAT` rather than `DOT NUMBER_INT`, and `...` -> `ELLIPSIS` rather than three `DOT` tokens.],
  [R6], [The scanner shall convert Python-like physical layout in JIT blocks into `NEWLINE`, `INDENT`, and `DEDENT` tokens, while suppressing layout inside open delimiters.], [The parser needs explicit block-boundary tokens, but the source language expresses blocks through indentation and line structure.], [Nested suites produce balanced `INDENT`/`DEDENT` tokens; multiline expressions inside parentheses, brackets, or braces do not produce block-layout tokens.],
  [R7], [The scanner shall build a symbol table for emitted JIT tokens that require reporting, storing each unique token/lexeme pair with the first source line where it appears.], [The standalone scanner must provide an auditable lexical report in addition to the raw token stream.], [Repeated lexemes appear once in the scanner symbol table with their first observed line.],
  [R8], [The scanner shall report lexical errors for unsupported characters, inconsistent indentation, excessive indentation nesting, and unsupported nested JIT declarations, then continue scanning when recovery is possible.], [Users and tests need precise diagnostics for inputs that cannot be tokenized according to the supported lexical model.], [Invalid lexical fixtures produce the documented error category, increment the lexical-error count, and cause a nonzero scanner/compiler result.],
)
 <tab-scanner-requirements>

== Informal lexical component specification <informal-lexical-components>

The lexical inventory is the source for the design: DFAs, transition tables, and Flex rules must implement these families rather than inventing new scanner behavior later.

*Decorators and mode boundary.* A column-zero `@triton.jit` decorator marks the start of a formal JIT island inside a larger Python file. Inside that island, the decorator itself is tokenized as ordinary lexical content: `AT NAME DOT NAME`, optionally followed by a decorator call. This lets the parser validate that the token-emitting block really begins with the expected Triton decorator.

*Identifiers and keywords.* Identifiers name kernels, variables, parameters, aliases such as `tl`, symbolic constants such as `BLOCK_SIZE`, and attribute components such as `program_id` in `tl.program_id`. Keywords are not a different character shape; they are exact identifier lexemes that receive special token names. For example, `def` becomes `DEF`, while `BLOCK_SIZE` remains `NAME`.

*Numeric literals.* Decimal integers, hexadecimal integers, binary integers, and decimal/scientific floating-point literals are required for offsets, masks, tensor dimensions, loop bounds, constants, and numerical sentinels. Octal literals are intentionally omitted because they are not needed by the supported Triton kernel subset.

*String literals.* Single-line quoted strings and triple-quoted strings are accepted, with optional Python-style string prefixes. They cover practical kernel expressions such as `float("inf")` and string arguments. Single-line strings cannot contain an unescaped physical newline; triple-quoted strings continue until the matching triple delimiter.

*Operators.* Arithmetic, matrix multiplication, bitwise, comparison, shift, assignment, augmented-assignment, annotation-arrow, and ellipsis operators are included because Triton kernels use vectorized expressions, masks, pointer arithmetic, indexing expressions, return annotations, and optional sentinel forms. Longest-match classification is required so compound lexemes such as `**=`, `//`, `->`, and `...` are not split into shorter tokens.

*Delimiters.* Parentheses, brackets, braces, colon, comma, dot, and semicolon delimit calls, parameter lists, indexing, slicing, dictionaries or brace-delimited expressions, suite headers, attribute access, annotations, and same-line simple statements.

*Comments and ordinary whitespace.* Comments and inline spaces or tabs separate lexemes but do not produce ordinary tokens. Newlines are different: in JIT mode, they may produce layout tokens depending on indentation and delimiter nesting.

*Layout.* Physical newlines and indentation are lexical components in the supported Python-like syntax. At delimiter depth zero, the scanner must expose line and block structure as `NEWLINE`, `INDENT`, and `DEDENT`. Inside open parentheses, brackets, or braces, physical newlines are implicit continuations and do not define block structure.

== Formal lexical specification <formal-lexical-specification>

The following catalogue gives the formal lexeme families, token names, examples, and reasons for inclusion. Exact numeric token IDs are part of the design catalogue; the analysis model identifies the abstract token names and the lexeme sets they classify.

#table(
  columns: (1.0fr, 1.5fr, 2.4fr, 1.25fr, 2.0fr),
  inset: 4pt,
  align: left,
  [*Family*], [*Token name(s)*], [*Regular expression or lexical rule*], [*Examples*], [*Identification and justification*],
  [JIT boundary], [No boundary token in HOST; inside JIT emits `AT NAME DOT NAME`], [`@triton[ \t]*\.[ \t]*jit` at column zero], [`@triton.jit`], [Identifies the beginning of a formal JIT island. The decorator is then tokenized normally so the parser can validate it.],
  [Identifier], [`NAME`], [`[A-Za-z_][A-Za-z0-9_]*`], [`add_kernel`, `tl`, `BLOCK_SIZE`], [Covers user-defined names, aliases, parameter names, local variables, and attribute components.],
  [Keyword], [`DEF`, `IF`, `ELIF`, `ELSE`, `FOR`, `IN`, `WHILE`, `WITH`, `AS`, `RETURN`, `ASSERT`, `PASS`, `AND`, `OR`, `NOT`, `IS`, `TRUE`, `FALSE`, `NONE`], [First match the `NAME` expression; then relabel exact reserved lexemes listed in @tab-keyword-map.], [`def`, `return`, `True`], [Keeps one identifier recognizer while still exposing control-flow, declaration, Boolean, and null-like words as distinct tokens.],
  [Decimal integer], [`NUMBER_INT`], [`[0-9]+`], [`0`, `128`], [Represents integer dimensions, offsets, loop bounds, constants, and masks.],
  [Hexadecimal integer], [`NUMBER_HEX`], [`0[xX][0-9a-fA-F]+`], [`0xFF`], [Represents bit masks or constants written in base 16.],
  [Binary integer], [`NUMBER_BIN`], [`0[bB][01]+`], [`0b1010`], [Represents bit masks or constants written in base 2.],
  [Floating literal], [`NUMBER_FLOAT`], [`(([0-9]+\.[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+)`], [`0.0`, `.5`, `1.5e-3`], [Represents numerical scales, epsilons, and sentinel values. The expression accepts decimal-point and exponent forms.],
  [Triple double-quoted string], [`STRING`], [`[fFrRbBuU]*"""([^"\\]|\\.|"[^"\\]|""[^"\\])*"""`], [`"""text"""`], [Allows prefixed multiline string literals until the matching triple delimiter.],
  [Triple single-quoted string], [`STRING`], [`[fFrRbBuU]*'''([^'\\]|\\.|'[^'\\]|''[^'\\])*'''`], [`'''text'''`], [Same as triple double-quoted strings, using the single-quote delimiter.],
  [Single-line double-quoted string], [`STRING`], [`[fFrRbBuU]*"([^"\\\n]|\\.)*"`], [`"inf"`, `f"x"`], [Accepts escaped characters but rejects unescaped physical newlines.],
  [Single-line single-quoted string], [`STRING`], [`[fFrRbBuU]*'([^'\\\n]|\\.)*'`], [`'raw'`, `r'raw'`], [Same as double-quoted single-line strings, using the single-quote delimiter.],
  [Comment], [No token], [`#[^\n]*`], [`# mask values`], [Comments separate or explain code but are not syntactic tokens for the parser.],
  [Inline whitespace], [No token], [`[ \t]+` outside leading indentation processing], [Spaces, tabs], [Whitespace separates adjacent lexemes but does not by itself carry syntax inside a line.],
  [Explicit continuation], [No token], [`\\\n`], [Backslash-newline], [Continues a physical line without producing layout tokens.],
  [Layout], [`NEWLINE`, `INDENT`, `DEDENT`], [`\n[ \t]*` plus indentation comparison at delimiter depth zero], [Line breaks and indentation], [Converts Python-like block structure into explicit parser-visible tokens.],
)
#align(center)[#text(size: 9pt, fill: muted)[Formal lexical families and token-identification rationale.]]

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    inset: 4pt,
    align: left,
    [*Lexeme*], [*Token*], [*Lexeme*], [*Token*], [*Lexeme*], [*Token*],
    [`def`], [`DEF`], [`if`], [`IF`], [`elif`], [`ELIF`],
    [`else`], [`ELSE`], [`for`], [`FOR`], [`in`], [`IN`],
    [`while`], [`WHILE`], [`with`], [`WITH`], [`as`], [`AS`],
    [`return`], [`RETURN`], [`assert`], [`ASSERT`], [`pass`], [`PASS`],
    [`and`], [`AND`], [`or`], [`OR`], [`not`], [`NOT`],
    [`is`], [`IS`], [`True`], [`TRUE`], [`False`], [`FALSE`],
    [`None`], [`NONE`], [], [], [], [],
  ),
  kind: table,
  caption: [Exact keyword relabeling after identifier recognition.],
) <tab-keyword-map>

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    inset: 4pt,
    align: left,
    [*Lexeme*], [*Token*], [*Lexeme*], [*Token*], [*Lexeme*], [*Token*],
    [`**=`], [`DOUBLESTAREQ`], [`//=`], [`DOUBLESLASHEQ`], [`<<=`], [`LSHIFTEQ`],
    [`>>=`], [`RSHIFTEQ`], [`+=`], [`PLUSEQ`], [`-=`], [`MINUSEQ`],
    [`*=`], [`STAREQ`], [`/=`], [`SLASHEQ`], [`%=`], [`PERCENTEQ`],
    [`&=`], [`AMPEQ`], [`|=`], [`PIPEEQ`], [`^=`], [`CARETEQ`],
    [`==`], [`EQEQ`], [`!=`], [`NOTEQ`], [`<=`], [`LTEQ`],
    [`>=`], [`GTEQ`], [`**`], [`DOUBLESTAR`], [`//`], [`DOUBLESLASH`],
    [`<<`], [`LSHIFT`], [`>>`], [`RSHIFT`], [`->`], [`ARROW`],
    [`...`], [`ELLIPSIS`], [`+`], [`PLUS`], [`-`], [`MINUS`],
    [`*`], [`STAR`], [`/`], [`SLASH`], [`%`], [`PERCENT`],
    [`&`], [`AMP`], [`|`], [`PIPE`], [`^`], [`CARET`],
    [`~`], [`TILDE`], [`@`], [`AT`], [`<`], [`LT`],
    [`>`], [`GT`], [`=`], [`EQ`], [`(`], [`LPAREN`],
    [`)`], [`RPAREN`], [`[`], [`LBRACKET`], [`]`], [`RBRACKET`],
    [`{`], [`LBRACE`], [`}`], [`RBRACE`], [`:`], [`COLON`],
    [`,`], [`COMMA`], [`.`], [`DOT`], [`;`], [`SEMI`],
  ),
  kind: table,
  caption: [Exact operator and delimiter token identification.],
) <tab-operator-delimiter-map>

The formal expressions above make the identification rule explicit: identifiers are recognized first and then relabeled as keywords; overlapping operators and literals require longest-match behavior; comments and inline whitespace are consumed but not emitted; and layout is specified as a deterministic newline-plus-indentation rule rather than as a simple regular token.

== Lexical error model <lexical-error-model>

Each scanner diagnostic corresponds to a condition where the input cannot be classified according to the supported lexical model. The scanner reports the condition, records that a lexical error occurred, and uses the recovery action below so later errors can still be observed when possible.

#figure(
  table(
    columns: (1.1fr, 1.65fr, 1.9fr, 1.8fr, 1.45fr),
    inset: 5pt,
    align: left,
    [*Error type*], [*Trigger condition*], [*Diagnostic text*], [*Justification*], [*Recovery behavior*],
    [Nested JIT block], [A `@triton.jit` decorator is found on an indented, otherwise whitespace-only host line.], [`LEXICAL ERROR line n: nested @triton.jit blocks are outside the supported subset`], [The formal language begins only at column-zero JIT decorators. Accepting an indented decorator would imply host-language indentation tracking that the scanner does not specify.], [Report the error, keep HOST mode active, and continue skipping the host line as transport text.],
    [Inconsistent indentation], [After a physical newline in JIT mode, the next nonblank line's indentation width is smaller than the current indentation but does not match any previous indentation-stack level.], [`LEXICAL ERROR line n: inconsistent indentation`], [A dedent that does not return to a known indentation level makes block structure ambiguous and cannot be converted into a correct sequence of `DEDENT` tokens.], [Queue the `NEWLINE`, report the error, retain the nearest lower known indentation state, and continue scanning.],
    [Excessive indentation nesting], [A JIT-mode line increases indentation when the indentation stack is already at its configured maximum depth.], [`LEXICAL ERROR line n: indentation nesting is too deep`], [The scanner cannot safely represent more nested block levels than its layout stack allows.], [Report the error, do not accept the new indentation level, and continue scanning from the current line.],
    [Unsupported character], [In JIT mode, no token rule, whitespace rule, comment rule, newline rule, or continuation rule accepts the current character.], [`LEXICAL ERROR line n: unsupported character 'x'`], [The character is outside the formal lexical alphabet for the supported Triton kernel subset.], [Report the error, consume the offending character, and continue in JIT mode.],
  ),
  kind: table,
  caption: [Complete scanner lexical-error model and recovery policy.],
) <tab-lexical-error-model>

== Validation expectations <validation-expectations>

The analysis model is testable when every requirement can be observed without inspecting the scanner source code. The following expectations define the minimum validation evidence for the scanner.

#figure(
  table(
    columns: (auto, 2.2fr, 2.35fr),
    inset: 5pt,
    align: left,
    [*Req.*], [*Validation input*], [*Expected observation*],
    [R1], [A valid JIT fixture supplied from a file and from stdin.], [Both runs produce the same ordered token stream.],
    [R2], [A source file with imports, host helper code, and a top-level JIT kernel.], [Tokens begin at the JIT decorator; host-code lexemes before the decorator are absent from the token stream.],
    [R3], [A source file containing an indented `@triton.jit` decorator.], [The nested-JIT lexical diagnostic is reported and the indented decorator does not start a valid JIT token stream.],
    [R4], [Representative Triton kernels using declarations, parameters, annotations, assignments, calls, indexing, control flow, literals, comments, and layout.], [Every required lexical family appears with the expected token names.],
    [R5], [Inputs that place ambiguous-prefix lexemes in expression context, such as `**=`, `**`, `*`, `...`, `.`, `.5`, `//`, `/`, `/=`, `->`, `<<=`, and `<<`.], [The scanner emits the longest valid token for each case: for example `DOUBLESTAREQ`, `DOUBLESTAR`, `STAR`, `ELLIPSIS`, `DOT`, `NUMBER_FLOAT`, `DOUBLESLASH`, `SLASH`, `SLASHEQ`, `ARROW`, `LSHIFTEQ`, and `LSHIFT` respectively.],
    [R6], [Nested suites and multiline expressions inside parentheses, brackets, or braces.], [Nested suites produce balanced `INDENT` and `DEDENT`; multiline expressions inside delimiters suppress layout tokens.],
    [R7], [A kernel where the same identifier or literal appears multiple times.], [The scanner symbol table contains one row for each unique token/lexeme pair and records the first occurrence line.],
    [R8], [Inputs with unsupported characters, inconsistent indentation, excessive indentation depth, and unsupported nested JIT declarations.], [Each input produces the documented lexical diagnostic and a nonzero failure result.],
  ),
  kind: table,
  caption: [Validation expectations derived from the analysis requirements.],
) <tab-validation-expectations>

= Design <design>

== Design traceability to analysis <design-traceability>

The design turns the analysis contract into a construction blueprint.
By mapping requirements to a deliberate design decision, the implementation can remain organized, attestable to the original requirements, and portable across developer teams.

#figure(
  table(
    columns: (auto, 1.55fr, 1.8fr, 1.85fr),
    inset: 5pt,
    align: left,
    [*Req.*], [*Analysis requirement*], [*Design element*], [*Implementation guidance supplied by design*],
    [R1], [Read a file or standard input in source order.], [Front-end architecture and end-to-end scanner loop.], [Input is read as a character stream; token recognition never rewrites source text or reorders lexemes.],
    [R2], [Emit tokens only for top-level JIT blocks.], [HOST/JIT scanner-control algorithm.], [HOST mode skips transport text; a column-zero `@triton.jit` enters JIT mode and reprocesses the decorator as tokens.],
    [R3], [Reject unsupported nested JIT declarations.], [HOST-mode decorator transition and lexical-error policy.], [An indented `@triton.jit` on an otherwise whitespace-only host line reports the nested-JIT diagnostic and does not enter JIT mode.],
    [R4], [Recognize the complete lexical vocabulary.], [Token ID catalogue, token-recognition DFAs, and transition tables.], [Identifiers, keywords, literals, operators, delimiters, comments, whitespace, and layout are all assigned recognizers and accepting actions.],
    [R5], [Choose the longest valid token when lexeme prefixes overlap.], [Numeric and operator/delimiter trie transition tables.], [`**=`, `**`, `*`, `...`, `.`, `.5`, `//`, `/`, `/=`, `->`, `<<=`, and `<<` are resolved by consuming the longest accepted path before emitting a token.],
    [R6], [Convert physical layout into parser-visible block tokens.], [Delimiter-depth tracking, indentation stack, pending-token queue, and layout pseudocode.], [Newlines at delimiter depth zero queue `NEWLINE`, `INDENT`, and `DEDENT`; newlines inside delimiters or explicit continuations do not create block tokens.],
    [R7], [Build the scanner symbol table.], [Symbol table data structure and insertion algorithm.], [Each emitted non-layout token/lexeme pair is inserted once, assigned a stable one-based ID, and reported with its first source line.],
    [R8], [Report lexical errors and recover when possible.], [Control-transition table and lexical-error recovery actions.], [Nested JIT, inconsistent indentation, excessive indentation depth, and unsupported characters all have specified triggers, messages, and continuation behavior.],
  ),
  kind: table,
  caption: [Traceability from analysis requirements to design components.],
) <tab-design-traceability>

The source cross-reference in @tab-design-source-cross-reference points to Appendix A, where `compiler/triton.l` is printed with original source-file line numbers.
Each requirement can be traced to the indicated line ranges in the project implementation.

#figure(
  table(
    columns: (auto, 1.95fr, 2.55fr),
    inset: 5pt,
    align: left,
    [*Req.*], [*Design element in this section*], [*Source evidence in Appendix A*],
    [R1], [Front-end scanner loop reads input in source order.], [`compiler/triton.l` lines 509--531 open the optional input file, assign `yyin`, call `next_token()` until EOF, print the final symbol table, and return nonzero if lexical errors were seen.],
    [R2], [HOST/JIT scanner-control algorithm.], [Lines 279--280 declare exclusive scanner states; lines 294--320 skip HOST text, enter `JIT` only for a column-zero decorator, and leave ordinary host text un-emitted; lines 442--446 return to HOST after the JIT body dedents to column zero.],
    [R3], [Nested-JIT diagnostic policy.], [Lines 294--307 distinguish a column-zero decorator from an indented decorator; lines 299--303 print the nested-JIT lexical error and keep scanning in HOST mode.],
    [R4], [Token ID catalogue and token-recognition DFAs.], [Lines 20--33 define standalone token IDs, lines 282--290 name reusable regex fragments, lines 324--357 recognize strings, numbers, identifiers, and keywords, and lines 359--406 recognize operators and delimiters.],
    [R5], [Longest-match operator, delimiter, and numeric recognition.], [Lines 329--332 put floating and base-prefixed numeric rules before decimal integers; lines 359--381 put multi-character operator and delimiter rules such as `**=`, `//=` and `...` before their shorter prefixes; lines 383--406 contain the one-character fallbacks.],
    [R6], [Delimiter depth, indentation stack, pending-token queue, and layout algorithm.], [Lines 170--211 define the pending-token queue; lines 242--249 define delimiter and indentation state; lines 251--270 compute indentation width and reset JIT state; lines 397--402 update delimiter depth; lines 408--449 translate physical newlines into queued layout tokens; lines 457 and 460--472 handle explicit continuations and EOF dedents.],
    [R7], [Scanner symbol table and insertion algorithm.], [Lines 38--47 declare symbol rows and storage; lines 63--86 insert unique `(token, lexeme)` pairs while preserving the first line; lines 226--240 call `add_symbol` for emitted tokens; lines 215--223 print the standalone symbol table.],
    [R8], [Lexical-error recovery actions.], [Lines 299--303 report nested JIT decorators, lines 421--425 report excessive indentation, lines 431--439 and 453--455 report and skip after inconsistent indentation, lines 475--478 report unsupported characters, and line 531 propagates lexical errors through the scanner exit status.],
  ),
  kind: table,
  caption: [Appendix-backed source cross-reference for the scanner design requirements.],
) <tab-design-source-cross-reference>

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

As shown in @fig-front-end-architecture, the scanner owns character-level recognition, mode control, indentation handling, lexical diagnostics, the pending-token queue, and the scanner symbol table. The parser owns grammar validation and higher-level reporting after the scanner has produced tokens. The two executables share the scanner design: standalone mode prints tokens immediately, while parser mode returns one token at a time to Bison.

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


== Token-recognition automata <scanner-automata>

This section contains only the DFAs that recognize emitted lexical tokens. Scanner-control behavior that is not itself a token recognizer, like HOST/JIT mode switching, delimiter-depth tracking, indentation-stack processing, pending-token buffering, EOF flushing, and error recovery, is specified separately in #ref(<scanner-control-algorithms>). This separation keeps the token recognizers regular and makes the non-regular layout behavior explicit.

Large token families are divided into readable DFAs, and tokens with identical structure are grouped when the only difference is the accepted character set or emitted token name. In each DFA, an accepting state emits the token named in the caption when no longer valid transition can be taken.

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

The numeric design is split into decimal/floating forms and base-prefixed integer forms. This keeps exponent and prefix behavior readable while preserving the longest-match rule. The dispatcher gives the floating recognizer priority over `DOT` when a dot is followed by a digit, so `.5` becomes `NUMBER_FLOAT`; the operator recognizer handles `.` and `...` when the dot is not part of a floating literal.

#figure(
  diagram(
    node-stroke: 0.8pt + ink,
    edge-stroke: 0.8pt + ink,
    node((-0.8, 0), [Start], stroke: none),
    edge((-0.8, 0), (0, 0), "-|>"),
    node((0, 0), [$D_0$], radius: 6mm),
    edge((0, 0), (1.3, 0), "-|>", [digit]),
    accepting-state((1.3, 0), [$D_1$], name: <dec-d1>),
    edge(<dec-d1>, "-|>", <dec-d1>, [digit], bend: 125deg, loop-angle: 90deg),
    edge((0, 0), (1.3, 1.1), "-|>", [`.`], label-side: left),
    node((1.3, 1.1), [$P$], radius: 6mm),
    edge((1.3, 1.1), (2.6, 1.1), "-|>", [digit]),
    accepting-state((2.6, 1.1), [$F_2$], name: <dec-f2>),
    edge(<dec-f2>, "-|>", <dec-f2>, [digit], bend: 125deg, loop-angle: 90deg),
    edge((1.3, 0), (2.6, -0.85), "-|>", [`.`]),
    accepting-state((2.6, -0.85), [$F_1$], name: <dec-f1>),
    edge(<dec-f1>, "-|>", <dec-f1>, [digit], bend: 125deg, loop-angle: 90deg),
    node((4.05, 0.05), [$E_s$], radius: 6mm),
    edge((1.3, 0), (4.05, 0.05), "-|>", [`e/E`], label-pos: 0.55),
    edge((2.6, 1.1), (4.05, 0.05), "-|>", [`e/E`], label-pos: 0.55),
    edge((2.6, -0.85), (4.05, 0.05), "-|>", [`e/E`], label-pos: 0.55),
    edge((4.05, 0.05), (5.25, 0.95), "-|>", [`+/-`]),
    node((5.25, 0.95), [$E_±$], radius: 6mm),
    edge((4.05, 0.05), (5.25, -0.45), "-|>", [digit]),
    edge((5.25, 0.95), (5.25, -0.45), "-|>", [digit], label-side: right),
    accepting-state((5.25, -0.45), [$E_1$], name: <dec-e1>),
    edge(<dec-e1>, "-|>", <dec-e1>, [digit], bend: 125deg, loop-angle: 90deg),
  ),
  caption: [Decimal integer and floating-literal DFA. `D_1` accepts `NUMBER_INT`; `F_1`, `F_2`, and `E_1` accept `NUMBER_FLOAT`.],
) <fig-decimal-float-dfa>

#figure(
  diagram(
    node-stroke: 0.8pt + ink,
    edge-stroke: 0.8pt + ink,
    node((-0.8, 0), [Start], stroke: none),
    edge((-0.8, 0), (0, 0), "-|>"),
    node((0, 0), [$N_0$], radius: 6mm),
    edge((0, 0), (1.2, 0), "-|>", [`0`]),
    accepting-state((1.2, 0), [$Z$]),
    edge((1.2, 0), (2.5, -0.85), "-|>", [`x/X`], label-pos: 0.5),
    node((2.5, -0.85), [$H_x$], radius: 6mm),
    edge((2.5, -0.85), (3.8, -0.85), "-|>", [hex]),
    accepting-state((3.8, -0.85), [$H_1$], name: <dfa-h1>),
    edge(<dfa-h1>, "-|>", <dfa-h1>, [hex], bend: 125deg, loop-angle: 270deg),
    edge((1.2, 0), (2.5, 0.85), "-|>", [`b/B`], label-pos: 0.5),
    node((2.5, 0.85), [$B_x$], radius: 6mm),
    edge((2.5, 0.85), (3.8, 0.85), "-|>", [`0/1`]),
    accepting-state((3.8, 0.85), [$B_1$], name: <dfa-b1>),
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
    node((-0.85, 0), [Start], stroke: none),
    edge((-0.85, 0), (0, 0), "-|>"),
    node((0, 0), [$S_0$], name: <str-s0>, radius: 6mm),
    edge(<str-s0>, "-|>", <str-s0>, [prefix], bend: 125deg, loop-angle: 130deg),
    edge((0, 0), (1.75, 0), "-|>", [open quote]),
    node((1.75, 0), [$S_1$], name: <str-s1>, radius: 6mm),
    edge(<str-s1>, "-|>", <str-s1>, [body char], bend: 125deg, loop-angle: 90deg),
    edge(<str-s1>, "-|>", <str-s1>, [escape], bend: 125deg, loop-angle: 270deg),
    edge((1.75, 0), (3.5, 0), "-|>", [close quote]),
    accepting-state((3.5, 0), [$S_2$]),
  ),
  caption: [Grouped string-literal DFA. Single-line variants reject unescaped physical newlines; triple-delimited variants use the same body loop until the matching triple delimiter.],
) <fig-string-dfa>

=== Operator and delimiter DFAs

Operators are designed as a trie so the scanner can take the longest valid path before accepting a token. The same structure applies to operator families that differ only by the first character.

#figure(
  diagram(
    node-stroke: 0.8pt + ink,
    edge-stroke: 0.8pt + ink,
    node((-0.8, 0), [Start], stroke: none),
    edge((-0.8, 0), (0, 0), "-|>"),
    node((0, 0), [$O_0$], radius: 6mm),
    edge((0, 0), (1.4, -1.2), "-|>", [`* /`]),
    accepting-state((1.4, -1.2), [$A_1$]),
    edge((1.4, -1.2), (2.8, -1.2), "-|>", [same]),
    accepting-state((2.8, -1.2), [$A_2$]),
    edge((1.4, -1.2), (2.8, -2.05), "-|>", [`=`]),
    accepting-state((2.8, -2.05), [$A_3$]),
    edge((2.8, -1.2), (4.15, -1.2), "-|>", [`=`]),
    accepting-state((4.15, -1.2), [$A_4$]),
    edge((0, 0), (1.4, 0), "-|>", [`< > = !`]),
    accepting-state((1.4, 0), [$C_1$]),
    edge((1.4, 0), (2.8, 0), "-|>", [`=` or shift]),
    accepting-state((2.8, 0), [$C_2$]),
    edge((2.8, 0), (4.15, 0), "-|>", [`=`]),
    accepting-state((4.15, 0), [$C_3$]),
    edge((0, 0), (1.4, 1.2), "-|>", [delimiter]),
    accepting-state((1.4, 1.2), [$D_1$]),
  ),
  caption: [Grouped operator/delimiter DFA. Accepting states map to one-character operators, multi-character operators, comparisons, shifts, augmented assignments, and one-character delimiters.],
) <fig-operator-dfa>

The delimiter branch accepts `(`, `)`, `[`, `]`, `{`, `}`, `:`, `,`, `.`, and `;`. Open delimiters increment `paren_depth`; close delimiters decrement it when positive. This value is used to ensure line continuations within delimeters do not emit `NEWLINE`, `INDENT`, or `DEDENT`.

== Efficient transition tables <finite-transition-table>

The transition tables use character classes rather than individual characters where those characters have identical behavior. This is the efficient DFA form that a developer can implement directly or translate into Lex/Flex regular-expression rules. In these tables, *stop* means: emit the token from the last accepting state and leave the current character to be scanned as the first character of the next token.

=== Identifier and keyword transition table

#figure(
  table(
    columns: (auto, 1.35fr, 1.25fr, 1.85fr),
    inset: 5pt,
    align: left,
    [*State*], [*Input class*], [*Next state*], [*Accepting action*],
    [`I0`], [letter or `_`], [`I1`], [None],
    [`I0`], [any other input], [no transition], [No identifier token begins here.],
    [`I1`], [letter, digit, or `_`], [`I1`], [Continue current identifier candidate.],
    [`I1`], [any other input], [stop], [If lexeme is in @tab-keyword-map, emit its keyword token; otherwise emit `NAME`.],
  ),
  kind: table,
  caption: [Efficient transition table for identifier recognition and keyword relabeling.],
) <tab-identifier-transition-table>

=== Numeric-literal transition table

#figure(
  table(
    columns: (auto, 1.25fr, 1.15fr, 1.9fr),
    inset: 5pt,
    align: left,
    [*State*], [*Input class*], [*Next state*], [*Accepting action*],
    [`N0`], [`0`], [`Z`], [`NUMBER_INT` if no longer numeric path matches.],
    [`N0`], [`1`-`9`], [`D1`], [`NUMBER_INT` if no longer numeric path matches.],
    [`N0`], [`.` followed by digit], [`P` then `F2`], [None until at least one digit after `.` is consumed.],
    [`D1`], [digit], [`D1`], [`NUMBER_INT`],
    [`D1`], [`.`], [`F1`], [`NUMBER_FLOAT`; digits after the dot remain in `F1`.],
    [`D1`], [`e` or `E`], [`Es`], [None until exponent digits are accepted.],
    [`Z`], [`x` or `X`], [`Hx`], [Requires at least one following hex digit to accept `NUMBER_HEX`; otherwise longest-match fallback emits `NUMBER_INT` for `0`.],
    [`Z`], [`b` or `B`], [`Bx`], [Requires at least one following binary digit to accept `NUMBER_BIN`; otherwise longest-match fallback emits `NUMBER_INT` for `0`.],
    [`Z`], [digit], [`D1`], [`NUMBER_INT`],
    [`Z`], [`.`], [`F1`], [`NUMBER_FLOAT`],
    [`F1`], [digit], [`F1`], [`NUMBER_FLOAT`],
    [`F1`], [`e` or `E`], [`Es`], [None until exponent digits are accepted.],
    [`P`], [digit], [`F2`], [`NUMBER_FLOAT`],
    [`F2`], [digit], [`F2`], [`NUMBER_FLOAT`],
    [`F2`], [`e` or `E`], [`Es`], [None until exponent digits are accepted.],
    [`Es`], [`+` or `-`], [`Epm`], [None],
    [`Es`], [digit], [`E1`], [`NUMBER_FLOAT`],
    [`Epm`], [digit], [`E1`], [`NUMBER_FLOAT`],
    [`E1`], [digit], [`E1`], [`NUMBER_FLOAT`],
    [`Hx`], [hex digit], [`H1`], [`NUMBER_HEX`],
    [`H1`], [hex digit], [`H1`], [`NUMBER_HEX`],
    [`Bx`], [`0` or `1`], [`B1`], [`NUMBER_BIN`],
    [`B1`], [`0` or `1`], [`B1`], [`NUMBER_BIN`],
  ),
  kind: table,
  caption: [Efficient transition table for integer and floating-literal recognition.],
) <tab-number-transition-table>

=== String-literal transition table

#figure(
  table(
    columns: (auto, 1.35fr, 1.15fr, 1.9fr),
    inset: 5pt,
    align: left,
    [*State*], [*Input class*], [*Next state*], [*Accepting action*],
    [`S0`], [`f`, `F`, `r`, `R`, `b`, `B`, `u`, or `U`], [`S0`], [Consume optional string prefix characters.],
    [`S0`], [single quote, double quote, triple single quote, or triple double quote], [`S1`], [Remember delimiter kind and length.],
    [`S1`], [escaped character `\\x`], [`S1`], [Consume escape as part of current string.],
    [`S1`], [body character that is not the matching delimiter], [`S1`], [Continue string body.],
    [`S1`], [matching closing delimiter], [`S2`], [`STRING`],
    [`S1`], [unescaped physical newline in single-line variant], [no transition], [No `STRING`; newline is not part of a single-line string.],
    [`S2`], [any input], [stop], [Emit `STRING` and scan next input separately.],
  ),
  kind: table,
  caption: [Efficient transition table for the grouped string-literal recognizer.],
) <tab-string-transition-table>

=== Operator and delimiter transition table

#figure(
  table(
    columns: (1.05fr, 1.65fr, 1.7fr, 1.5fr),
    inset: 5pt,
    align: left,
    [*Consumed prefix*], [*Longer valid continuation*], [*Token if longest path ends here*], [*Notes*],
    [`*`], [`*` -> `**`; `=` -> `*=`; `**` then `=` -> `**=`], [`STAR`, `DOUBLESTAR`, `STAREQ`, or `DOUBLESTAREQ`], [Longest match chooses `**=` before `**` before `*`.],
    [`/`], [`/` -> `//`; `=` -> `/=`; `//` then `=` -> `//=`], [`SLASH`, `DOUBLESLASH`, `SLASHEQ`, or `DOUBLESLASHEQ`], [Longest match chooses `//=` before `//` before `/`.],
    [`<`], [`=` -> `<=`; `<` -> `<<`; `<<` then `=` -> `<<=`], [`LT`, `LTEQ`, `LSHIFT`, or `LSHIFTEQ`], [Shift and comparison prefixes share the same first character.],
    [`>`], [`=` -> `>=`; `>` -> `>>`; `>>` then `=` -> `>>=`], [`GT`, `GTEQ`, `RSHIFT`, or `RSHIFTEQ`], [Same trie shape as `<`.],
    [`=`], [`=` -> `==`], [`EQ` or `EQEQ`], [Assignment and equality comparison.],
    [`!`], [`=` -> `!=`], [`NOTEQ` only if followed by `=`], [A bare `!` has no token and is reported as unsupported.],
    [`.`], [`.` then `.` -> `...`; digit after first `.` belongs to numeric DFA], [`DOT` or `ELLIPSIS`], [Dispatch separates `.5` from `.` and `...`.],
    [`+`, `-`, `%`, `&`, `|`, `^`], [`=` for augmented assignment; `-` also accepts `>` for `->`], [Single operator, augmented operator, or `ARROW`], [Covers arithmetic, bitwise, and return-annotation syntax.],
    [`~`, `@`], [None], [`TILDE` or `AT`], [`@` is both decorator marker and matrix-multiplication operator token.],
    [`(`, `[`, `{`], [None], [`LPAREN`, `LBRACKET`, or `LBRACE`], [Emit delimiter and increment delimiter depth.],
    [`)`, `]`, `}`], [None], [`RPAREN`, `RBRACKET`, or `RBRACE`], [Emit delimiter and decrement delimiter depth if positive.],
    [`:`, `,`, `;`], [None], [`COLON`, `COMMA`, or `SEMI`], [One-character separators.],
  ),
  kind: table,
  caption: [Efficient trie-style transition table for operators and delimiters.],
) <tab-operator-transition-table>

Together, @tab-identifier-transition-table, @tab-number-transition-table, @tab-string-transition-table, and @tab-operator-transition-table provide the token-recognition transition tables. Scanner-control transitions for mode, layout, EOF, and diagnostics are specified separately below because they depend on state variables rather than only the current token DFA state.

== Scanner control algorithms <scanner-control-algorithms>

The scanner requires control logic in addition to token DFAs. This logic decides when token recognition is active, suppresses host Python, converts layout into queued tokens, and recovers from lexical errors. It is not presented as a token DFA because indentation depends on a stack of previous line widths and delimiter depth.

=== HOST/JIT mode control

#figure(
  automaton(
    (
      HOST: (JIT: [`@triton.jit` at column 0]),
      JIT: (HOST: [body dedents to column 0 after at least one body indent]),
    ),
    initial: "HOST",
    final: ("HOST",),
  ),
  caption: [Scanner-control automaton for host skipping and JIT token emission.],
) <fig-host-jit-automaton>

`HOST` mode consumes transport text without token emission. A column-zero `@triton.jit` decorator resets scanner-control state, enters `JIT`, and reprocesses the decorator so the formal token stream begins with `AT NAME DOT NAME`. An indented decorator on an otherwise whitespace-only host line reports the nested-JIT lexical error and remains in `HOST`.

=== Layout state variables

#figure(
  table(
    columns: (1.15fr, 1.4fr, 2.35fr),
    inset: 5pt,
    align: left,
    [*Variable*], [*Initial value*], [*Purpose*],
    [`mode`], [`HOST`], [Controls whether the scanner skips host text or emits JIT tokens.],
    [`indent_stack`], [`[0]`], [Stores accepted indentation widths. The top value is the current block indentation.],
    [`indent_top`], [`0`], [Indexes the current stack top and bounds indentation nesting.],
    [`saw_jit_body_indent`], [`false`], [Prevents the scanner from leaving JIT mode before the decorated function body has actually started.],
    [`paren_depth`], [`0`], [Counts open `(`, `[`, and `{` delimiters so physical newlines inside expressions can be suppressed.],
    [`pending`], [empty FIFO], [Stores synthetic `NEWLINE`, `INDENT`, and `DEDENT` tokens that must be returned one at a time.],
    [`host_column`], [`0`], [Tracks whether a host-mode decorator begins at column zero or is indented.],
    [`host_line_has_only_space`], [`true`], [Distinguishes indented nested decorators from `@triton.jit` text that appears later in an ordinary host line.],
  ),
  kind: table,
  caption: [Scanner-control state required for mode and layout handling.],
) <tab-layout-state>

Indentation width is computed from the whitespace after a physical newline. Each space adds one column. Each tab advances to the next multiple of four columns. The stack has a fixed maximum depth of 64 indentation levels; exceeding that depth triggers the excessive-indentation lexical error.

=== Layout and token-return pseudocode

```text
next_token():
    if pending is not empty:
        return pending.pop_front()

    if mode == HOST:
        scan characters without emitting tokens
        maintain host_column and host_line_has_only_space

        if column-zero "@triton.jit" is found:
            reset paren_depth, indent_stack = [0], saw_jit_body_indent = false
            mode = JIT
            re-read the decorator as normal JIT input

        if indented "@triton.jit" is found on a whitespace-only host line:
            report nested-JIT lexical error
            remain in HOST

    if mode == JIT:
        if the next lexeme is a token DFA match:
            emit the longest accepted token
            record the token/lexeme pair in the scanner symbol table
            update paren_depth for opening or closing delimiters
            return token

        if the next input is comment text before a newline:
            consume the comment and let the newline rule handle line structure

        if the next input is explicit backslash-newline:
            consume it and emit no layout token

        if the next input is ordinary inline spaces or tabs:
            consume it and emit no token

        if the next input is a physical newline:
            handle_layout_newline()
            return pending.pop_front()

        if no rule accepts the current character:
            report unsupported-character lexical error
            consume that character and continue scanning in JIT mode
```

```text
handle_layout_newline():
    if paren_depth > 0:
        consume the physical newline and following indentation
        emit no NEWLINE, INDENT, or DEDENT
        return

    width = indentation_width(whitespace after newline)
    pending.push(NEWLINE)

    if the next line is blank or comment-only:
        do not push or pop indentation levels
        return

    if width > indent_stack.top:
        if indent_stack is full:
            report excessive-indentation lexical error
        else:
            indent_stack.push(width)
            saw_jit_body_indent = true
            pending.push(INDENT)
        return

    while width < indent_stack.top:
        indent_stack.pop()
        pending.push(DEDENT)

    if width != indent_stack.top:
        report inconsistent-indentation lexical error
        return

    if saw_jit_body_indent and width == 0 and indent_stack.top == 0:
        mode = HOST
        saw_jit_body_indent = false
```

```text
handle_eof():
    if pending is not empty:
        return pending.pop_front()

    if mode == JIT and indent_stack.top > 0:
        pending.push(NEWLINE)
        while indent_stack.top > 0:
            indent_stack.pop()
            pending.push(DEDENT)
        return pending.pop_front()

    return EOF
```

Blank or comment-only lines in JIT mode may produce a `NEWLINE` so the parser can consume harmless line breaks, but they must not change indentation state. Layout is suppressed entirely inside open parentheses, brackets, or braces because those physical newlines are implicit continuations instead of statement or block boundaries.

=== Scanner-control transition table

#figure(
  table(
    columns: (auto, 1.45fr, 1.25fr, 2.05fr),
    inset: 5pt,
    align: left,
    [*State*], [*Input condition*], [*Next state*], [*Action*],
    [`HOST`], [Column-zero `@triton.jit`], [`JIT`], [Reset delimiter and indentation state; re-read decorator for normal token emission.],
    [`HOST`], [Indented `@triton.jit` on a whitespace-only host line], [`HOST`], [Report nested-JIT lexical error and continue skipping host text.],
    [`HOST`], [Other non-newline text], [`HOST`], [Update host-column state and skip.],
    [`HOST`], [Newline], [`HOST`], [Reset host column to zero and mark the next line as whitespace-only until a non-space character appears.],
    [`JIT`], [Token DFA accepts lexeme], [`JIT`], [Emit longest token, record symbol-table entry, and update delimiter depth when applicable.],
    [`JIT`], [Whitespace or comment before a significant newline], [`JIT`], [Skip; newline processing remains separate.],
    [`JIT`], [Explicit backslash-newline], [`JIT`], [Skip continuation and emit no layout token.],
    [`JIT`], [Newline while `paren_depth > 0`], [`JIT`], [Skip implicit continuation and emit no layout token.],
    [`JIT`], [Newline with same indentation width], [`JIT`], [Queue `NEWLINE`.],
    [`JIT`], [Newline with greater indentation width], [`JIT`], [Queue `NEWLINE`; push width and queue `INDENT`, unless the indentation stack is full.],
    [`JIT`], [Newline with smaller width matching a previous stack level], [`JIT` or `HOST`], [Queue `NEWLINE`; pop stack and queue one `DEDENT` per popped level; return to `HOST` after body dedent to column zero.],
    [`JIT`], [Newline with smaller width not matching any previous stack level], [`JIT`], [Queue `NEWLINE`; report inconsistent-indentation lexical error.],
    [`JIT`], [Unsupported character], [`JIT`], [Report unsupported-character lexical error, consume the character, and continue.],
    [`JIT`], [EOF with pending indentation], [EOF after queue flush], [Queue final `NEWLINE` and all required `DEDENT` tokens, then return queued tokens before EOF.],
  ),
  kind: table,
  caption: [Control transition table for scanner modes, layout, EOF, and lexical recovery.],
) <tab-control-transition-table>

== Downstream parser contract <parser-grammar-model>

The lexical analyzer is complete when it provides the token stream expected by the parser. The parser contract is therefore included only to show why the scanner emits particular token families; parser implementation details are left to the Bison section. The parser validates a sequence of JIT blocks. Each block consists of a decorator, a `def` signature, an optional return annotation, and an indented suite. The statement layer supports simple statements separated by semicolons and compound statements with nested suites. The expression layer consumes the scanner's arithmetic, bitwise, comparison, call, attribute, indexing, and literal tokens.

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

The important scanner-facing requirement is that `NEWLINE`, `INDENT`, and `DEDENT` delimit suites; `AT NAME DOT NAME` exposes the decorator; `DEF`, `NAME`, delimiters, and `ARROW` expose the function signature; and the expression tokens preserve enough structure for calls, attributes, indexing, masks, and arithmetic to parse deterministically.

== End-to-end processing algorithm <processing-pseudocode>

#figure(
  table(
    columns: (auto, 1.25fr, 2.7fr),
    inset: 5pt,
    align: left,
    [*Step*], [*Component*], [*Algorithmic action*],
    [1], [Scanner], [Start in `HOST`; skip ordinary host text until a top-level `@triton.jit` decorator is found.],
    [2], [Scanner], [On a top-level JIT decorator, reset indentation and parenthesis state, enter `JIT`, and reprocess the decorator as normal tokens.],
    [3], [Scanner], [In `JIT`, use the token-recognition transition tables to emit the longest accepted lexeme and record its symbol-table entry.],
    [4], [Scanner], [For physical newlines at delimiter depth zero, apply the layout algorithm and return queued `NEWLINE`, `INDENT`, and `DEDENT` tokens one at a time.],
    [5], [Scanner], [Report specified lexical errors, recover according to @tab-control-transition-table, and flush pending layout tokens at EOF.],
    [6], [Parser], [Consume the scanner token stream, validate the JIT grammar, and build the final parser report.],
    [7], [Program], [Print the token listing or parser report and exit nonzero when lexical, syntax, semantic, or missing-kernel errors are present.],
  ),
  kind: table,
  caption: [Algorithmic description of scanner and parser processing.],
) <tab-processing-algorithm>

== Symbol table and runtime data model <symbol-and-report-data-model>

The scanner symbol table is a first-class design requirement, not a side effect of the implementation. It reports what the scanner actually emitted for JIT islands, while avoiding duplicate rows for repeated lexemes.

#figure(
  table(
    columns: (1fr, 1.25fr, 2.25fr),
    inset: 5pt,
    align: left,
    [*Structure*], [*Required fields*], [*Design responsibility*],
    [`Symbol`], [`id`, `token_name`, `lexeme`, `first_line`], [Stores one scanner symbol-table row for each unique emitted non-layout token/lexeme pair. `id` is assigned in first-seen order starting at 1.],
    [`SymbolTable`], [bounded array of `Symbol`, count, capacity 4096], [Provides deterministic insertion order and a fixed memory bound for the standalone scanner report.],
    [`PendingToken`], [`token_id`, `lexeme`, `line`], [Represents a queued synthetic layout token or any token that must be returned after a previous scan action discovered multiple parser-visible tokens.],
    [`PendingQueue`], [circular FIFO, capacity 256], [Allows layout processing to queue `NEWLINE`, `INDENT`, and multiple `DEDENT` tokens while the public scanner API still returns one token per call.],
    [`IndentStack`], [integer widths, top index, capacity 64], [Stores valid indentation levels so dedents can be translated into one `DEDENT` per popped block and invalid dedents can be detected.],
    [`ScannerControlState`], [`mode`, `paren_depth`, `saw_jit_body_indent`, `host_column`, `host_line_has_only_space`], [Coordinates host skipping, JIT activation, delimiter-depth newline suppression, and return to host mode after top-level dedent.],
    [`ParameterInfo`], [`name`, `is_constexpr`], [Parser report structure used after scanning to record formal parameters and exact `tl.constexpr` annotations.],
    [`KernelInfo`], [`name`, `line`, parameters, locals, `tl.*` calls], [Parser report structure used after scanning to summarize accepted kernels.],
  ),
  kind: table,
  caption: [Runtime data structures required by scanner control, scanner reporting, and parser reporting.],
) <tab-data-model>

The scanner inserts symbols only for tokens emitted by token-recognition rules. Synthetic layout tokens are returned to the parser or printed in the token stream but are not inserted into the scanner symbol table. The symbol key is the pair `(token_name, lexeme)`, so the same lexeme may appear in separate rows if it is emitted under different token names, and repeated occurrences of the same pair keep the first source line.

```text
record_symbol(token_name, lexeme, line):
    if lexeme is empty:
        return 0

    for each existing symbol in first-seen order:
        if symbol.token_name == token_name and symbol.lexeme == lexeme:
            return symbol.id

    if the symbol table is full:
        return -1

    id = symbol_count + 1
    append Symbol(id, token_name, lexeme, line)
    return id
```

The concrete scanner code in Appendix A mirrors this algorithm in `compiler/triton.l` lines 63--86: it rejects empty lexemes, searches existing rows before appending, assigns `symbol_count + 1` as the stable ID, and stores the line passed by `emit_token`. A developer implementing the scanner may use a linear array as specified here or replace it with a hash table, but the observable behavior--unique `(token, lexeme)` rows in first-seen order--must remain unchanged.

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
    [Rules section], [Implements the token DFA transitions from @scanner-automata and @finite-transition-table, plus the scanner-control behavior from @scanner-control-algorithms.], [Uses `INITIAL` rules for host skipping and top-level JIT activation, `JIT` rules for strings, numbers, identifiers, keywords, operators, delimiters, comments, layout, EOF cleanup, and unsupported-character diagnostics. Rule order preserves longest-match behavior for multi-character operators and literals.],
    [User code section], [Provides executable entry points around the scanner core.], [Defines `triton_next_token` for Bison builds, `next_token` for standalone queue-aware scanning, and `main` for command-line scanner output and final symbol-table printout.],
  ),
  kind: table,
  caption: [Implementation explanation of the three Lex source sections.],
) <tab-flex-section-implementation>

Appendix A shows the same mapping in source form: named regular expressions appear in `compiler/triton.l` lines 282--290; token actions in lines 324--406 return the IDs catalogued in @tab-token-ids; the `INITIAL` and `JIT` rules in lines 294--320 and 322--478 implement the mode-control transitions; the newline branches in lines 408--449 correspond to @tab-control-transition-table; and `add_symbol` in lines 63--86 implements the symbol-table insertion described in @symbol-and-report-data-model.

== Runtime behavior <runtime-behavior>

The scanner emits tokens only for JIT islands. This behavior is visible in the vector-add fixture: the file header and imports are skipped, and the first emitted token is the decorator's `AT` token. Layout is represented with synthetic `NEWLINE`, `INDENT`, and `DEDENT` tokens, including final dedents at end-of-file.

The parser accepts multiple kernels in one file. Duplicate kernel names are treated as semantic errors. Files with no top-level JIT kernels produce a report stating that no kernels were found and exit with status 1. This makes the command-line tool useful both for positive validation and for CI-style failure checks.

== Known boundaries <known-boundaries>

The implementation is intentionally a compiler front end for the reduced Triton kernel subset, not a Python interpreter or full Python parser. Host Python is ignored unless it contains a top-level JIT decorator. General Python constructs outside the modeled subset may be skipped in host mode or rejected in JIT mode. The parser report is also intentionally structural: it does not execute kernels, infer types, build an AST, or check numerical correctness.

= Verification and Validation <verification-and-validation>

== Test set <test-set>

The automated tests compile and run the scanner and parser against fixtures under `tests/compiler/fixtures`. Valid fixtures cover code samples from real Triton codebases: vector addition, softmax, dropout, matrix multiplication, flash attention, and layer normalization. Invalid fixtures cover malformed syntax, duplicate kernel names, and missing top-level JIT decorators. Additional in-memory end-to-end tests exercise comments at the beginning of kernel bodies, embedded `tl.arange` calls, `tl.dot` in augmented assignment, return annotations, nested suites, bitwise and shift expressions, multiple decorators, numeric forms, and string forms.

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
#let flex-definition-source = triton-l-lines.slice(0, 291).join("\n")
#let flex-rules-source = triton-l-lines.slice(292, 479).join("\n")
#let flex-user-source = triton-l-lines.slice(480).join("\n")

== Definition section <appendix-flex-definition>

#source-listing(
  flex-definition-source,
  start: 1,
  caption: [Definition section of `compiler/triton.l` with original line numbers.],
) <fig-flex-definition-source>

== Rules section <appendix-flex-rules>

#source-listing(
  flex-rules-source,
  start: 293,
  caption: [Rules section of `compiler/triton.l` with original line numbers.],
) <fig-flex-rules-source>

== User code section <appendix-flex-user-code>

#source-listing(
  flex-user-source,
  start: 481,
  caption: [User code section of `compiler/triton.l` with original line numbers.],
) <fig-flex-user-source>


= References <references>

[1] Free Software Foundation, "Bison: The Yacc-compatible Parser Generator," GNU Project, Sep. 11, 2021. [Online]. Available: https://www.gnu.org/software/bison/manual/. Accessed: Jun. 10, 2026.

[2] V. Paxson, W. Estes, and J. Millaway, "The flex Manual," flex, version 2.6.2, Oct. 22, 2016. [Online]. Available: https://westes.github.io/flex/manual/. Accessed: Jun. 10, 2026.

[3] Python Software Foundation, "Lexical analysis," _The Python Language Reference_, Python Documentation. [Online]. Available: https://docs.python.org/3/reference/lexical_analysis.html. Accessed: Jun. 10, 2026.

[4] Triton Contributors, "triton.jit," Triton Documentation. [Online]. Available: https://triton-lang.org/main/python-api/generated/triton.jit.html. Accessed: Jun. 10, 2026.
