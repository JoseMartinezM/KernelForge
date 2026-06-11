# Oral Exam Answers - Triton GPU Direct Syntax Translator

This document summarizes the compiler project in English for oral-exam review.
The current implementation lives in `compiler/triton.l` and `compiler/triton.y`
and uses Flex/Bison in C.

## 1. General Compiler Design

**What is the main goal of the compiler?**

The goal is to validate Python-style source code that contains Triton GPU
kernels. The scanner tokenizes the source, the parser checks that the token
stream matches the grammar, and the semantic actions build a report that says
which functions were found, which ones are kernels, which parameters are
`tl.constexpr`, which local variables were assigned, and which `tl.*` API calls
were detected.

**What does "Triton-oriented code" mean in this project?**

It means Python functions that follow the Triton kernel programming pattern:
they are usually decorated with `@triton.jit`, receive pointer-like arguments,
use `tl.program_id` to identify the current program instance, compute offsets
with `tl.arange`, and access memory with `tl.load` and `tl.store`.

**Is this a full compiler, an interpreter, or a Direct Syntax Translator?**

It is a **Direct Syntax Translator**. It does not generate machine code and it
does not execute the input program. Instead, it runs semantic actions during
parsing and produces a validation report directly from those reductions.

**What are the main phases?**

1. **Lexical analysis** in `triton.l`: converts characters into tokens.
2. **Syntax analysis** in `triton.y`: recognizes grammar rules.
3. **Direct translation** in Bison semantic actions: fills the function report.
4. **Reporting**: prints kernels, parameters, locals, API calls, warnings, and
   errors.

**What was the hardest part?**

The hardest part is Python-style indentation. Flex reads characters, but Bison
works best with explicit tokens. The scanner therefore converts indentation
changes into `INDENT` and `DEDENT` tokens, while ignoring newlines inside
parentheses, brackets, and braces.

## 2. Lexical Analysis

**What does the scanner do?**

The scanner recognizes comments, strings, numbers, identifiers, keywords,
operators, delimiters, newlines, and indentation changes. It emits tokens such
as `DEF`, `NAME`, `STRING`, `NUMBER_INT`, `COLON`, `LPAREN`, `NEWLINE`,
`INDENT`, and `DEDENT`.

**How are identifiers and keywords distinguished?**

The scanner first matches identifier-shaped text with a regular expression.
Then it compares the matched text against Python keywords. For example, `def`
becomes `DEF`, while `def_kernel` remains `NAME`.

**Are there Triton-specific lexical tokens?**

No. Triton APIs such as `tl.load` and `tl.store` are lexed as ordinary names,
dots, and calls. The parser and semantic actions later reconstruct strings like
`tl.load(...)` and classify them as Triton API usage.

**How does indentation work?**

The scanner keeps an indentation stack that starts at level `0`. After each
real newline, it counts the leading spaces or tabs on the next line.

- If the new level is larger than the stack top, it emits `NEWLINE` and
  `INDENT`.
- If the new level is smaller, it emits `NEWLINE` and enough `DEDENT` tokens to
  return to the matching level.
- If the level is the same, it emits only `NEWLINE`.

At end of file, it emits any remaining `DEDENT` tokens so all open blocks are
closed.

**Why is there a pending-token queue?**

A single newline can produce more than one logical token, such as `NEWLINE`
plus several `DEDENT` tokens. Flex actions can return only one token at a time,
so the scanner stores extra tokens in a queue and returns them on later calls.

## 3. Syntax Analysis and Grammar

**What grammar does the parser implement?**

The parser implements a practical subset of Python used by Triton kernels:
function definitions, decorators, imports, assignments, augmented assignments,
returns, assertions, `if`/`elif`/`else`, `for`, `while`, `with`, classes,
`try`/`except`/`finally`, calls, indexing, slices, literals, and expressions
with Python-like precedence.

**How are blocks represented?**

Blocks are represented with explicit indentation tokens:

```text
suite -> nl_seq INDENT stmt_seq DEDENT
```

The scanner creates those tokens from whitespace, so the parser does not need
to inspect raw spaces.

**Why does the grammar avoid left recursion?**

The course requirement asks for a grammar without left recursion. Binary
expression rules therefore use the standard transformation:

```text
A -> A alpha | beta

becomes:

A  -> beta A'
A' -> alpha A' | epsilon
```

This pattern appears in expression layers such as `sum`, `term`, `shift`,
`bitand`, `bitxor`, and `bitor`.

**How does the parser recognize a function?**

The main rule is `funcdef`. It recognizes either decorated functions or plain
functions. A mid-rule action runs right after `DEF NAME` so the current function
is registered before the parameter list is parsed.

**Why is the mid-rule action important?**

Parameters are added while `param_seq` is being reduced. If the function were
not declared until the end of `funcdef`, `add_param()` would not know which
function should receive those parameters.

## 4. Semantic Actions

**What semantic information is collected?**

For each function, the translator stores:

- function name
- declaration line
- whether `@triton.jit` was detected
- parameters and annotations
- whether each relevant parameter is `tl.constexpr`
- assigned local variables
- detected Triton API calls

**How is `@triton.jit` detected?**

Decorator syntax is reconstructed as text by the decorator grammar. The parser
then checks whether that text contains `triton.jit`. If it does, the function
is reported as a kernel; otherwise, the translator emits a warning.

**How are `tl.*` calls detected?**

The `primary` rule combines an atom with trailers such as `.name`, `(...)`, and
`[...]`. That lets the parser reconstruct strings like `tl.program_id(...)` and
`tl.load(...)`. When a reconstructed expression starts with `tl.` and includes
a call trailer, it is recorded as a Triton API call.

**Does the translator build an AST?**

No. The report is built directly from semantic actions during parsing. This is
why the project is a Direct Syntax Translator instead of an AST-based compiler.

## 5. Error Handling

**What lexical errors can be reported?**

The scanner reports unrecognized characters as lexical errors with a line
number.

**What syntax errors can be reported?**

Bison calls `yyerror()` when the token stream does not match the grammar. The
message includes the current line number and is also stored in the final report.

**What semantic errors can be reported?**

The current semantic table reports duplicate function names. A duplicate is an
error because the function name is used as the stable identity in the report.

**Are missing `@triton.jit` decorators errors?**

They are warnings, not hard errors. This lets helper functions appear in a file
beside real kernels.

## 6. Example Flow

For this source:

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    tl.store(out_ptr + pid, 0.0)
```

The scanner emits tokens for the decorator, `def`, function name, parameters,
colon, newline, indent, body statements, and dedent. The parser reduces those
tokens into a `funcdef` and a `suite`. Semantic actions record `add_kernel` as
a kernel, mark `BLOCK_SIZE` as constexpr, and record calls to `tl.program_id`
and `tl.store`.

## 7. Current Limitations

- The translator validates syntax and extracts metadata; it does not execute
  kernels.
- It does not generate Triton source or GPU machine code.
- It does not perform full Python name resolution.
- It detects `tl.*` calls syntactically, not through type checking.
- It does not prove memory safety, race freedom, or shape correctness.

## 8. Commands to Demonstrate

```bash
cd compiler
make scanner
./triton_scanner ../tests/compiler/fixtures/valid/vector_add.py

make compiler
./triton_compiler ../tests/compiler/fixtures/valid/vector_add.py
./triton_compiler ../tests/compiler/fixtures/invalid/bad_syntax.py
```

Use the scanner output to explain tokenization and indentation. Use the
compiler output to explain parsing, semantic actions, warnings, and errors.
