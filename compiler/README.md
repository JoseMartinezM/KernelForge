# Compiler

Home for the course LEX/YACC parser once implementation starts.

Intended scope:

- syntactic validation of Triton/Python-like source;
- fixtures that represent accepted and rejected syntax;
- generated parser artifacts only if they are needed for reproducible builds.

This parser should stay syntactic. Semantic checks such as whether `tl.foo` is a
real Triton export belong in the constrained-generation grammar/tooling, not in
the LEX/YACC assignment layer.
