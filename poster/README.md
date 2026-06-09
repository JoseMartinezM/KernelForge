# KernelForge poster

Self-contained source for the final KernelForge poster.

Build locally from this directory with:

```bash
nix shell nixpkgs#texliveFull -c lualatex -interaction=nonstopmode -halt-on-error poster.tex
nix shell nixpkgs#texliveFull -c lualatex -interaction=nonstopmode -halt-on-error poster.tex
```

The included `poster.pdf` was built from `poster.tex` with the same LuaLaTeX flow.
