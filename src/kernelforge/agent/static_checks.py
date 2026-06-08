from __future__ import annotations

import ast

from .schemas import StaticCheckResult
from kernelforge.benchmark.llm_results import code_metrics
from kernelforge.benchmark.semantic_checker import check_kernel


def _is_triton_jit_decorator(node: ast.expr) -> bool:
    if isinstance(node, ast.Call):
        node = node.func
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "jit"
        and isinstance(node.value, ast.Name)
        and node.value.id == "triton"
    )


def jit_only_source(source: str) -> str:
    """Return source containing only top-level @triton.jit functions.

    The project semantic checker intentionally scans functions heuristically; in
    full generated modules, wrappers can look kernel-like because they reference
    names such as BLOCK_SIZE. Restricting semantic warnings to real JIT blocks
    avoids false positives without weakening syntax or anti-cheat metrics.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    lines = source.splitlines()
    blocks = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if not any(_is_triton_jit_decorator(decorator) for decorator in node.decorator_list):
            continue
        start_line = min(decorator.lineno for decorator in node.decorator_list) - 1
        end_line = node.end_lineno or node.lineno
        blocks.append("\n".join(lines[start_line:end_line]))
    return "\n\n".join(blocks) if blocks else source


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        prefix = _call_name(func.value)
        return f"{prefix}.{func.attr}" if prefix else func.attr
    return None


def _is_triton_jit_function(node: ast.AST) -> bool:
    return isinstance(node, ast.FunctionDef) and any(
        _is_triton_jit_decorator(decorator) for decorator in node.decorator_list
    )


def torch_call_names(source: str) -> list[str]:
    """Return dotted torch.* calls found in source, best effort."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name and name.startswith("torch."):
            calls.append(name)
    return calls


def static_guardrail_flags(source: str) -> list[str]:
    """Return best-effort flags for common Triton API hallucinations.

    These are advisory diagnostics for ledgers and repair prompts. They should
    not reject code by themselves because a public test failure is often more
    useful than an overzealous static filter.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    flags: list[str] = []
    jit_function_ids = {id(node) for node in tree.body if _is_triton_jit_function(node)}
    constexpr_by_function: dict[int, set[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not _is_triton_jit_function(node):
            continue
        constexpr_names: set[str] = set()
        for arg in node.args.args + node.args.kwonlyargs:
            annotation = arg.annotation
            if annotation is not None and _call_name(annotation) == "tl.constexpr":
                constexpr_names.add(arg.arg)
        constexpr_by_function[id(node)] = constexpr_names

    parent_jit_stack: list[bool] = []
    constexpr_stack: list[set[str]] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            is_jit = id(node) in jit_function_ids
            parent_jit_stack.append(is_jit)
            constexpr_stack.append(constexpr_by_function.get(id(node), set()))
            self.generic_visit(node)
            constexpr_stack.pop()
            parent_jit_stack.pop()

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            name = _call_name(node.func)
            in_jit = any(parent_jit_stack)
            if any(keyword.arg == "stream" for keyword in node.keywords):
                flags.append("kernel_launch_stream_arg")
            if name in {"tl.program_id", "tl.arange", "tl.load", "tl.store"} and not in_jit:
                flags.append("triton_intrinsic_outside_jit")
            if name in {"tl.zeros", "tl.full"} and node.args:
                self._check_tl_vector_shape(node.args[0])
            if name == "tl.arange" and len(node.args) >= 2:
                self._check_compile_time_shape_expr(node.args[1])
            self.generic_visit(node)

        def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802
            if isinstance(node.value, ast.Attribute) and node.value.attr == "stride":
                flags.append("stride_method_subscript")
            self.generic_visit(node)

        def _check_tl_vector_shape(self, shape_node: ast.AST) -> None:
            if not isinstance(shape_node, (ast.List, ast.Tuple)):
                self._check_compile_time_shape_expr(shape_node)
                return
            for element in shape_node.elts:
                self._check_compile_time_shape_expr(element)

        def _check_compile_time_shape_expr(self, expr: ast.AST) -> None:
            if isinstance(expr, ast.Constant) and isinstance(expr.value, int):
                return
            if isinstance(expr, ast.Name) and expr.id in (constexpr_stack[-1] if constexpr_stack else set()):
                return
            flags.append("runtime_triton_vector_shape")

    Visitor().visit(tree)
    return list(dict.fromkeys(flags))


def has_incomplete_marker(source: str) -> bool:
    """Return True for actual TODOs or Python pass statements, not prose words."""
    if "TODO" in source:
        return True
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Pass) for node in ast.walk(tree))


def run_static_checks(source: str) -> StaticCheckResult:
    """Run CPU-only candidate checks used before GPU evaluation."""
    metrics = code_metrics(source)
    semantic_warnings = check_kernel(jit_only_source(source))
    torch_calls = torch_call_names(source)

    flags = [
        flag
        for flag in metrics["flags"]
        if flag != "incomplete_marker" or has_incomplete_marker(source)
    ]
    if semantic_warnings:
        flags.append("semantic_warnings")
    if metrics["triton_jit_count"] == 0:
        flags.append("no_triton_jit")
    if "tl.load" not in source or "tl.store" not in source:
        flags.append("missing_tl_load_or_store")
    if torch_calls:
        flags.append("torch_calls_present")
    flags.extend(static_guardrail_flags(source))

    deduped_flags = list(dict.fromkeys(flags))
    return StaticCheckResult(
        syntax_ok=bool(metrics["syntax_ok"]),
        syntax_error=metrics["syntax_error"],
        content_chars=int(metrics["content_chars"]),
        content_lines=int(metrics["content_lines"]),
        starts_with_import=bool(metrics["starts_with_import"]),
        markdown_fence_count=int(metrics["markdown_fence_count"]),
        triton_jit_count=int(metrics["triton_jit_count"]),
        torch_call_count=int(metrics["torch_call_count"]),
        torch_calls=torch_calls,
        semantic_warnings=semantic_warnings,
        flags=deduped_flags,
        flags_text=", ".join(deduped_flags),
    )
