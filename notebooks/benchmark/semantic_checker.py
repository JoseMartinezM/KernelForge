"""Semantic checker for Triton kernels.

Implements `check_kernel(source: str) -> list[str]`.
Returns a list of warning strings (never raises).
"""
from __future__ import annotations

import ast
from typing import List


def _is_tl_attr_call(node: ast.Call, name: str) -> bool:
    """Return True if node is a call to `tl.<name>(...)`."""
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id == "tl" and func.attr == name
    return False


def _has_keyword_arg(call: ast.Call, kw_name: str) -> bool:
    for kw in call.keywords:
        if kw.arg == kw_name:
            return True
    return False


def _annotation_is_constexpr(annotation: ast.AST | None) -> bool:
    if annotation is None:
        return False
    # Case: BLOCK_SIZE: tl.constexpr
    if isinstance(annotation, ast.Attribute) and isinstance(annotation.value, ast.Name):
        return annotation.value.id == "tl" and annotation.attr == "constexpr"
    # Case: BLOCK_SIZE: "tl.constexpr"
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return "tl.constexpr" in annotation.value
    # Fallback: name equals 'tl.constexpr' (uncommon)
    if isinstance(annotation, ast.Name):
        return annotation.id == "tl.constexpr"
    return False


def check_kernel(source: str) -> List[str]:
    """Analyze a Python/Triton source string and return semantic warnings.

    The function never raises; on parse errors it returns a single warning.
    """
    warnings: List[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ["Could not parse source code; semantic checks skipped."]

    for node in tree.body:
        # Look for function definitions that look like Triton kernels.
        if not isinstance(node, ast.FunctionDef):
            continue

        func: ast.FunctionDef = node

        # Scan function body for tl.load, tl.store, tl.program_id, and BLOCK_SIZE usage.
        has_load = False
        has_store = False
        has_program_id = False
        uses_block_size = False

        # Collect variable names assigned from tl.arange(...) to detect vectorized offsets
        arange_vars: set[str] = set()
        for node_assign in func.body:
            # handle simple assignments and annotated assigns
            if isinstance(node_assign, ast.Assign):
                value = node_assign.value
                for subval in ast.walk(value):
                    if isinstance(subval, ast.Call) and _is_tl_attr_call(subval, "arange"):
                        for target in node_assign.targets:
                            if isinstance(target, ast.Name):
                                arange_vars.add(target.id)
            elif isinstance(node_assign, ast.AnnAssign):
                value = node_assign.value
                if value is not None:
                    for subval in ast.walk(value):
                        if isinstance(subval, ast.Call) and _is_tl_attr_call(subval, "arange"):
                            target = node_assign.target
                            if isinstance(target, ast.Name):
                                arange_vars.add(target.id)

        def _arg_uses_arange(call: ast.Call) -> bool:
            # Return True if the first positional arg of the call uses tl.arange
            if not call.args:
                return False
            arg = call.args[0]
            for sub in ast.walk(arg):
                if isinstance(sub, ast.Call) and _is_tl_attr_call(sub, "arange"):
                    return True
                if isinstance(sub, ast.Name) and sub.id in arange_vars:
                    return True
            return False

        for sub in ast.walk(func):
            if isinstance(sub, ast.Call):
                if _is_tl_attr_call(sub, "load"):
                    has_load = True
                    # Only warn if this load is vectorized (uses tl.arange-derived offsets)
                    if not _has_keyword_arg(sub, "mask") and _arg_uses_arange(sub):
                        warnings.append(f"tl.load call missing mask for vectorized load in function {func.name} (line {sub.lineno})")
                elif _is_tl_attr_call(sub, "store"):
                    has_store = True
                    if not _has_keyword_arg(sub, "mask") and _arg_uses_arange(sub):
                        warnings.append(f"tl.store call missing mask for vectorized store in function {func.name} (line {sub.lineno})")
                elif _is_tl_attr_call(sub, "program_id"):
                    has_program_id = True
            if isinstance(sub, ast.Name) and sub.id == "BLOCK_SIZE":
                uses_block_size = True

        # Heuristic: treat a function as a kernel if it uses tl.load/tl.store/program_id or BLOCK_SIZE.
        is_kernel_candidate = has_load or has_store or has_program_id or uses_block_size

        if not is_kernel_candidate:
            continue

        # Check decorator @triton.jit presence
        has_triton_jit = False
        for dec in func.decorator_list:
            if isinstance(dec, ast.Attribute) and isinstance(dec.value, ast.Name):
                if dec.value.id == "triton" and dec.attr == "jit":
                    has_triton_jit = True
                    break

        if not has_triton_jit:
            warnings.append(f"Function {func.name} missing @triton.jit decorator")

        # Check tl.program_id presence
        if not has_program_id:
            warnings.append(f"Function {func.name} missing tl.program_id call")

        # Check BLOCK_SIZE constexpr annotation if BLOCK_SIZE used
        if uses_block_size:
            # Find arg named BLOCK_SIZE
            has_block_param = False
            block_param_constexpr = False
            for arg in func.args.args:
                if arg.arg == "BLOCK_SIZE":
                    has_block_param = True
                    if _annotation_is_constexpr(arg.annotation):
                        block_param_constexpr = True
                    break
            if not (has_block_param and block_param_constexpr):
                warnings.append(f"BLOCK_SIZE used in function {func.name} without tl.constexpr annotation in signature")

    return warnings
