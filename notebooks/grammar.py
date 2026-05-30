import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import ast
    import io
    import math
    import tokenize
    from collections import Counter
    from pathlib import Path

    import altair as alt
    import marimo as mo

    from kernelforge.benchmark import load_jsonl

    return Counter, Path, alt, ast, io, load_jsonl, math, mo, tokenize


@app.cell
def _(Path, load_jsonl):
    notebook_dir = Path(__file__).resolve().parent
    data_dir = notebook_dir / "data"
    jsonl_paths = sorted(data_dir.glob("*.jsonl"))

    inference_runs = []
    for _path in jsonl_paths:
        for _row in load_jsonl(_path):
            _row = dict(_row)
            _row["result_path"] = str(_path)
            inference_runs.append(_row)

    {
        "jsonl_files": [p.name for p in jsonl_paths],
        "responses": len(inference_runs),
        "models": sorted({row.get("model_label") or row.get("model") for row in inference_runs}),
        "cases": len({row.get("entry_file") for row in inference_runs}),
    }
    return (inference_runs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        "\n".join(
            [
                "# Triton constrained-grammar discovery",
                "",
                "Recent-thread takeaways for the starting grammar surface:",
                "",
                "- Triton lowers a Python AST subset in `@triton.jit` kernels. The useful baseline includes",
                "  assignments, arithmetic expressions, comparisons, boolean expressions, `if`, bounded",
                "  `for range(...)` / `tl.static_range(...)`, calls, attributes, subscripts, tuples/lists,",
                "  and returns in helper functions.",
                "- The compiler technically has more surface (`while`, `with`, walrus, limited list",
                "  comprehensions), but the first constrained grammar should be narrower than Triton's full",
                "  implementation surface.",
                "- Host Python and device Triton should be separated: wrappers launch kernels and allocate",
                "  outputs; device functions use `tl.*` primitives and pointer arithmetic.",
                "- Because xgrammar/llguidance-style decoding cannot conveniently synthesize virtual",
                "  `INDENT`/`DEDENT` tokens, indentation should be represented with explicit 4-space literals",
                "  and a bounded nesting depth chosen from observed completions.",
            ]
        )
    )
    return


@app.cell
def _(Counter, ast, inference_runs, io, math, tokenize):
    def _call_name(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = _call_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        if isinstance(node, ast.Subscript):
            base = _call_name(node.value)
            return f"{base}[...]" if base else "<subscript>"
        return type(node).__name__


    def _decorator_name(node):
        if isinstance(node, ast.Call):
            return _call_name(node.func)
        return _call_name(node)


    def _is_triton_jit_function(node):
        return any(
            _decorator_name(_decorator) in {"triton.jit", "jit"}
            for _decorator in node.decorator_list
        )


    def _max_indent(content):
        _max_line_spaces = 0
        _max_block_spaces = 0
        _indent_widths = Counter()
        _non_four_space_lines = 0
        _tab_indented_lines = 0

        for _line in content.splitlines():
            if not _line.strip():
                continue
            _prefix = _line[: len(_line) - len(_line.lstrip(" \t"))]
            if "\t" in _prefix:
                _tab_indented_lines += 1
            _spaces = len(_prefix.expandtabs(4))
            _max_line_spaces = max(_max_line_spaces, _spaces)
            if _spaces:
                _indent_widths[_spaces] += 1
                if _spaces % 4 != 0:
                    _non_four_space_lines += 1

        try:
            for _token in tokenize.generate_tokens(io.StringIO(content).readline):
                if _token.type == tokenize.INDENT:
                    _max_block_spaces = max(_max_block_spaces, len(_token.string.expandtabs(4)))
        except tokenize.TokenError:
            pass

        return {
            "max_indent_spaces": _max_block_spaces,
            "max_indent_levels": math.ceil(_max_block_spaces / 4) if _max_block_spaces else 0,
            "max_line_indent_spaces": _max_line_spaces,
            "max_line_indent_levels": math.ceil(_max_line_spaces / 4) if _max_line_spaces else 0,
            "indent_widths": _indent_widths,
            "non_four_space_indent_lines": _non_four_space_lines,
            "tab_indented_lines": _tab_indented_lines,
        }


    def _collect_ast_features(tree):
        _all_nodes = Counter()
        _jit_nodes = Counter()
        _host_nodes = Counter()
        _statements = Counter()
        _jit_statements = Counter()
        _calls = Counter()
        _jit_calls = Counter()
        _decorators = Counter()
        _imports = Counter()
        _binops = Counter()
        _boolops = Counter()
        _unaryops = Counter()
        _compares = Counter()
        _for_iters = Counter()
        _kernel_names = []

        _jit_descendant_ids = set()
        for _node in ast.walk(tree):
            if isinstance(_node, ast.FunctionDef) and _is_triton_jit_function(_node):
                _kernel_names.append(_node.name)
                for _descendant in ast.walk(_node):
                    _jit_descendant_ids.add(id(_descendant))

        for _node in ast.walk(tree):
            _name = type(_node).__name__
            _all_nodes[_name] += 1
            _inside_jit = id(_node) in _jit_descendant_ids
            if _inside_jit:
                _jit_nodes[_name] += 1
            else:
                _host_nodes[_name] += 1

            if isinstance(_node, ast.stmt):
                _statements[_name] += 1
                if _inside_jit:
                    _jit_statements[_name] += 1

            if isinstance(_node, ast.Call):
                _call = _call_name(_node.func)
                _calls[_call] += 1
                if _inside_jit:
                    _jit_calls[_call] += 1
            elif isinstance(_node, ast.FunctionDef):
                for _decorator in _node.decorator_list:
                    _decorators[_decorator_name(_decorator)] += 1
            elif isinstance(_node, ast.Import):
                for _alias in _node.names:
                    _imports[_alias.name] += 1
            elif isinstance(_node, ast.ImportFrom):
                _module = _node.module or ""
                for _alias in _node.names:
                    _imports[f"{_module}.{_alias.name}" if _module else _alias.name] += 1
            elif isinstance(_node, ast.BinOp):
                _binops[type(_node.op).__name__] += 1
            elif isinstance(_node, ast.BoolOp):
                _boolops[type(_node.op).__name__] += 1
            elif isinstance(_node, ast.UnaryOp):
                _unaryops[type(_node.op).__name__] += 1
            elif isinstance(_node, ast.Compare):
                for _op in _node.ops:
                    _compares[type(_op).__name__] += 1
            elif isinstance(_node, ast.For):
                _for_iters[_call_name(_node.iter.func) if isinstance(_node.iter, ast.Call) else type(_node.iter).__name__] += 1

        return {
            "all_nodes": _all_nodes,
            "jit_nodes": _jit_nodes,
            "host_nodes": _host_nodes,
            "statements": _statements,
            "jit_statements": _jit_statements,
            "calls": _calls,
            "jit_calls": _jit_calls,
            "decorators": _decorators,
            "imports": _imports,
            "binops": _binops,
            "boolops": _boolops,
            "unaryops": _unaryops,
            "compares": _compares,
            "for_iters": _for_iters,
            "kernel_names": _kernel_names,
        }


    response_features = []
    for _index, _run in enumerate(inference_runs, start=1):
        _content = _run.get("content") or ""
        _indent = _max_indent(_content)
        _row = {
            "row": _index,
            "entry_file": _run.get("entry_file"),
            "model": _run.get("model"),
            "model_label": _run.get("model_label") or _run.get("model"),
            "content_lines": len(_content.splitlines()),
            "content_chars": len(_content),
            "syntax_ok": True,
            "syntax_error": None,
            **_indent,
        }

        try:
            _tree = ast.parse(_content)
        except SyntaxError as _exc:
            _row.update(
                {
                    "syntax_ok": False,
                    "syntax_error": f"{_exc.msg} at {_exc.lineno}:{_exc.offset}",
                    "features": {},
                    "kernel_count": 0,
                    "tl_call_count": 0,
                    "torch_call_count": _content.count("torch."),
                    "triton_call_count": _content.count("triton."),
                }
            )
        else:
            _features = _collect_ast_features(_tree)
            _calls = _features["calls"]
            _row.update(
                {
                    "features": _features,
                    "kernel_count": len(_features["kernel_names"]),
                    "tl_call_count": sum(_count for _name, _count in _calls.items() if _name.startswith("tl.")),
                    "torch_call_count": sum(_count for _name, _count in _calls.items() if _name.startswith("torch.")),
                    "triton_call_count": sum(_count for _name, _count in _calls.items() if _name.startswith("triton.")),
                }
            )
        response_features.append(_row)

    response_features[0]
    return (response_features,)


@app.cell
def _(Counter, response_features):
    def _counter_table(counter, response_counter=None, *, limit=80, name="feature"):
        _rows = []
        for _feature, _count in counter.most_common(limit):
            _row = {name: _feature, "count": _count}
            if response_counter is not None:
                _row["responses"] = response_counter.get(_feature, 0)
            _rows.append(_row)
        return _rows


    total_responses = len(response_features)
    syntax_ok_responses = [row for row in response_features if row["syntax_ok"]]

    node_counts = Counter()
    node_response_counts = Counter()
    jit_node_counts = Counter()
    jit_node_response_counts = Counter()
    statement_counts = Counter()
    statement_response_counts = Counter()
    jit_statement_counts = Counter()
    call_counts = Counter()
    call_response_counts = Counter()
    jit_call_counts = Counter()
    import_counts = Counter()
    decorator_counts = Counter()
    binop_counts = Counter()
    boolop_counts = Counter()
    unaryop_counts = Counter()
    compare_counts = Counter()
    for_iter_counts = Counter()

    for _row in syntax_ok_responses:
        _features = _row["features"]
        node_counts.update(_features["all_nodes"])
        node_response_counts.update(_features["all_nodes"].keys())
        jit_node_counts.update(_features["jit_nodes"])
        jit_node_response_counts.update(_features["jit_nodes"].keys())
        statement_counts.update(_features["statements"])
        statement_response_counts.update(_features["statements"].keys())
        jit_statement_counts.update(_features["jit_statements"])
        call_counts.update(_features["calls"])
        call_response_counts.update(_features["calls"].keys())
        jit_call_counts.update(_features["jit_calls"])
        import_counts.update(_features["imports"])
        decorator_counts.update(_features["decorators"])
        binop_counts.update(_features["binops"])
        boolop_counts.update(_features["boolops"])
        unaryop_counts.update(_features["unaryops"])
        compare_counts.update(_features["compares"])
        for_iter_counts.update(_features["for_iters"])

    grammar_feature_names = [
        "Assign",
        "AnnAssign",
        "AugAssign",
        "NamedExpr",
        "If",
        "IfExp",
        "For",
        "While",
        "With",
        "Return",
        "Assert",
        "Pass",
        "Break",
        "Continue",
        "Try",
        "Raise",
        "ClassDef",
        "Lambda",
        "ListComp",
        "DictComp",
        "SetComp",
        "GeneratorExp",
        "Subscript",
        "Slice",
        "Attribute",
        "Tuple",
        "List",
        "Dict",
        "Set",
        "BoolOp",
        "Compare",
        "BinOp",
        "UnaryOp",
        "Call",
    ]
    grammar_feature_rows = [
        {
            "feature": _name,
            "responses": node_response_counts.get(_name, 0),
            "responses_%": round(100 * node_response_counts.get(_name, 0) / total_responses, 1)
            if total_responses
            else 0,
            "total_count": node_counts.get(_name, 0),
            "jit_responses": jit_node_response_counts.get(_name, 0),
            "jit_count": jit_node_counts.get(_name, 0),
        }
        for _name in grammar_feature_names
    ]

    aggregate_summary = {
        "responses": total_responses,
        "syntax_ok": len(syntax_ok_responses),
        "syntax_errors": total_responses - len(syntax_ok_responses),
        "responses_with_jit_kernel": sum(row.get("kernel_count", 0) > 0 for row in response_features),
        "total_jit_kernels": sum(row.get("kernel_count", 0) for row in response_features),
        "responses_with_tl_calls": sum(row.get("tl_call_count", 0) > 0 for row in response_features),
        "responses_with_torch_calls": sum(row.get("torch_call_count", 0) > 0 for row in response_features),
        "responses_with_triton_calls": sum(row.get("triton_call_count", 0) > 0 for row in response_features),
    }

    statement_rows = _counter_table(statement_counts, statement_response_counts, name="statement")
    jit_statement_rows = _counter_table(jit_statement_counts, name="jit_statement")
    call_rows = _counter_table(call_counts, call_response_counts, limit=120, name="call")
    jit_call_rows = _counter_table(jit_call_counts, limit=120, name="jit_call")
    import_rows = _counter_table(import_counts, limit=80, name="import")
    decorator_rows = _counter_table(decorator_counts, limit=40, name="decorator")
    operator_rows = {
        "binops": _counter_table(binop_counts, name="binop"),
        "boolops": _counter_table(boolop_counts, name="boolop"),
        "unaryops": _counter_table(unaryop_counts, name="unaryop"),
        "compares": _counter_table(compare_counts, name="compare"),
        "for_iters": _counter_table(for_iter_counts, name="for_iter"),
    }

    aggregate_summary
    return (
        aggregate_summary,
        call_rows,
        decorator_rows,
        grammar_feature_rows,
        import_rows,
        jit_call_rows,
        jit_statement_rows,
        operator_rows,
        statement_rows,
    )


@app.cell
def _(Counter, response_features):
    def _percentile(sorted_values, percentile):
        if not sorted_values:
            return None
        _index = (len(sorted_values) - 1) * percentile / 100
        _lower = int(_index)
        _upper = min(_lower + 1, len(sorted_values) - 1)
        if _lower == _upper:
            return sorted_values[_lower]
        _fraction = _index - _lower
        return sorted_values[_lower] * (1 - _fraction) + sorted_values[_upper] * _fraction


    indent_levels = sorted(row["max_indent_levels"] for row in response_features)
    indent_spaces = sorted(row["max_indent_spaces"] for row in response_features)
    line_indent_levels = sorted(row["max_line_indent_levels"] for row in response_features)
    line_indent_spaces = sorted(row["max_line_indent_spaces"] for row in response_features)
    indent_histogram = [
        {"max_indent_levels": _level, "responses": _count}
        for _level, _count in Counter(indent_levels).most_common()
    ]
    indent_histogram.sort(key=lambda row: row["max_indent_levels"])

    indent_summary = {
        "responses": len(response_features),
        "max_indent_levels_min": indent_levels[0] if indent_levels else None,
        "max_indent_levels_p50": _percentile(indent_levels, 50),
        "max_indent_levels_p90": _percentile(indent_levels, 90),
        "max_indent_levels_p95": _percentile(indent_levels, 95),
        "max_indent_levels_p99": _percentile(indent_levels, 99),
        "max_indent_levels_max": indent_levels[-1] if indent_levels else None,
        "max_indent_spaces_max": indent_spaces[-1] if indent_spaces else None,
        "max_line_indent_levels_p95": _percentile(line_indent_levels, 95),
        "max_line_indent_levels_max": line_indent_levels[-1] if line_indent_levels else None,
        "max_line_indent_spaces_max": line_indent_spaces[-1] if line_indent_spaces else None,
        "responses_with_non_4_space_indents": sum(
            row["non_four_space_indent_lines"] > 0 for row in response_features
        ),
        "responses_with_tab_indents": sum(row["tab_indented_lines"] > 0 for row in response_features),
    }

    deepest_indent_rows = [
        {
            "entry_file": row["entry_file"],
            "model": row["model_label"],
            "max_indent_levels": row["max_indent_levels"],
            "max_indent_spaces": row["max_indent_spaces"],
            "max_line_indent_levels": row["max_line_indent_levels"],
            "max_line_indent_spaces": row["max_line_indent_spaces"],
            "content_lines": row["content_lines"],
            "syntax_ok": row["syntax_ok"],
        }
        for row in sorted(
            response_features,
            key=lambda row: (row["max_indent_levels"], row["content_lines"]),
            reverse=True,
        )[:20]
    ]

    indent_by_model = []
    for _model in sorted({row["model_label"] for row in response_features}):
        _rows = [row for row in response_features if row["model_label"] == _model]
        _levels = sorted(row["max_indent_levels"] for row in _rows)
        indent_by_model.append(
            {
                "model": _model,
                "responses": len(_rows),
                "p50_levels": _percentile(_levels, 50),
                "p90_levels": _percentile(_levels, 90),
                "p95_levels": _percentile(_levels, 95),
                "max_levels": max(_levels) if _levels else None,
            }
        )

    indent_summary
    return (
        deepest_indent_rows,
        indent_by_model,
        indent_histogram,
        indent_summary,
    )


@app.cell(hide_code=True)
def _(aggregate_summary, alt, indent_histogram, indent_summary, mo):
    _indent_chart = (
        alt.Chart(alt.Data(values=indent_histogram))
        .mark_bar()
        .encode(
            x=alt.X("max_indent_levels:O", title="Tokenizer INDENT levels"),
            y=alt.Y("responses:Q", title="Responses"),
            tooltip=["max_indent_levels:O", "responses:Q"],
        )
        .properties(height=220)
    )

    mo.vstack(
        [
            mo.md("## Corpus overview"),
            mo.hstack(
                [
                    mo.stat(aggregate_summary["responses"], label="responses", bordered=True),
                    mo.stat(aggregate_summary["syntax_ok"], label="syntax OK", bordered=True),
                    mo.stat(
                        aggregate_summary["responses_with_jit_kernel"],
                        label="with @triton.jit",
                        bordered=True,
                    ),
                    mo.stat(
                        aggregate_summary["responses_with_torch_calls"],
                        label="with torch.* calls",
                        bordered=True,
                    ),
                ]
            ),
            mo.md("## Indentation headline statistics"),
            mo.hstack(
                [
                    mo.stat(indent_summary["max_indent_levels_p95"], label="p95 block levels", bordered=True),
                    mo.stat(indent_summary["max_indent_levels_max"], label="max block levels", bordered=True),
                    mo.stat(
                        indent_summary["max_line_indent_levels_p95"],
                        label="p95 visual levels",
                        bordered=True,
                    ),
                    mo.stat(
                        indent_summary["max_line_indent_levels_max"],
                        label="max visual levels",
                        bordered=True,
                    ),
                ]
            ),
            mo.ui.altair_chart(_indent_chart, chart_selection=False, legend_selection=False),
        ]
    )
    return


@app.cell(hide_code=True)
def _(alt, grammar_feature_rows, mo):
    _feature_chart_rows = [
        {
            "feature": _row["feature"],
            "scope": _scope,
            "responses": _row[_key],
        }
        for _row in grammar_feature_rows
        for _scope, _key in [("module", "responses"), ("@triton.jit", "jit_responses")]
        if _row[_key] > 0
    ]
    _feature_chart = (
        alt.Chart(alt.Data(values=_feature_chart_rows))
        .mark_bar()
        .encode(
            y=alt.Y("feature:N", sort="-x", title=None),
            x=alt.X("responses:Q", title="Responses using feature"),
            color=alt.Color("scope:N", title="Scope"),
            tooltip=["feature:N", "scope:N", "responses:Q"],
        )
        .properties(height=520)
    )

    mo.vstack(
        [
            mo.md("## Python/Triton grammar features observed"),
            mo.md(
                "`responses` counts any syntactically valid completion using the AST node anywhere; "
                "`jit_*` counts only nodes nested under `@triton.jit` functions."
            ),
            mo.ui.altair_chart(_feature_chart, chart_selection=False, legend_selection=False),
        ]
    )
    return


@app.cell(hide_code=True)
def _(alt, call_rows, import_rows, jit_call_rows, mo):
    def _bar_chart(rows, label_field, count_field="count", *, title, height=360, limit=30):
        _rows = rows[:limit]
        return (
            alt.Chart(alt.Data(values=_rows), title=title)
            .mark_bar()
            .encode(
                y=alt.Y(f"{label_field}:N", sort="-x", title=None),
                x=alt.X(f"{count_field}:Q", title="Count"),
                tooltip=[f"{label_field}:N", f"{count_field}:Q"],
            )
            .properties(height=height)
        )

    _call_chart = _bar_chart(call_rows, "call", title="Top calls across the whole module")
    _jit_call_chart = _bar_chart(
        jit_call_rows,
        "jit_call",
        title="Top calls inside @triton.jit kernels",
    )
    _import_chart = _bar_chart(import_rows, "import", title="Imports used by generated completions", limit=20)

    mo.vstack(
        [
            mo.md("## Calls and imports to consider for the grammar manifest"),
            mo.ui.altair_chart(_call_chart, chart_selection=False, legend_selection=False),
            mo.ui.altair_chart(_jit_call_chart, chart_selection=False, legend_selection=False),
            mo.ui.altair_chart(_import_chart, chart_selection=False, legend_selection=False),
        ]
    )
    return


@app.cell(hide_code=True)
def _(alt, decorator_rows, jit_statement_rows, mo, operator_rows, statement_rows):
    _statement_chart_rows = [
        {"statement": _row["statement"], "scope": "module", "count": _row["count"]}
        for _row in statement_rows
    ] + [
        {"statement": _row["jit_statement"], "scope": "@triton.jit", "count": _row["count"]}
        for _row in jit_statement_rows
    ]
    _statement_chart = (
        alt.Chart(alt.Data(values=_statement_chart_rows), title="Statement counts by scope")
        .mark_bar()
        .encode(
            y=alt.Y("statement:N", sort="-x", title=None),
            x=alt.X("count:Q", title="Count"),
            color=alt.Color("scope:N", title="Scope"),
            tooltip=["statement:N", "scope:N", "count:Q"],
        )
        .properties(height=420)
    )

    _operator_chart_rows = [
        {
            "kind": _kind,
            "operator": _row[_kind.removesuffix("s")],
            "count": _row["count"],
        }
        for _kind, _rows in operator_rows.items()
        for _row in _rows
    ]
    _operator_chart = (
        alt.Chart(alt.Data(values=_operator_chart_rows), title="Operators and loop iterators")
        .mark_bar()
        .encode(
            y=alt.Y("operator:N", sort="-x", title=None),
            x=alt.X("count:Q", title="Count"),
            color=alt.Color("kind:N", title="Kind"),
            tooltip=["kind:N", "operator:N", "count:Q"],
        )
        .properties(height=360)
    )

    _decorator_chart = (
        alt.Chart(alt.Data(values=decorator_rows), title="Decorators")
        .mark_bar()
        .encode(
            y=alt.Y("decorator:N", sort="-x", title=None),
            x=alt.X("count:Q", title="Count"),
            tooltip=["decorator:N", "count:Q"],
        )
        .properties(height=180)
    )

    mo.vstack(
        [
            mo.md("## Statement, decorator, and operator usage"),
            mo.ui.altair_chart(_statement_chart, chart_selection=False, legend_selection=False),
            mo.ui.altair_chart(_decorator_chart, chart_selection=False, legend_selection=False),
            mo.ui.altair_chart(_operator_chart, chart_selection=False, legend_selection=False),
        ]
    )
    return


@app.cell(hide_code=True)
def _(alt, deepest_indent_rows, indent_by_model, mo):
    _model_indent_rows = [
        {"model": _row["model"], "stat": _stat, "levels": _row[_key]}
        for _row in indent_by_model
        for _stat, _key in [("p50", "p50_levels"), ("p90", "p90_levels"), ("p95", "p95_levels"), ("max", "max_levels")]
    ]
    _model_indent_chart = (
        alt.Chart(alt.Data(values=_model_indent_rows), title="Maximum tokenizer indentation by model")
        .mark_bar()
        .encode(
            x=alt.X("stat:N", title=None, sort=["p50", "p90", "p95", "max"]),
            y=alt.Y("levels:Q", title="Tokenizer INDENT levels"),
            color=alt.Color("model:N", title="Model"),
            column=alt.Column("model:N", title=None),
            tooltip=["model:N", "stat:N", "levels:Q"],
        )
        .properties(height=220)
    )

    mo.vstack(
        [
            mo.md("## Indentation distribution for explicit grammar limits"),
            mo.ui.altair_chart(_model_indent_chart, chart_selection=False, legend_selection=False),
            mo.md("Deepest individual responses:"),
            mo.ui.table(deepest_indent_rows),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, response_features):
    syntax_error_rows = [
        {
            "entry_file": row["entry_file"],
            "model": row["model_label"],
            "syntax_error": row["syntax_error"],
            "max_indent_levels": row["max_indent_levels"],
        }
        for row in response_features
        if not row["syntax_ok"]
    ]

    mo.vstack(
        [
            mo.md("## Syntax-error completions"),
            mo.md("These rows are excluded from AST feature counts but still included in indentation stats."),
            mo.ui.table(syntax_error_rows),
        ]
    )
    return


@app.cell(hide_code=True)
def _(indent_summary, mo):
    mo.md(
        "\n".join(
            [
                "## Initial grammar implications",
                "",
                "- A first-pass grammar should include the observed high-frequency AST nodes before adding",
                "  rarely used Triton-supported constructs such as `while`, `with`, walrus, comprehensions,",
                "  or exception handling.",
                "- The indentation data gives an empirical bound for hard-coded indentation productions:",
                f"  p95 is **{indent_summary['max_indent_levels_p95']}** levels and max is",
                f"  **{indent_summary['max_indent_levels_max']}** tokenizer `INDENT` levels across this corpus.",
                "  Physical line alignment can be deeper inside multiline calls/expressions:",
                f"  p95 is **{indent_summary['max_line_indent_levels_p95']}** and max is",
                f"  **{indent_summary['max_line_indent_levels_max']}** visual 4-space levels.",
                "- Because the corpus still includes host wrappers, keep separate grammar contexts for module",
                "  imports, host wrapper orchestration, and `@triton.jit` kernel bodies rather than allowing",
                "  every observed node everywhere.",
            ]
        )
    )
    return


if __name__ == "__main__":
    app.run()
