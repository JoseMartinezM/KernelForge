"""
compiler/triton_parser.py
=========================
Parser + Direct Syntax Translator para código Triton-GPU usando PLY Yacc.

Este módulo es un DIRECT SYNTAX TRANSLATOR (DST):
  - Analiza sintácticamente el código fuente usando gramática BNF.
  - Durante el parsing, ejecuta ACCIONES SEMÁNTICAS que:
      • Construyen la TABLA DE SÍMBOLOS (funciones, parámetros, variables locales).
      • Detectan patrones específicos de Triton (@triton.jit, tl.load, etc.).
      • Emiten un REPORTE DE TRADUCCIÓN (la "salida" del translator).

DIFERENCIA ENTRE COMPILER Y DST:
  Compiler tradicional:  fuente → tokens → AST → IR → código máquina
  Direct Syntax Translator: fuente → tokens → traducción DURANTE el parsing
  (no necesitamos un AST separado: actuamos en cada regla a medida que se reduce)

GRAMÁTICA (BNF sin recursión izquierda):
  La técnica de eliminación es: A → A α | β  se convierte en:
      A  → β A'
      A' → α A' | ε
  Esto se aplica a todas las expresiones binarias para evitar recursión izquierda.

CONFLICTOS SHIFT/REDUCE:
  Yacc tiene un conflicto natural con el "dangling else":
      if expr: suite
      if expr: suite else: suite
  Con INDENT/DEDENT esto desaparece porque el `else` debe estar en el mismo
  nivel de indentación que el `if`. El parser sabe a qué `if` pertenece.
"""

from __future__ import annotations

import ply.yacc as yacc
from typing import Any

from .triton_lexer import tokens, build_lexer  # noqa: F401 — PLY necesita `tokens` en scope


# ---------------------------------------------------------------------------
# TABLA DE SÍMBOLOS
# ---------------------------------------------------------------------------
# Estructura global que el parser llena durante la traducción.
# El parser de PLY no es re-entrante con estado global, por lo que
# la reiniciamos en cada llamada a parse().

class SymbolTable:
    """
    Almacena información sobre cada kernel Triton encontrado.

    Pregunta típica del oral: "¿Qué información guardas en la tabla de símbolos?"
    Respuesta: nombre de función, línea de declaración, si tiene @triton.jit,
    parámetros (con anotación y si son constexpr), variables locales asignadas,
    y llamadas a la API de Triton (tl.*) detectadas.
    """

    def __init__(self) -> None:
        self.functions:  dict[str, dict] = {}   # nombre → info de la función
        self.errors:     list[str]        = []
        self.warnings:   list[str]        = []

    def declare_function(self, name: str, lineno: int, params: list[dict], decorators: list[str]) -> None:
        if name in self.functions:
            self.errors.append(f"Línea {lineno}: función '{name}' ya declarada")
            return
        self.functions[name] = {
            "linea":            lineno,
            "tiene_triton_jit": any("triton.jit" in d for d in decorators),
            "parametros":       params,        # [{"nombre": ..., "anotacion": ..., "es_constexpr": bool}]
            "variables_locales": [],
            "llamadas_triton":  [],
        }

    def add_local_var(self, func_name: str, var_name: str) -> None:
        if func_name and func_name in self.functions:
            locs = self.functions[func_name]["variables_locales"]
            if var_name not in locs:
                locs.append(var_name)

    def add_triton_call(self, func_name: str, call_name: str) -> None:
        if func_name and func_name in self.functions:
            calls = self.functions[func_name]["llamadas_triton"]
            if call_name not in calls:
                calls.append(call_name)

    def report(self) -> str:
        """Genera el reporte de traducción (la 'salida' del DST)."""
        lines: list[str] = []
        lines.append("═" * 60)
        lines.append("  REPORTE DE TRADUCCIÓN — Triton Kernel Validator")
        lines.append("═" * 60)

        if not self.functions:
            lines.append("  (no se encontraron definiciones de función)")
        else:
            for fname, info in self.functions.items():
                tag = "[KERNEL]" if info["tiene_triton_jit"] else "[FUNC]  "
                params_str = ", ".join(
                    f"{p['nombre']}: {p['anotacion']}" if p["anotacion"] else p["nombre"]
                    for p in info["parametros"]
                )
                lines.append(f"\n  {tag} {fname}({params_str})")
                lines.append(f"           Línea: {info['linea']}")
                lines.append(f"           @triton.jit: {'✓' if info['tiene_triton_jit'] else '✗'}")

                constexprs = [p["nombre"] for p in info["parametros"] if p["es_constexpr"]]
                if constexprs:
                    lines.append(f"           constexpr: {', '.join(constexprs)}")

                if info["llamadas_triton"]:
                    lines.append(f"           API Triton: {', '.join(info['llamadas_triton'])}")

                if info["variables_locales"]:
                    lines.append(f"           Locales: {', '.join(info['variables_locales'])}")

        if self.errors:
            lines.append("\n  ── ERRORES ──")
            for e in self.errors:
                lines.append(f"  ✗ {e}")
        if self.warnings:
            lines.append("\n  ── ADVERTENCIAS ──")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")

        status = "✓ VÁLIDO" if not self.errors else "✗ INVÁLIDO"
        lines.append(f"\n  Estado: {status}")
        lines.append("═" * 60)
        return "\n".join(lines)


# Instancia global reiniciada en cada parse()
_symtab: SymbolTable = SymbolTable()
_current_func: str = ""   # nombre de la función que se está parseando ahora


# ---------------------------------------------------------------------------
# PRECEDENCIA DE OPERADORES
# ---------------------------------------------------------------------------
# Yacc usa esta tabla para resolver conflictos shift/reduce en expresiones.
# Se lista de MENOR a MAYOR precedencia. `left`/`right`/`nonassoc` indica
# la asociatividad.
#
# Pregunta oral: "¿Tuviste conflictos shift/reduce? ¿Cómo los resolviste?"
# Respuesta: Sí, en operadores binarios. Los resolví con declaraciones de
# precedencia en lugar de reescribir las reglas gramaticales.

precedence = (
    ("left",    "OR"),
    ("left",    "AND"),
    ("right",   "NOT"),
    ("left",    "IN", "IS"),
    ("left",    "LT", "GT", "LTEQ", "GTEQ", "EQEQ", "NOTEQ"),
    ("left",    "PIPE"),
    ("left",    "CARET"),
    ("left",    "AMP"),
    ("left",    "LSHIFT", "RSHIFT"),
    ("left",    "PLUS", "MINUS"),
    ("left",    "STAR", "SLASH", "DOUBLESLASH", "PERCENT"),
    ("right",   "UMINUS", "UTILDE", "UPLUS"),   # unarios (con alias)
    ("right",   "DOUBLESTAR"),
    ("left",    "DOT", "LPAREN", "LBRACKET"),   # llamadas y subscripts
)


# ---------------------------------------------------------------------------
# REGLAS GRAMATICALES (Yacc)
# ---------------------------------------------------------------------------
# Cada función p_* define una o más producciones BNF.
# p[0] = valor del símbolo no-terminal de la izquierda (lo que "retornamos").
# p[1], p[2], ... = valores de los símbolos del lado derecho.
#
# NOTE sobre recursión izquierda:
#   Yacc LALR(1) PUEDE manejar recursión izquierda eficientemente.
#   Sin embargo, para satisfacer el requisito del curso (gramática sin
#   recursión izquierda, como se enseña en parsers LL), usamos la técnica
#   estándar de transformación:
#       A  → β A'
#       A' → α A' | ε
#   Esto aplica a todas las listas y expresiones binarias.


# ── Programa ──────────────────────────────────────────────────────────────

def p_program(p):
    """program : stmt_seq"""
    p[0] = ("program", p[1])


def p_stmt_seq_nonempty(p):
    """stmt_seq : stmt stmt_seq"""
    # Recursión DERECHA: stmt_seq → stmt stmt_seq | ε
    p[0] = [p[1]] + (p[2] or [])


def p_stmt_seq_blank(p):
    """stmt_seq : NEWLINE stmt_seq"""
    # Líneas en blanco entre statements (NEWLINEs sueltos)
    p[0] = p[2] or []


def p_stmt_seq_empty(p):
    """stmt_seq : empty"""
    p[0] = []


# ── Statement ────────────────────────────────────────────────────────────

def p_stmt_simple(p):
    """stmt : simple_line"""
    p[0] = p[1]


def p_stmt_compound(p):
    """stmt : compound_stmt"""
    p[0] = p[1]


# ── Línea simple (termina en NEWLINE) ────────────────────────────────────

def p_simple_line(p):
    """simple_line : simple_stmt NEWLINE"""
    p[0] = p[1]


def p_simple_line_semi(p):
    """simple_line : simple_stmt SEMI simple_stmt NEWLINE"""
    p[0] = ("seq", p[1], p[3])


# ── Statements simples ────────────────────────────────────────────────────

def p_simple_stmt(p):
    """simple_stmt : assign_stmt
                   | aug_assign_stmt
                   | ann_assign_stmt
                   | return_stmt
                   | assert_stmt
                   | pass_stmt
                   | import_stmt
                   | expr_stmt"""
    p[0] = p[1]


# ── Asignación  target = expr ─────────────────────────────────────────────

def p_assign_stmt(p):
    """assign_stmt : target_expr EQ expr_list"""
    global _current_func
    # ACCIÓN SEMÁNTICA: registrar variable local si estamos dentro de una función
    if isinstance(p[1], tuple) and p[1][0] == "name":
        _symtab.add_local_var(_current_func, p[1][1])
    p[0] = ("assign", p[1], p[3])


def p_assign_stmt_chained(p):
    """assign_stmt : target_expr EQ assign_stmt"""
    p[0] = ("assign_chain", p[1], p[3])


# ── Asignación aumentada  target += expr ─────────────────────────────────

def p_aug_assign_stmt(p):
    """aug_assign_stmt : target_expr aug_op expr"""
    p[0] = ("aug_assign", p[1], p[2], p[3])


def p_aug_op(p):
    """aug_op : PLUSEQ
              | MINUSEQ
              | STAREQ
              | SLASHEQ
              | DOUBLESLASHEQ
              | PERCENTEQ
              | AMPEQ
              | PIPEEQ
              | CARETEQ
              | LSHIFTEQ
              | RSHIFTEQ
              | DOUBLESTAREQ"""
    p[0] = p[1]


# ── Asignación anotada  target : type  o  target : type = expr ───────────

def p_ann_assign_stmt(p):
    """ann_assign_stmt : target_expr COLON expr"""
    p[0] = ("ann_assign", p[1], p[3], None)


def p_ann_assign_stmt_val(p):
    """ann_assign_stmt : target_expr COLON expr EQ expr"""
    p[0] = ("ann_assign", p[1], p[3], p[5])


# ── Return ────────────────────────────────────────────────────────────────

def p_return_stmt(p):
    """return_stmt : RETURN"""
    p[0] = ("return", None)


def p_return_stmt_val(p):
    """return_stmt : RETURN expr_list"""
    p[0] = ("return", p[2])


# ── Assert ────────────────────────────────────────────────────────────────

def p_assert_stmt(p):
    """assert_stmt : ASSERT expr"""
    p[0] = ("assert", p[2], None)


def p_assert_stmt_msg(p):
    """assert_stmt : ASSERT expr COMMA expr"""
    p[0] = ("assert", p[2], p[4])


# ── Pass ──────────────────────────────────────────────────────────────────

def p_pass_stmt(p):
    """pass_stmt : PASS"""
    p[0] = ("pass",)


# ── Expr statement (llamadas a funciones sueltas, etc.) ───────────────────

def p_expr_stmt(p):
    """expr_stmt : expr"""
    p[0] = ("expr_stmt", p[1])


# ── Import ────────────────────────────────────────────────────────────────

def p_import_stmt_simple(p):
    """import_stmt : IMPORT dotted_name"""
    p[0] = ("import", p[2])


def p_import_stmt_as(p):
    """import_stmt : IMPORT dotted_name AS NAME"""
    p[0] = ("import_as", p[2], p[4])


def p_import_stmt_from(p):
    """import_stmt : FROM dotted_name IMPORT NAME"""
    p[0] = ("from_import", p[2], p[4])


def p_import_stmt_from_as(p):
    """import_stmt : FROM dotted_name IMPORT NAME AS NAME"""
    p[0] = ("from_import_as", p[2], p[4], p[6])


def p_import_stmt_from_star(p):
    """import_stmt : FROM dotted_name IMPORT STAR"""
    p[0] = ("from_import_star", p[2])


def p_dotted_name(p):
    """dotted_name : NAME dotted_name_tail"""
    p[0] = p[1] + p[2]


def p_dotted_name_tail_more(p):
    """dotted_name_tail : DOT NAME dotted_name_tail"""
    p[0] = "." + p[2] + p[3]


def p_dotted_name_tail_empty(p):
    """dotted_name_tail : empty"""
    p[0] = ""


# ── Statements compuestos ─────────────────────────────────────────────────

def p_compound_stmt(p):
    """compound_stmt : funcdef
                     | if_stmt
                     | for_stmt
                     | while_stmt
                     | with_stmt"""
    p[0] = p[1]


# ── Definición de función ─────────────────────────────────────────────────

def p_funcdef_decorated(p):
    """funcdef : decorator_seq DEF NAME LPAREN param_seq RPAREN COLON suite"""
    global _current_func
    name       = p[3]
    decorators = p[1]
    params     = p[5]
    lineno     = p.lineno(2)
    # ACCIÓN SEMÁNTICA: registrar en tabla de símbolos
    _symtab.declare_function(name, lineno, params, decorators)
    _current_func = ""  # salimos de la función
    p[0] = ("funcdef", name, decorators, params, p[8])


def p_funcdef_plain(p):
    """funcdef : DEF NAME LPAREN param_seq RPAREN COLON suite"""
    global _current_func
    name   = p[2]
    params = p[4]
    lineno = p.lineno(1)
    _symtab.declare_function(name, lineno, params, decorators=[])
    _symtab.warnings.append(
        f"Función '{name}' (línea {lineno}) declarada sin decorador @triton.jit"
    )
    _current_func = ""
    p[0] = ("funcdef", name, [], params, p[7])


def p_funcdef_decorated_arrow(p):
    """funcdef : decorator_seq DEF NAME LPAREN param_seq RPAREN ARROW expr COLON suite"""
    global _current_func
    name       = p[3]
    decorators = p[1]
    params     = p[5]
    lineno     = p.lineno(2)
    _symtab.declare_function(name, lineno, params, decorators)
    _current_func = ""
    p[0] = ("funcdef", name, decorators, params, p[10])


# ── Decoradores ───────────────────────────────────────────────────────────

def p_decorator_seq_multi(p):
    """decorator_seq : decorator decorator_seq"""
    p[0] = [p[1]] + p[2]


def p_decorator_seq_one(p):
    """decorator_seq : decorator"""
    p[0] = [p[1]]


def p_decorator(p):
    """decorator : AT expr NEWLINE"""
    # Convierte el nodo AST del decorador a string legible, ej: "triton.jit"
    p[0] = _expr_to_str(p[2])


# ── Suite (bloque indentado) ──────────────────────────────────────────────
# Pregunta oral: "¿Cómo representas los bloques?"
# Respuesta: El lexer emite INDENT al entrar y DEDENT al salir.
# La regla suite siempre es: NEWLINE INDENT <cuerpo> DEDENT

def p_suite(p):
    """suite : NEWLINE INDENT stmt_seq DEDENT"""
    p[0] = ("suite", p[3])


def p_suite_blank_before(p):
    """suite : NEWLINE INDENT NEWLINE stmt_seq DEDENT"""
    # Suite con línea en blanco al inicio del bloque
    p[0] = ("suite", p[4])


# ── If ────────────────────────────────────────────────────────────────────

def p_if_stmt(p):
    """if_stmt : IF expr COLON suite else_part"""
    p[0] = ("if", p[2], p[4], p[5])


def p_else_part_elif(p):
    """else_part : ELIF expr COLON suite else_part"""
    p[0] = ("elif", p[2], p[4], p[5])


def p_else_part_else(p):
    """else_part : ELSE COLON suite"""
    p[0] = ("else", p[3])


def p_else_part_empty(p):
    """else_part : empty"""
    p[0] = None


# ── For ───────────────────────────────────────────────────────────────────

def p_for_stmt(p):
    """for_stmt : FOR target_expr IN expr COLON suite"""
    p[0] = ("for", p[2], p[4], p[6], None)


def p_for_stmt_else(p):
    """for_stmt : FOR target_expr IN expr COLON suite ELSE COLON suite"""
    p[0] = ("for", p[2], p[4], p[6], p[9])


# ── While ─────────────────────────────────────────────────────────────────

def p_while_stmt(p):
    """while_stmt : WHILE expr COLON suite"""
    p[0] = ("while", p[2], p[4], None)


def p_while_stmt_else(p):
    """while_stmt : WHILE expr COLON suite ELSE COLON suite"""
    p[0] = ("while", p[2], p[4], p[7])


# ── With ──────────────────────────────────────────────────────────────────

def p_with_stmt(p):
    """with_stmt : WITH with_item COLON suite"""
    p[0] = ("with", p[2], p[4])


def p_with_item_plain(p):
    """with_item : expr"""
    p[0] = ("with_item", p[1], None)


def p_with_item_as(p):
    """with_item : expr AS target_expr"""
    p[0] = ("with_item", p[1], p[3])


# ── Parámetros de función ─────────────────────────────────────────────────

def p_param_seq_nonempty(p):
    """param_seq : param param_seq_tail"""
    p[0] = [p[1]] + p[2]


def p_param_seq_empty(p):
    """param_seq : empty"""
    p[0] = []


def p_param_seq_tail_more(p):
    """param_seq_tail : COMMA param param_seq_tail"""
    p[0] = [p[2]] + p[3]


def p_param_seq_tail_trailing(p):
    """param_seq_tail : COMMA"""
    p[0] = []


def p_param_seq_tail_empty(p):
    """param_seq_tail : empty"""
    p[0] = []


def p_param_plain(p):
    """param : NAME"""
    p[0] = {"nombre": p[1], "anotacion": None, "es_constexpr": False}


def p_param_annotated(p):
    """param : NAME COLON expr"""
    ann = _expr_to_str(p[3])
    is_ce = "constexpr" in ann
    p[0] = {"nombre": p[1], "anotacion": ann, "es_constexpr": is_ce}


def p_param_default(p):
    """param : NAME EQ expr"""
    p[0] = {"nombre": p[1], "anotacion": None, "es_constexpr": False, "default": p[3]}


def p_param_annotated_default(p):
    """param : NAME COLON expr EQ expr"""
    ann = _expr_to_str(p[3])
    is_ce = "constexpr" in ann
    p[0] = {"nombre": p[1], "anotacion": ann, "es_constexpr": is_ce, "default": p[5]}


def p_param_star(p):
    """param : STAR NAME"""
    p[0] = {"nombre": "*" + p[2], "anotacion": None, "es_constexpr": False}


def p_param_doublestar(p):
    """param : DOUBLESTAR NAME"""
    p[0] = {"nombre": "**" + p[2], "anotacion": None, "es_constexpr": False}


# ── Lista de expresiones (para el lado derecho de asignaciones, etc.) ─────

def p_expr_list_one(p):
    """expr_list : expr"""
    p[0] = ("expr_list", [p[1]])


def p_expr_list_multi(p):
    """expr_list : expr COMMA expr_list_tail"""
    p[0] = ("expr_list", [p[1]] + p[3])


def p_expr_list_tail_more(p):
    """expr_list_tail : expr COMMA expr_list_tail"""
    p[0] = [p[1]] + p[3]


def p_expr_list_tail_last(p):
    """expr_list_tail : expr"""
    p[0] = [p[1]]


def p_expr_list_tail_trailing(p):
    """expr_list_tail : empty"""
    p[0] = []


# ── Target para asignaciones ──────────────────────────────────────────────

def p_target_expr(p):
    """target_expr : primary"""
    p[0] = p[1]


# ── EXPRESIONES — eliminación de recursión izquierda ─────────────────────
#
# Técnica: A → A α | β  ⟹  A → β A'  y  A' → α A' | ε
#
# Cada nivel de la jerarquía de precedencia sigue este patrón.
# Los nombres con _tail son los A' (colas recursivas derechas).

def p_expr(p):
    """expr : cond_expr"""
    p[0] = p[1]


# ── Condicional (ternario)  x if cond else y ─────────────────────────────

def p_cond_expr_ternary(p):
    """cond_expr : or_expr IF or_expr ELSE cond_expr"""
    p[0] = ("ternary", p[2], p[1], p[5])


def p_cond_expr_plain(p):
    """cond_expr : or_expr"""
    p[0] = p[1]


# ── Or ────────────────────────────────────────────────────────────────────

def p_or_expr(p):
    """or_expr : and_expr or_tail"""
    p[0] = _fold_tail(p[1], p[2])


def p_or_tail_more(p):
    """or_tail : OR and_expr or_tail"""
    p[0] = ("OR", p[2], p[3])


def p_or_tail_empty(p):
    """or_tail : empty"""
    p[0] = None


# ── And ───────────────────────────────────────────────────────────────────

def p_and_expr(p):
    """and_expr : not_expr and_tail"""
    p[0] = _fold_tail(p[1], p[2])


def p_and_tail_more(p):
    """and_tail : AND not_expr and_tail"""
    p[0] = ("AND", p[2], p[3])


def p_and_tail_empty(p):
    """and_tail : empty"""
    p[0] = None


# ── Not ───────────────────────────────────────────────────────────────────

def p_not_expr_not(p):
    """not_expr : NOT not_expr"""
    p[0] = ("not", p[2])


def p_not_expr_comparison(p):
    """not_expr : comparison"""
    p[0] = p[1]


# ── Comparación ───────────────────────────────────────────────────────────

def p_comparison_op(p):
    """comparison : bitor comp_op bitor"""
    p[0] = ("cmp", p[2], p[1], p[3])


def p_comparison_plain(p):
    """comparison : bitor"""
    p[0] = p[1]


def p_comp_op(p):
    """comp_op : EQEQ
               | NOTEQ
               | LT
               | GT
               | LTEQ
               | GTEQ
               | IN
               | IS"""
    p[0] = p[1]


def p_comp_op_not_in(p):
    """comp_op : NOT IN"""
    p[0] = "not in"


def p_comp_op_is_not(p):
    """comp_op : IS NOT"""
    p[0] = "is not"


# ── Bitwise Or, Xor, And ──────────────────────────────────────────────────

def p_bitor(p):
    """bitor : bitxor bitor_tail"""
    p[0] = _fold_tail(p[1], p[2])


def p_bitor_tail_more(p):
    """bitor_tail : PIPE bitxor bitor_tail"""
    p[0] = ("|", p[2], p[3])


def p_bitor_tail_empty(p):
    """bitor_tail : empty"""
    p[0] = None


def p_bitxor(p):
    """bitxor : bitand bitxor_tail"""
    p[0] = _fold_tail(p[1], p[2])


def p_bitxor_tail_more(p):
    """bitxor_tail : CARET bitand bitxor_tail"""
    p[0] = ("^", p[2], p[3])


def p_bitxor_tail_empty(p):
    """bitxor_tail : empty"""
    p[0] = None


def p_bitand(p):
    """bitand : shift bitand_tail"""
    p[0] = _fold_tail(p[1], p[2])


def p_bitand_tail_more(p):
    """bitand_tail : AMP shift bitand_tail"""
    p[0] = ("&", p[2], p[3])


def p_bitand_tail_empty(p):
    """bitand_tail : empty"""
    p[0] = None


# ── Shift ─────────────────────────────────────────────────────────────────

def p_shift(p):
    """shift : sum shift_tail"""
    p[0] = _fold_tail(p[1], p[2])


def p_shift_tail_left(p):
    """shift_tail : LSHIFT sum shift_tail"""
    p[0] = ("<<", p[2], p[3])


def p_shift_tail_right(p):
    """shift_tail : RSHIFT sum shift_tail"""
    p[0] = (">>", p[2], p[3])


def p_shift_tail_empty(p):
    """shift_tail : empty"""
    p[0] = None


# ── Sum ───────────────────────────────────────────────────────────────────

def p_sum(p):
    """sum : term sum_tail"""
    p[0] = _fold_tail(p[1], p[2])


def p_sum_tail_plus(p):
    """sum_tail : PLUS term sum_tail"""
    p[0] = ("+", p[2], p[3])


def p_sum_tail_minus(p):
    """sum_tail : MINUS term sum_tail"""
    p[0] = ("-", p[2], p[3])


def p_sum_tail_empty(p):
    """sum_tail : empty"""
    p[0] = None


# ── Term (multiplicación / división) ─────────────────────────────────────

def p_term(p):
    """term : factor term_tail"""
    p[0] = _fold_tail(p[1], p[2])


def p_term_tail_star(p):
    """term_tail : STAR factor term_tail"""
    p[0] = ("*", p[2], p[3])


def p_term_tail_slash(p):
    """term_tail : SLASH factor term_tail"""
    p[0] = ("/", p[2], p[3])


def p_term_tail_dslash(p):
    """term_tail : DOUBLESLASH factor term_tail"""
    p[0] = ("//", p[2], p[3])


def p_term_tail_percent(p):
    """term_tail : PERCENT factor term_tail"""
    p[0] = ("%", p[2], p[3])


def p_term_tail_at(p):
    """term_tail : AT factor term_tail"""
    p[0] = ("@", p[2], p[3])


def p_term_tail_empty(p):
    """term_tail : empty"""
    p[0] = None


# ── Factor (unarios) ──────────────────────────────────────────────────────

def p_factor_uplus(p):
    """factor : PLUS factor %prec UPLUS"""
    p[0] = ("uplus", p[2])


def p_factor_uminus(p):
    """factor : MINUS factor %prec UMINUS"""
    p[0] = ("uminus", p[2])


def p_factor_utilde(p):
    """factor : TILDE factor %prec UTILDE"""
    p[0] = ("invert", p[2])


def p_factor_power(p):
    """factor : power"""
    p[0] = p[1]


# ── Power ─────────────────────────────────────────────────────────────────

def p_power_exp(p):
    """power : primary DOUBLESTAR factor"""
    p[0] = ("**", p[1], p[3])


def p_power_plain(p):
    """power : primary"""
    p[0] = p[1]


# ── Primary (acceso a atributos, llamadas, subscripts) ───────────────────
# Usamos técnica A → β A':
#   primary     → atom trailer_seq
#   trailer_seq → trailer trailer_seq | ε

def p_primary(p):
    """primary : atom trailer_seq"""
    node = p[1]
    for trailer in p[2]:
        node = ("access", node, trailer)
    p[0] = node


def p_trailer_seq_more(p):
    """trailer_seq : trailer trailer_seq"""
    p[0] = [p[1]] + p[2]


def p_trailer_seq_empty(p):
    """trailer_seq : empty"""
    p[0] = []


def p_trailer_attr(p):
    """trailer : DOT NAME"""
    p[0] = ("attr", p[2])


def p_trailer_call(p):
    """trailer : LPAREN arg_seq RPAREN"""
    p[0] = ("call", p[2])


def p_trailer_subscript(p):
    """trailer : LBRACKET slice_seq RBRACKET"""
    p[0] = ("subscript", p[2])


# ── Argumentos de llamada a función ──────────────────────────────────────

def p_arg_seq_nonempty(p):
    """arg_seq : arg arg_seq_tail"""
    p[0] = [p[1]] + p[2]


def p_arg_seq_empty(p):
    """arg_seq : empty"""
    p[0] = []


def p_arg_seq_tail_more(p):
    """arg_seq_tail : COMMA arg arg_seq_tail"""
    p[0] = [p[2]] + p[3]


def p_arg_seq_tail_trailing(p):
    """arg_seq_tail : COMMA"""
    p[0] = []


def p_arg_seq_tail_empty(p):
    """arg_seq_tail : empty"""
    p[0] = []


def p_arg_plain(p):
    """arg : expr"""
    p[0] = ("arg", p[1])


def p_arg_keyword(p):
    """arg : NAME EQ expr"""
    p[0] = ("kwarg", p[1], p[3])


def p_arg_star(p):
    """arg : STAR expr"""
    p[0] = ("star_arg", p[2])


def p_arg_doublestar(p):
    """arg : DOUBLESTAR expr"""
    p[0] = ("dstar_arg", p[2])


# ── Slices ────────────────────────────────────────────────────────────────

def p_slice_seq_nonempty(p):
    """slice_seq : slice_item slice_seq_tail"""
    p[0] = [p[1]] + p[2]


def p_slice_seq_empty(p):
    """slice_seq : empty"""
    p[0] = []


def p_slice_seq_tail_more(p):
    """slice_seq_tail : COMMA slice_item slice_seq_tail"""
    p[0] = [p[2]] + p[3]


def p_slice_seq_tail_trailing(p):
    """slice_seq_tail : COMMA"""
    p[0] = []


def p_slice_seq_tail_empty(p):
    """slice_seq_tail : empty"""
    p[0] = []


def p_slice_full(p):
    """slice_item : expr COLON expr COLON expr"""
    p[0] = ("slice", p[1], p[3], p[5])


def p_slice_start_stop(p):
    """slice_item : expr COLON expr"""
    p[0] = ("slice", p[1], p[3], None)


def p_slice_start_only(p):
    """slice_item : expr COLON"""
    p[0] = ("slice", p[1], None, None)


def p_slice_stop_only(p):
    """slice_item : COLON expr"""
    p[0] = ("slice", None, p[2], None)


def p_slice_all(p):
    """slice_item : COLON"""
    p[0] = ("slice", None, None, None)


def p_slice_expr(p):
    """slice_item : expr"""
    p[0] = ("index", p[1])


# ── Átomos (valores literales y agrupaciones) ─────────────────────────────

def p_atom_name(p):
    """atom : NAME"""
    p[0] = ("name", p[1])
    # ACCIÓN SEMÁNTICA: detectar llamadas a la API de Triton (tl.*)
    # Se detectan en el trailer (DOT NAME + call), manejado abajo.


def p_atom_number(p):
    """atom : NUMBER"""
    p[0] = ("number", p[1])


def p_atom_string(p):
    """atom : STRING"""
    p[0] = ("string", p[1])


def p_atom_true(p):
    """atom : TRUE"""
    p[0] = ("bool", True)


def p_atom_false(p):
    """atom : FALSE"""
    p[0] = ("bool", False)


def p_atom_none(p):
    """atom : NONE"""
    p[0] = ("none",)


def p_atom_ellipsis(p):
    """atom : ELLIPSIS"""
    p[0] = ("ellipsis",)


def p_atom_paren_empty(p):
    """atom : LPAREN RPAREN"""
    p[0] = ("tuple", [])


def p_atom_paren(p):
    """atom : LPAREN expr_list RPAREN"""
    p[0] = ("paren", p[2])


def p_atom_list_empty(p):
    """atom : LBRACKET RBRACKET"""
    p[0] = ("list", [])


def p_atom_list(p):
    """atom : LBRACKET expr_list RBRACKET"""
    p[0] = ("list", p[2])


def p_atom_dict_empty(p):
    """atom : LBRACE RBRACE"""
    p[0] = ("dict", [])


# ── Producción vacía ──────────────────────────────────────────────────────

def p_empty(p):
    """empty :"""
    p[0] = None


# ── Manejo de errores sintácticos ─────────────────────────────────────────

def p_error(p):
    global _symtab
    if p:
        _symtab.errors.append(
            f"Error sintáctico en línea {p.lineno}: token inesperado '{p.value}' ({p.type})"
        )
    else:
        _symtab.errors.append("Error sintáctico: fin de archivo inesperado")


# ---------------------------------------------------------------------------
# HELPERS INTERNOS
# ---------------------------------------------------------------------------

def _fold_tail(base: Any, tail: Any) -> Any:
    """
    Reconstruye el árbol desde la forma A' de la eliminación de recursión izquierda.
    Convierte la estructura de cola derecha en un árbol balanceado a izquierda.

    Ejemplo:  sum_tail = ("+", 3, ("-", 4, None))
              base     = 2
              resultado = ("-", ("+", 2, 3), 4)
    """
    if tail is None:
        return base
    op, right, rest = tail
    node = (op, base, right)
    return _fold_tail(node, rest)


def _expr_to_str(node: Any) -> str:
    """Convierte un nodo de expresión a string legible (para anotaciones)."""
    if node is None:
        return ""
    if isinstance(node, tuple):
        if node[0] == "name":
            return node[1]
        if node[0] == "access":
            base, trailer = node[1], node[2]
            if isinstance(trailer, tuple) and trailer[0] == "attr":
                return f"{_expr_to_str(base)}.{trailer[1]}"
        return str(node)
    return str(node)


def _detect_triton_calls(node: Any, func_name: str) -> None:
    """Recorre el AST buscando llamadas tipo tl.* y las registra en la tabla."""
    if not isinstance(node, tuple):
        return
    if node[0] == "access":
        base, trailer = node[1], node[2]
        if (
            isinstance(trailer, tuple)
            and trailer[0] == "call"
            and isinstance(base, tuple)
            and base[0] == "access"
        ):
            # Patrón: (access (access (name 'tl') (attr 'load')) (call ...))
            inner_base, inner_trailer = base[1], base[2]
            if (
                isinstance(inner_base, tuple)
                and inner_base[0] == "name"
                and inner_base[1] == "tl"
                and isinstance(inner_trailer, tuple)
                and inner_trailer[0] == "attr"
            ):
                call_name = f"tl.{inner_trailer[1]}"
                _symtab.add_triton_call(func_name, call_name)
    for child in node[1:]:
        _detect_triton_calls(child, func_name)


# ---------------------------------------------------------------------------
# PARSER PÚBLICO
# ---------------------------------------------------------------------------

_parser = None


def _build_parser() -> yacc.LRParser:
    return yacc.yacc(debug=False, write_tables=False, errorlog=yacc.NullLogger())


def parse(source: str) -> tuple[SymbolTable, Any]:
    """
    Parsea código fuente Triton y retorna (tabla_de_simbolos, ast).

    Args:
        source: código fuente como string.

    Returns:
        (SymbolTable, ast_node) — la tabla de símbolos llena y el AST.
    """
    global _parser, _symtab, _current_func

    if _parser is None:
        _parser = _build_parser()

    # Reiniciar estado global para este parse
    _symtab       = SymbolTable()
    _current_func = ""

    lexer = build_lexer()
    ast   = _parser.parse(source, lexer=lexer, tracking=True)

    # Post-proceso: detectar llamadas tl.* en el AST
    if ast is not None:
        for fname in _symtab.functions:
            _detect_triton_calls(ast, fname)

    return _symtab, ast
