"""
compiler/triton_lexer.py
========================
Lexer (análisis léxico) para código Triton-GPU usando PLY (Python Lex-Yacc).

FLUJO:
  código fuente (str)
      │
      ▼
  IndentLexer.input(source)
      │
      ▼  token() llamado por el parser en cada paso
  [NAME, DEF, COLON, NEWLINE, INDENT, DEDENT, ...]
      │
      ▼
  Parser (triton_parser.py)

INDENTACIÓN:
  Python (y Triton) usa espacios para delimitar bloques, no llaves {}.
  El lexer convierte eso en tokens INDENT / DEDENT usando una pila:

    stack = [0]          # pila de niveles de indentación

    Al ver \n + espacios al inicio de línea:
      nuevo = contar_espacios()
      si nuevo > stack.top  → push(nuevo),  emitir INDENT
      si nuevo < stack.top  → pop hasta igualar, emitir DEDENT por cada pop
      si nuevo == stack.top → solo emitir NEWLINE
      si estamos dentro de ( ) [ ] → ignorar la newline (continuación implícita)

    Al EOF → emitir DEDENT por cada nivel > 0 que quede en la pila

ALFABETO DEL LENGUAJE:
  Letras:    A-Z  a-z  _
  Dígitos:   0-9
  Operadores: + - * / % ** // << >> & | ^ ~ < > = ! @
  Delimitadores: ( ) [ ] { } : , . ;
  Espacio/Tab: solo significativo al inicio de línea
  Newline:   \n  (delimitador lógico de statements)
"""

from __future__ import annotations

import collections
import ply.lex as lex


# ---------------------------------------------------------------------------
# 1. KEYWORDS — palabras reservadas del lenguaje
#    Un NAME que coincide con una keyword se convierte en su token propio.
# ---------------------------------------------------------------------------
_KEYWORDS: dict[str, str] = {
    "and":      "AND",
    "as":       "AS",
    "assert":   "ASSERT",
    "break":    "BREAK",
    "class":    "CLASS",
    "continue": "CONTINUE",
    "def":      "DEF",
    "del":      "DEL",
    "elif":     "ELIF",
    "else":     "ELSE",
    "except":   "EXCEPT",
    "False":    "FALSE",
    "finally":  "FINALLY",
    "for":      "FOR",
    "from":     "FROM",
    "global":   "GLOBAL",
    "if":       "IF",
    "import":   "IMPORT",
    "in":       "IN",
    "is":       "IS",
    "lambda":   "LAMBDA",
    "None":     "NONE",
    "nonlocal": "NONLOCAL",
    "not":      "NOT",
    "or":       "OR",
    "pass":     "PASS",
    "raise":    "RAISE",
    "return":   "RETURN",
    "True":     "TRUE",
    "try":      "TRY",
    "while":    "WHILE",
    "with":     "WITH",
    "yield":    "YIELD",
}

# ---------------------------------------------------------------------------
# 2. LISTA DE TOKENS — todo lo que el lexer puede producir
#    PLY la necesita explícitamente.
# ---------------------------------------------------------------------------
tokens: tuple[str, ...] = (
    # Tokens de indentación (inyectados por IndentLexer, no por reglas PLY)
    "INDENT",
    "DEDENT",
    "NEWLINE",
    # Identificadores y literales
    "NAME",
    "NUMBER",
    "STRING",
    # ── Operadores de asignación compuesta (más largos primero) ──
    "DOUBLESTAREQ",   # **=
    "DOUBLESLASHEQ",  # //=
    "LSHIFTEQ",       # <<=
    "RSHIFTEQ",       # >>=
    "PLUSEQ",         # +=
    "MINUSEQ",        # -=
    "STAREQ",         # *=
    "SLASHEQ",        # /=
    "PERCENTEQ",      # %=
    "AMPEQ",          # &=
    "PIPEEQ",         # |=
    "CARETEQ",        # ^=
    # ── Operadores de comparación ──
    "EQEQ",           # ==
    "NOTEQ",          # !=
    "LTEQ",           # <=
    "GTEQ",           # >=
    # ── Operadores aritméticos dobles ──
    "DOUBLESTAR",     # **
    "DOUBLESLASH",    # //
    "LSHIFT",         # <<
    "RSHIFT",         # >>
    # ── Otros multi-char ──
    "ARROW",          # ->
    "WALRUS",         # :=
    "ELLIPSIS",       # ...
    # ── Operadores simples ──
    "PLUS",    "MINUS",   "STAR",    "SLASH",  "PERCENT",
    "AMP",     "PIPE",    "CARET",   "TILDE",
    "LT",      "GT",      "EQ",      "AT",
    # ── Delimitadores ──
    "LPAREN",    "RPAREN",
    "LBRACKET",  "RBRACKET",
    "LBRACE",    "RBRACE",
    "COLON",     "COMMA",   "DOT",   "SEMI",
) + tuple(_KEYWORDS.values())


# ---------------------------------------------------------------------------
# 3. REGLAS DEL LEXER PLY (raw — sin manejo de indentación)
#    Las reglas más largas deben ir PRIMERO para que PLY las priorice.
# ---------------------------------------------------------------------------

class _RawLexer:
    """Reglas PLY internas. IndentLexer envuelve esta clase."""

    tokens = tokens

    # ── Operadores compuestos de asignación (4+ chars) ──
    t_DOUBLESTAREQ  = r"\*\*="
    t_DOUBLESLASHEQ = r"//="
    t_LSHIFTEQ      = r"<<="
    t_RSHIFTEQ      = r">>="

    # ── Operadores compuestos de asignación (2+1 chars) ──
    t_PLUSEQ    = r"\+="
    t_MINUSEQ   = r"-="
    t_STAREQ    = r"\*="
    t_SLASHEQ   = r"/="
    t_PERCENTEQ = r"%="
    t_AMPEQ     = r"&="
    t_PIPEEQ    = r"\|="
    t_CARETEQ   = r"\^="

    # ── Comparadores (2 chars) ──
    t_EQEQ  = r"=="
    t_NOTEQ = r"!="
    t_LTEQ  = r"<="
    t_GTEQ  = r">="

    # ── Aritmética doble ──
    t_DOUBLESTAR  = r"\*\*"
    t_DOUBLESLASH = r"//"
    t_LSHIFT      = r"<<"
    t_RSHIFT      = r">>"

    # ── Otros multi-char ──
    t_ARROW   = r"->"
    t_WALRUS  = r":="
    t_ELLIPSIS = r"\.\.\."

    # ── Operadores simples (1 char) ──
    t_PLUS    = r"\+"
    t_MINUS   = r"-"
    t_STAR    = r"\*"
    t_SLASH   = r"/"
    t_PERCENT = r"%"
    t_AMP     = r"&"
    t_PIPE    = r"\|"
    t_CARET   = r"\^"
    t_TILDE   = r"~"
    t_LT      = r"<"
    t_GT      = r">"
    t_EQ      = r"="
    t_AT      = r"@"

    # ── Delimitadores ──
    t_LPAREN   = r"\("
    t_RPAREN   = r"\)"
    t_LBRACKET = r"\["
    t_RBRACKET = r"\]"
    t_LBRACE   = r"\{"
    t_RBRACE   = r"\}"
    t_COLON    = r":"
    t_COMMA    = r","
    t_DOT      = r"\."
    t_SEMI     = r";"

    # ── Identificadores y keywords ──
    def t_NAME(self, t):
        r"[A-Za-z_][A-Za-z0-9_]*"
        t.type = _KEYWORDS.get(t.value, "NAME")
        return t

    # ── Números (float antes de int para que '3.14' no quede como '3' + '.14') ──
    def t_NUMBER(self, t):
        r"""
        0[xX][0-9a-fA-F]+          # hexadecimal
        | 0[bB][01]+               # binario
        | 0[oO][0-7]+              # octal
        | [0-9]+\.[0-9]*([eE][+-]?[0-9]+)?   # float con parte entera
        | \.[0-9]+([eE][+-]?[0-9]+)?          # float sin parte entera
        | [0-9]+[eE][+-]?[0-9]+               # float solo exponente
        | [0-9]+                              # entero decimal
        """
        t.value = t.value  # guardamos el string tal cual
        return t

    # ── Strings (triple antes que simple para prioridad) ──
    def t_STRING(self, t):
        r"""
        [fFrRbBuU]*\"\"\"[\s\S]*?\"\"\"    # triple-double
        | [fFrRbBuU]*\'\'\'[\s\S]*?\'\'\'   # triple-single
        | [fFrRbBuU]*\"([^\"\\\n]|\\.)*\"   # double-quoted
        | [fFrRbBuU]*\'([^\'\\\n]|\\.)*\'   # single-quoted
        """
        return t

    # ── Newline — captura \n + espacios del inicio de la SIGUIENTE línea ──
    # Esto nos da el nivel de indentación de la siguiente línea en un solo token.
    def t_NEWLINE(self, t):
        r"\n[ \t]*"
        t.lexer.lineno += 1
        return t

    # ── Comentarios — se descartan ──
    def t_COMMENT(self, t):
        r"\#[^\n]*"
        pass  # no retornar nada = ignorar

    # ── Espacios y tabs en medio de línea — se descartan ──
    def t_WHITESPACE(self, t):
        r"[ \t]+"
        pass

    # ── Backslash continuation — línea lógica continúa ──
    def t_LINE_CONTINUATION(self, t):
        r"\\\n"
        t.lexer.lineno += 1

    def t_error(self, t):
        raise LexError(
            f"Carácter ilegal '{t.value[0]}' en línea {t.lexer.lineno}",
            t.lexer.lineno,
        )


# ---------------------------------------------------------------------------
# 4. IndentLexer — envuelve _RawLexer e inyecta INDENT / DEDENT
# ---------------------------------------------------------------------------

class LexError(Exception):
    def __init__(self, msg: str, lineno: int):
        super().__init__(msg)
        self.lineno = lineno


class IndentLexer:
    """
    Lexer público para el parser.

    Proporciona la interfaz que PLY yacc espera: .input(text) y .token().
    Internamente usa _RawLexer y post-procesa los tokens NEWLINE para
    inyectar INDENT / DEDENT según el algoritmo estándar de Python.

    REGLA CLAVE para líneas en blanco:
    Python (PEP 8 / Language Reference §2.1.8) ignora las líneas en blanco
    para propósitos de indentación. Solo las líneas con contenido real afectan
    la pila de indentación. Implementamos esto con un token de lookahead:
    si un NEWLINE viene seguido de otro NEWLINE, es una línea en blanco y
    no ajustamos la indentación.
    """

    def __init__(self) -> None:
        self._raw: lex.Lexer = lex.lex(object=_RawLexer(), debug=False, errorlog=lex.NullLogger())
        self._queue: collections.deque = collections.deque()
        self._indent_stack: list[int] = [0]
        self._paren_depth: int = 0   # profundidad de ( [ {
        self._errors: list[str] = []
        self._lookahead = None       # token raw en buffer para lookahead

    # ── PLY interface ──────────────────────────────────────────────────────

    def input(self, data: str) -> None:
        """Alimenta el texto fuente al lexer."""
        if data and not data.endswith("\n"):
            data += "\n"
        self._raw.input(data)
        self._queue.clear()
        self._indent_stack = [0]
        self._paren_depth = 0
        self._errors = []
        self._lookahead = None

    def _next_raw(self):
        """Obtiene el próximo token del lexer raw, usando el buffer si hay uno."""
        if self._lookahead is not None:
            tok = self._lookahead
            self._lookahead = None
            return tok
        return self._raw.token()

    def token(self):
        """
        Retorna el próximo token.
        El parser llama este método repetidamente hasta recibir None (EOF).
        """
        # Si hay tokens en la cola (p.ej. múltiples DEDENT pendientes), entrégarlos primero.
        if self._queue:
            return self._queue.popleft()

        tok = self._next_raw()

        # ── EOF ────────────────────────────────────────────────────────────
        if tok is None:
            if self._indent_stack[-1] > 0:
                nl = self._make_token("NEWLINE", "\n", self._raw.lineno)
                while self._indent_stack[-1] > 0:
                    self._indent_stack.pop()
                    self._queue.append(self._make_token("DEDENT", "", self._raw.lineno))
                return nl
            return None

        # ── Rastrear profundidad de paréntesis / corchetes / llaves ────────
        if tok.type in ("LPAREN", "LBRACKET", "LBRACE"):
            self._paren_depth += 1
        elif tok.type in ("RPAREN", "RBRACKET", "RBRACE"):
            self._paren_depth = max(0, self._paren_depth - 1)

        # ── Procesar NEWLINE ────────────────────────────────────────────────
        if tok.type == "NEWLINE":
            if self._paren_depth > 0:
                # Dentro de ( ) [ ] { } → newline se ignora
                return self.token()

            # LOOKAHEAD: mirar el siguiente token raw para detectar líneas en blanco.
            # Si la siguiente línea también empieza con NEWLINE, la línea actual es
            # en blanco → NO ajustar indentación (Python la ignora).
            next_tok = self._next_raw()

            if next_tok is not None and next_tok.type == "NEWLINE":
                # Línea en blanco: devolver solo el NEWLINE sin tocar la pila.
                # Buffear el siguiente NEWLINE para que sea procesado después.
                self._lookahead = next_tok
                tok.value = "\n"
                return tok

            # Buffear el siguiente token (es contenido real).
            if next_tok is not None:
                self._lookahead = next_tok

            # Calcular nivel de indentación de la SIGUIENTE línea real.
            raw_indent = tok.value[1:]
            indent_level = self._measure_indent(raw_indent)
            current = self._indent_stack[-1]
            lineno  = tok.lineno

            if indent_level > current:
                self._indent_stack.append(indent_level)
                tok.value = "\n"
                self._queue.append(self._make_token("INDENT", "", lineno))

            elif indent_level < current:
                tok.value = "\n"
                while self._indent_stack[-1] > indent_level:
                    self._indent_stack.pop()
                    self._queue.append(self._make_token("DEDENT", "", lineno))
                if self._indent_stack[-1] != indent_level:
                    self._errors.append(
                        f"Línea {lineno}: error de indentación "
                        f"(nivel {indent_level} no coincide con ningún bloque abierto)"
                    )

        return tok

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _measure_indent(s: str) -> int:
        """Convierte espacios/tabs a número de columna. 1 tab = 4 espacios."""
        col = 0
        for ch in s:
            if ch == " ":
                col += 1
            elif ch == "\t":
                col = (col // 4 + 1) * 4  # próximo múltiplo de 4
        return col

    @staticmethod
    def _make_token(type_: str, value: str, lineno: int) -> lex.LexToken:
        tok = lex.LexToken()
        tok.type    = type_
        tok.value   = value
        tok.lineno  = lineno
        tok.lexpos  = 0
        return tok

    # ── Atributo lineno — requerido por PLY yacc con tracking=True ────────

    @property
    def lineno(self) -> int:
        return self._raw.lineno

    @lineno.setter
    def lineno(self, value: int) -> None:
        self._raw.lineno = value

    # ── Atributo lexpos — requerido por PLY yacc ────────────────────────

    @property
    def lexpos(self) -> int:
        return self._raw.lexpos

    # ── Utilidad para depuración / tests ───────────────────────────────────

    def tokenize(self, source: str) -> list[lex.LexToken]:
        """Retorna la lista completa de tokens para un string dado."""
        self.input(source)
        result = []
        while True:
            tok = self.token()
            if tok is None:
                break
            result.append(tok)
        return result


# ---------------------------------------------------------------------------
# 5. Factory pública
# ---------------------------------------------------------------------------

def build_lexer() -> IndentLexer:
    """Construye y retorna un IndentLexer listo para usar."""
    return IndentLexer()
