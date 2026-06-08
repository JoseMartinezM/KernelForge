/*
 * triton.y
 * ========
 * Parser (analizador sintáctico) + Direct Syntax Translator para
 * el lenguaje Triton GPU kernel.  Implementado con UNIX yacc (bison).
 *
 * COMPILACIÓN:
 *   flex  triton.l          -> lex.yy.c
 *   bison -d triton.y       -> triton.tab.c  triton.tab.h
 *   gcc lex.yy.c triton.tab.c -lfl -o triton_compiler
 *
 * USO:
 *   ./triton_compiler < archivo.py
 *   ./triton_compiler   archivo.py
 *
 * ARQUITECTURA — DIRECT SYNTAX TRANSLATOR (DST):
 *   En un DST las acciones semánticas se ejecutan DURANTE el parsing,
 *   en cada reducción de la gramática.  No se construye un AST separado;
 *   la "traducción" ocurre al mismo tiempo que el análisis sintáctico.
 *
 *   fuente  →  tokens (lex)  →  parser (yacc)
 *                                   │  acciones semánticas en cada reducción
 *                                   ↓
 *                              tabla de símbolos + reporte de traducción
 *
 * GRAMÁTICA — ELIMINACIÓN DE RECURSIÓN IZQUIERDA:
 *   La técnica estándar convierte:
 *       A  → A α | β
 *   en:
 *       A  → β A'
 *       A' → α A' | ε
 *   Aplicada a todas las listas y expresiones binarias.
 *   Esto mantiene la gramática compatible con parsers LL y es el
 *   requisito pedagógico del curso TC-3048.
 *
 * CONFLICTOS SHIFT/REDUCE:
 *   Los operadores binarios generan conflictos S/R naturales.
 *   Se resuelven con la tabla %left/%right/%nonassoc de precedencia
 *   (de menor a mayor prioridad), sin necesidad de reescribir la gramática.
 */

/* =========================================================================
 * SECCIÓN 1 — DEFINICIONES
 * Código C de cabecera, declaraciones de tokens, precedencia.
 * ========================================================================= */

%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Prototipo que bison necesita para el lexer generado por flex */
int  yylex(void);
void yyerror(const char *msg);

/* ------------------------------------------------------------------
 * TABLA DE SÍMBOLOS DEL PARSER
 * Complementa la tabla del lexer: registra funciones Triton kernel,
 * sus parámetros, variables locales y llamadas a la API tl.*.
 * ------------------------------------------------------------------ */
#define MAX_FUNCS   256
#define MAX_PARAMS  64
#define MAX_LOCALS  256
#define MAX_CALLS   128
#define MAX_NAME    128

typedef struct {
    char nombre[MAX_NAME];
    char anotacion[MAX_NAME];   /* tipo/anotación, e.g. "tl.constexpr" */
    int  es_constexpr;
} Param;

typedef struct {
    char    nombre[MAX_NAME];
    int     linea;
    int     tiene_triton_jit;
    Param   params[MAX_PARAMS];
    int     num_params;
    char    locals[MAX_LOCALS][MAX_NAME];
    int     num_locals;
    char    triton_calls[MAX_CALLS][MAX_NAME];
    int     num_calls;
} FuncInfo;

static FuncInfo funciones[MAX_FUNCS];
static int      num_funciones = 0;

/* Función actualmente parseada (contexto para registrar locales) */
static char func_actual[MAX_NAME] = "";

/* Lista de errores y advertencias semánticas */
static char errores[512][256];
static int  num_errores = 0;
static char advertencias[512][256];
static int  num_advertencias = 0;

/* ------------------------------------------------------------------
 * Helpers de la tabla de símbolos del parser
 * ------------------------------------------------------------------ */

static FuncInfo *buscar_funcion(const char *nombre)
{
    int i;
    for (i = 0; i < num_funciones; i++)
        if (strcmp(funciones[i].nombre, nombre) == 0)
            return &funciones[i];
    return NULL;
}

static void declarar_funcion(const char *nombre, int linea, int tiene_jit)
{
    if (buscar_funcion(nombre)) {
        if (num_errores < 512)
            snprintf(errores[num_errores++], 256,
                     "Línea %d: función '%s' ya declarada", linea, nombre);
        return;
    }
    if (num_funciones >= MAX_FUNCS) return;
    memset(&funciones[num_funciones], 0, sizeof(FuncInfo));
    strncpy(funciones[num_funciones].nombre, nombre, MAX_NAME - 1);
    funciones[num_funciones].linea            = linea;
    funciones[num_funciones].tiene_triton_jit = tiene_jit;
    num_funciones++;
    strncpy(func_actual, nombre, MAX_NAME - 1);
}

static void agregar_parametro(const char *func, const char *nombre,
                               const char *ann, int es_ce)
{
    FuncInfo *f = buscar_funcion(func);
    if (!f || f->num_params >= MAX_PARAMS) return;
    strncpy(f->params[f->num_params].nombre,    nombre, MAX_NAME - 1);
    strncpy(f->params[f->num_params].anotacion, ann,    MAX_NAME - 1);
    f->params[f->num_params].es_constexpr = es_ce;
    f->num_params++;
}

static void agregar_local(const char *func, const char *var)
{
    int i;
    FuncInfo *f = buscar_funcion(func);
    if (!f) return;
    for (i = 0; i < f->num_locals; i++)
        if (strcmp(f->locals[i], var) == 0) return;
    if (f->num_locals < MAX_LOCALS)
        strncpy(f->locals[f->num_locals++], var, MAX_NAME - 1);
}

static void agregar_llamada_triton(const char *func, const char *call)
{
    int i;
    FuncInfo *f = buscar_funcion(func);
    if (!f) return;
    for (i = 0; i < f->num_calls; i++)
        if (strcmp(f->triton_calls[i], call) == 0) return;
    if (f->num_calls < MAX_CALLS)
        strncpy(f->triton_calls[f->num_calls++], call, MAX_NAME - 1);
}

/* ------------------------------------------------------------------
 * Impresión del reporte de traducción
 * ------------------------------------------------------------------ */
static void imprimir_reporte(void)
{
    int i, j;
    printf("\n");
    printf("============================================================\n");
    printf("  REPORTE DE TRADUCCIÓN — Triton Kernel Validator\n");
    printf("============================================================\n");

    if (num_funciones == 0) {
        printf("  (no se encontraron definiciones de función)\n");
    } else {
        for (i = 0; i < num_funciones; i++) {
            FuncInfo *f = &funciones[i];
            printf("\n  %s %s\n",
                   f->tiene_triton_jit ? "[KERNEL]" : "[FUNC]  ",
                   f->nombre);
            printf("       Línea       : %d\n", f->linea);
            printf("       @triton.jit : %s\n",
                   f->tiene_triton_jit ? "SI" : "NO");

            if (f->num_params > 0) {
                printf("       Parámetros  : ");
                for (j = 0; j < f->num_params; j++) {
                    if (j) printf(", ");
                    printf("%s", f->params[j].nombre);
                    if (f->params[j].anotacion[0])
                        printf(":%s", f->params[j].anotacion);
                    if (f->params[j].es_constexpr)
                        printf("(constexpr)");
                }
                printf("\n");
            }

            if (f->num_calls > 0) {
                printf("       API Triton  : ");
                for (j = 0; j < f->num_calls; j++) {
                    if (j) printf(", ");
                    printf("%s", f->triton_calls[j]);
                }
                printf("\n");
            }

            if (f->num_locals > 0) {
                printf("       Locales     : ");
                for (j = 0; j < f->num_locals; j++) {
                    if (j) printf(", ");
                    printf("%s", f->locals[j]);
                }
                printf("\n");
            }
        }
    }

    if (num_errores > 0) {
        printf("\n  -- ERRORES --\n");
        for (i = 0; i < num_errores; i++)
            printf("  X %s\n", errores[i]);
    }
    if (num_advertencias > 0) {
        printf("\n  -- ADVERTENCIAS --\n");
        for (i = 0; i < num_advertencias; i++)
            printf("  ! %s\n", advertencias[i]);
    }

    printf("\n  Estado: %s\n", num_errores == 0 ? "VALIDO" : "INVALIDO");
    printf("============================================================\n");
}

%}

/* ------------------------------------------------------------------
 * TIPO SEMÁNTICO DE LOS NODOS (yylval)
 * Todos los nodos del árbol son strings; usamos un buffer estático.
 * ------------------------------------------------------------------ */
%union {
    char sval[512];
    int  ival;
}

/* ------------------------------------------------------------------
 * DECLARACIÓN DE TOKENS
 * Deben coincidir exactamente con los tipos que retorna el lexer.
 * El lexer (triton.l en modo integrado con bison) retornará estos
 * valores numéricos definidos en triton.tab.h.
 * ------------------------------------------------------------------ */

/* Literales */
%token <sval> NAME NUMBER_INT NUMBER_FLOAT NUMBER_HEX NUMBER_BIN NUMBER_OCT STRING

/* Keywords */
%token AND AS ASSERT BREAK CLASS CONTINUE DEF DEL ELIF ELSE EXCEPT
%token FALSE FINALLY FOR FROM GLOBAL IF IMPORT IN IS LAMBDA NONE
%token NONLOCAL NOT OR PASS RAISE RETURN TRUE TRY WHILE WITH YIELD

/* Operadores de asignación compuesta */
%token DOUBLESTAREQ DOUBLESLASHEQ LSHIFTEQ RSHIFTEQ
%token PLUSEQ MINUSEQ STAREQ SLASHEQ PERCENTEQ AMPEQ PIPEEQ CARETEQ

/* Comparadores */
%token EQEQ NOTEQ LTEQ GTEQ

/* Operadores aritméticos dobles */
%token DOUBLESTAR DOUBLESLASH LSHIFT RSHIFT

/* Otros multi-char */
%token ARROW WALRUS ELLIPSIS

/* Operadores simples */
%token PLUS MINUS STAR SLASH PERCENT AMP PIPE CARET TILDE LT GT EQ AT

/* Delimitadores */
%token LPAREN RPAREN LBRACKET RBRACKET LBRACE RBRACE COLON COMMA DOT SEMI

/* Control de bloque (emitidos por el lexer Triton) */
%token NEWLINE INDENT DEDENT

/* ------------------------------------------------------------------
 * TABLA DE PRECEDENCIA (de menor a mayor)
 * Resuelve conflictos shift/reduce en expresiones sin reescribir reglas.
 * ------------------------------------------------------------------ */
%left     OR
%left     AND
%right    NOT
%left     IN IS
%left     LT GT LTEQ GTEQ EQEQ NOTEQ
%left     PIPE
%left     CARET
%left     AMP
%left     LSHIFT RSHIFT
%left     PLUS MINUS
%left     STAR SLASH DOUBLESLASH PERCENT
%right    UMINUS UTILDE UPLUS
%right    DOUBLESTAR
%left     DOT LPAREN LBRACKET

%type <sval> name_expr decorator_str

/* Regla inicial */
%start program

%%

/* =========================================================================
 * SECCIÓN 2 — GRAMÁTICA Y ACCIONES SEMÁNTICAS
 *
 * Notación:
 *   $$ = valor del símbolo de la izquierda (lo que "retorna" la regla)
 *   $1 = primer símbolo de la derecha, $2 = segundo, etc.
 *   @1 = información de posición del primer símbolo (línea)
 *
 * Las acciones semánticas van entre llaves { } al final (o en medio) de
 * cada producción.  Se ejecutan al REDUCIR esa producción (no al parsear).
 * ========================================================================= */


/* ---- Programa ---------------------------------------------------------- */

program
    : stmt_seq
    { imprimir_reporte(); }
    ;


/* ---- Secuencia de sentencias (recursión DERECHA: sin rec. izq.) -------- */

stmt_seq
    : stmt stmt_seq          /* A → β A' */
    | NEWLINE stmt_seq       /* líneas en blanco entre sentencias */
    | /* vacío */
    ;


/* ---- Sentencia --------------------------------------------------------- */

stmt
    : simple_line
    | compound_stmt
    ;

simple_line
    : simple_stmt NEWLINE
    | simple_stmt SEMI simple_stmt NEWLINE
    ;

simple_stmt
    : assign_stmt
    | aug_assign_stmt
    | ann_assign_stmt
    | return_stmt
    | assert_stmt
    | pass_stmt
    | import_stmt
    | expr_stmt
    ;


/* ---- Asignación  target = expr ---------------------------------------- */

assign_stmt
    : target_expr EQ expr_list
    {
        /*
         * ACCIÓN SEMÁNTICA: si el target es un nombre simple y estamos
         * dentro de una función, registrarlo como variable local.
         */
        if (func_actual[0] != '\0')
            agregar_local(func_actual, $1);
    }
    | target_expr EQ assign_stmt   /* asignación encadenada: a = b = expr */
    ;

aug_assign_stmt
    : target_expr aug_op expr
    ;

aug_op
    : PLUSEQ | MINUSEQ | STAREQ | SLASHEQ | DOUBLESLASHEQ
    | PERCENTEQ | AMPEQ | PIPEEQ | CARETEQ | LSHIFTEQ | RSHIFTEQ
    | DOUBLESTAREQ
    ;

ann_assign_stmt
    : target_expr COLON expr
    | target_expr COLON expr EQ expr
    ;


/* ---- Return / Assert / Pass ------------------------------------------- */

return_stmt
    : RETURN
    | RETURN expr_list
    ;

assert_stmt
    : ASSERT expr
    | ASSERT expr COMMA expr
    ;

pass_stmt
    : PASS
    ;

expr_stmt
    : expr
    ;


/* ---- Import ------------------------------------------------------------ */

import_stmt
    : IMPORT dotted_name
    | IMPORT dotted_name AS NAME
    | FROM dotted_name IMPORT NAME
    | FROM dotted_name IMPORT NAME AS NAME
    | FROM dotted_name IMPORT STAR
    ;

dotted_name
    : NAME dotted_name_tail
    ;

dotted_name_tail
    : DOT NAME dotted_name_tail
    | /* vacío */
    ;


/* ---- Sentencias compuestas -------------------------------------------- */

compound_stmt
    : funcdef
    | if_stmt
    | for_stmt
    | while_stmt
    | with_stmt
    ;


/* ---- Definición de función -------------------------------------------- */

funcdef
    : decorator_seq DEF NAME LPAREN param_seq RPAREN COLON suite
    {
        /*
         * ACCIÓN SEMÁNTICA: registrar la función en la tabla de símbolos.
         * $1 contiene la cadena del decorador (ej. "triton.jit").
         * $3 es el nombre de la función.
         */
        int tiene_jit = (strstr($1, "triton.jit") != NULL) ? 1 : 0;
        declarar_funcion($3, @3.first_line, tiene_jit);
        if (!tiene_jit && num_advertencias < 512)
            snprintf(advertencias[num_advertencias++], 256,
                     "Función '%s' (línea %d) declarada sin @triton.jit",
                     $3, @3.first_line);
        func_actual[0] = '\0';   /* salir del contexto de función */
    }
    | DEF NAME LPAREN param_seq RPAREN COLON suite
    {
        declarar_funcion($2, @2.first_line, 0);
        if (num_advertencias < 512)
            snprintf(advertencias[num_advertencias++], 256,
                     "Función '%s' (línea %d) sin decorador @triton.jit",
                     $2, @2.first_line);
        func_actual[0] = '\0';
    }
    | decorator_seq DEF NAME LPAREN param_seq RPAREN ARROW expr COLON suite
    {
        int tiene_jit = (strstr($1, "triton.jit") != NULL) ? 1 : 0;
        declarar_funcion($3, @3.first_line, tiene_jit);
        func_actual[0] = '\0';
    }
    ;


/* ---- Decoradores ------------------------------------------------------- */

decorator_seq
    : decorator decorator_seq
    {
        /* Concatenar decoradores para pasarlos a funcdef */
        snprintf($$, sizeof($$), "%s %s", $1, $2);
    }
    | decorator
    { strncpy($$, $1, sizeof($$) - 1); }
    ;

decorator
    : AT decorator_str NEWLINE
    { strncpy($$, $2, sizeof($$) - 1); }
    ;

/* decorator_str captura expresiones de decorador como "triton.jit" */
decorator_str
    : NAME
    { strncpy($$, $1, sizeof($$) - 1); }
    | decorator_str DOT NAME
    { snprintf($$, sizeof($$), "%s.%s", $1, $3); }
    | decorator_str LPAREN arg_seq RPAREN
    { snprintf($$, sizeof($$), "%s(...)", $1); }
    ;


/* ---- Suite (bloque indentado) ----------------------------------------- */
/*
 * PREGUNTA ORAL: "¿Cómo representas los bloques?"
 * RESPUESTA: El lexer emite INDENT al entrar y DEDENT al salir del bloque.
 * La gramática siempre es: NEWLINE INDENT <cuerpo> DEDENT.
 */
suite
    : NEWLINE INDENT stmt_seq DEDENT
    | NEWLINE INDENT NEWLINE stmt_seq DEDENT   /* línea en blanco al inicio */
    ;


/* ---- If ---------------------------------------------------------------- */

if_stmt
    : IF expr COLON suite else_part
    ;

else_part
    : ELIF expr COLON suite else_part
    | ELSE COLON suite
    | /* vacío */
    ;


/* ---- For --------------------------------------------------------------- */

for_stmt
    : FOR target_expr IN expr COLON suite
    | FOR target_expr IN expr COLON suite ELSE COLON suite
    ;


/* ---- While ------------------------------------------------------------- */

while_stmt
    : WHILE expr COLON suite
    | WHILE expr COLON suite ELSE COLON suite
    ;


/* ---- With -------------------------------------------------------------- */

with_stmt
    : WITH with_item COLON suite
    ;

with_item
    : expr
    | expr AS target_expr
    ;


/* ---- Parámetros de función -------------------------------------------- */

param_seq
    : param param_seq_tail   /* A → β A' */
    | /* vacío */
    ;

param_seq_tail
    : COMMA param param_seq_tail
    | COMMA          /* coma final permitida */
    | /* vacío */
    ;

param
    : NAME
    {
        if (func_actual[0])
            agregar_parametro(func_actual, $1, "", 0);
    }
    | NAME COLON expr
    {
        /*
         * ACCIÓN SEMÁNTICA: detectar parámetro constexpr.
         * Si la anotación contiene "constexpr", marcarlo.
         */
        int es_ce = (strstr($3, "constexpr") != NULL) ? 1 : 0;
        if (func_actual[0])
            agregar_parametro(func_actual, $1, $3, es_ce);
    }
    | NAME EQ expr
    {
        if (func_actual[0])
            agregar_parametro(func_actual, $1, "", 0);
    }
    | NAME COLON expr EQ expr
    {
        int es_ce = (strstr($3, "constexpr") != NULL) ? 1 : 0;
        if (func_actual[0])
            agregar_parametro(func_actual, $1, $3, es_ce);
    }
    | STAR NAME
    | DOUBLESTAR NAME
    ;


/* ---- Lista de expresiones --------------------------------------------- */

expr_list
    : expr expr_list_tail
    ;

expr_list_tail
    : COMMA expr expr_list_tail
    | COMMA          /* coma final */
    | /* vacío */
    ;

target_expr
    : primary
    { strncpy($$, $1, sizeof($$) - 1); }
    ;


/* ---- Expresiones — jerarquía de precedencia ---------------------------- */
/*
 * ELIMINACIÓN DE RECURSIÓN IZQUIERDA aplicada a todos los niveles:
 *   expr → cond_expr
 *   cond_expr → or_expr | or_expr IF or_expr ELSE cond_expr
 *   or_expr → and_expr or_tail;  or_tail → OR and_expr or_tail | ε
 *   ... y así para cada nivel.
 */

expr
    : cond_expr
    { strncpy($$, $1, sizeof($$) - 1); }
    ;

cond_expr
    : or_expr IF or_expr ELSE cond_expr
    { strncpy($$, "ternary", sizeof($$) - 1); }
    | or_expr
    { strncpy($$, $1, sizeof($$) - 1); }
    ;

or_expr
    : and_expr or_tail
    { strncpy($$, $1, sizeof($$) - 1); }
    ;

or_tail
    : OR and_expr or_tail
    | /* vacío */
    ;

and_expr
    : not_expr and_tail
    { strncpy($$, $1, sizeof($$) - 1); }
    ;

and_tail
    : AND not_expr and_tail
    | /* vacío */
    ;

not_expr
    : NOT not_expr
    { strncpy($$, "not", sizeof($$) - 1); }
    | comparison
    { strncpy($$, $1, sizeof($$) - 1); }
    ;

comparison
    : bitor comp_op bitor
    { strncpy($$, "cmp", sizeof($$) - 1); }
    | bitor
    { strncpy($$, $1, sizeof($$) - 1); }
    ;

comp_op
    : EQEQ | NOTEQ | LT | GT | LTEQ | GTEQ | IN | IS
    | NOT IN
    | IS NOT
    ;

bitor
    : bitxor bitor_tail
    { strncpy($$, $1, sizeof($$) - 1); }
    ;

bitor_tail
    : PIPE bitxor bitor_tail
    | /* vacío */
    ;

bitxor
    : bitand bitxor_tail
    { strncpy($$, $1, sizeof($$) - 1); }
    ;

bitxor_tail
    : CARET bitand bitxor_tail
    | /* vacío */
    ;

bitand
    : shift bitand_tail
    { strncpy($$, $1, sizeof($$) - 1); }
    ;

bitand_tail
    : AMP shift bitand_tail
    | /* vacío */
    ;

shift
    : sum shift_tail
    { strncpy($$, $1, sizeof($$) - 1); }
    ;

shift_tail
    : LSHIFT sum shift_tail
    | RSHIFT sum shift_tail
    | /* vacío */
    ;

sum
    : term sum_tail
    { strncpy($$, $1, sizeof($$) - 1); }
    ;

sum_tail
    : PLUS  term sum_tail
    | MINUS term sum_tail
    | /* vacío */
    ;

term
    : factor term_tail
    { strncpy($$, $1, sizeof($$) - 1); }
    ;

term_tail
    : STAR        factor term_tail
    | SLASH       factor term_tail
    | DOUBLESLASH factor term_tail
    | PERCENT     factor term_tail
    | AT          factor term_tail
    | /* vacío */
    ;

factor
    : PLUS   factor %prec UPLUS
    { strncpy($$, "uplus", sizeof($$) - 1); }
    | MINUS  factor %prec UMINUS
    { strncpy($$, "uminus", sizeof($$) - 1); }
    | TILDE  factor %prec UTILDE
    { strncpy($$, "invert", sizeof($$) - 1); }
    | power
    { strncpy($$, $1, sizeof($$) - 1); }
    ;

power
    : primary DOUBLESTAR factor
    { strncpy($$, "pow", sizeof($$) - 1); }
    | primary
    { strncpy($$, $1, sizeof($$) - 1); }
    ;


/* ---- Primary: atom + trailers (acceso, llamada, subscript) ------------ */
/*
 * primary → atom trailer_seq
 * trailer_seq → trailer trailer_seq | ε    (A → α A' | ε)
 */

primary
    : atom trailer_seq
    { strncpy($$, $1, sizeof($$) - 1); }
    ;

trailer_seq
    : trailer trailer_seq
    | /* vacío */
    ;

trailer
    : DOT NAME
    {
        /*
         * ACCIÓN SEMÁNTICA: detectar llamadas tl.* al combinar con call.
         * Se detecta el patrón tl.<nombre>(...) cuando el atom anterior
         * es "tl" y este trailer es DOT NAME seguido de LPAREN.
         * La detección completa se hace en el trailer de llamada.
         */
    }
    | LPAREN arg_seq RPAREN
    {
        /*
         * ACCIÓN SEMÁNTICA: si el primary anterior era tl.algo,
         * registrar la llamada Triton.  En este DST simplificado
         * usamos una heurística: registrar cualquier acceso con
         * prefijo "tl." que llegue aquí como llamada Triton.
         */
        if (func_actual[0] && strncmp($0, "tl.", 3) == 0)
            agregar_llamada_triton(func_actual, $0);
    }
    | LBRACKET slice_seq RBRACKET
    ;


/* ---- Argumentos de llamada a función ---------------------------------- */

arg_seq
    : arg arg_seq_tail
    | /* vacío */
    ;

arg_seq_tail
    : COMMA arg arg_seq_tail
    | COMMA          /* coma final */
    | /* vacío */
    ;

arg
    : expr
    | NAME EQ expr       /* argumento keyword */
    | STAR expr          /* *args */
    | DOUBLESTAR expr    /* **kwargs */
    ;


/* ---- Slices ------------------------------------------------------------ */

slice_seq
    : slice_item slice_seq_tail
    | /* vacío */
    ;

slice_seq_tail
    : COMMA slice_item slice_seq_tail
    | COMMA
    | /* vacío */
    ;

slice_item
    : expr COLON expr COLON expr
    | expr COLON expr
    | expr COLON
    | COLON expr
    | COLON
    | expr
    ;


/* ---- Átomos (valores terminales) -------------------------------------- */

atom
    : NAME
    { strncpy($$, $1, sizeof($$) - 1); }
    | NUMBER_INT
    { strncpy($$, $1, sizeof($$) - 1); }
    | NUMBER_FLOAT
    { strncpy($$, $1, sizeof($$) - 1); }
    | NUMBER_HEX
    { strncpy($$, $1, sizeof($$) - 1); }
    | NUMBER_BIN
    { strncpy($$, $1, sizeof($$) - 1); }
    | NUMBER_OCT
    { strncpy($$, $1, sizeof($$) - 1); }
    | STRING
    { strncpy($$, $1, sizeof($$) - 1); }
    | TRUE
    { strncpy($$, "True", sizeof($$) - 1); }
    | FALSE
    { strncpy($$, "False", sizeof($$) - 1); }
    | NONE
    { strncpy($$, "None", sizeof($$) - 1); }
    | ELLIPSIS
    { strncpy($$, "...", sizeof($$) - 1); }
    | LPAREN RPAREN
    { strncpy($$, "tuple()", sizeof($$) - 1); }
    | LPAREN expr_list RPAREN
    { strncpy($$, "paren", sizeof($$) - 1); }
    | LBRACKET RBRACKET
    { strncpy($$, "list()", sizeof($$) - 1); }
    | LBRACKET expr_list RBRACKET
    { strncpy($$, "list", sizeof($$) - 1); }
    | LBRACE RBRACE
    { strncpy($$, "dict()", sizeof($$) - 1); }
    ;

/* ---- Nombre para acceso a atributos en expresiones -------------------- */
name_expr
    : NAME
    { strncpy($$, $1, sizeof($$) - 1); }
    | name_expr DOT NAME
    { snprintf($$, sizeof($$), "%s.%s", $1, $3); }
    ;

%%

/* =========================================================================
 * SECCIÓN 3 — CÓDIGO DE USUARIO
 * ========================================================================= */

/*
 * yyerror
 * -------
 * Llamada por bison cuando encuentra un error sintáctico.
 * Registra el error en la lista global para incluirlo en el reporte.
 */
void yyerror(const char *msg)
{
    extern int yylineno;
    if (num_errores < 512)
        snprintf(errores[num_errores++], 256,
                 "Error sintáctico en línea %d: %s", yylineno, msg);
    fprintf(stderr, "ERROR SINTÁCTICO (línea %d): %s\n", yylineno, msg);
}

/*
 * main
 * ----
 * Punto de entrada. Configura la entrada y lanza el parser.
 * yyparse() llama internamente a yylex() (generado por flex)
 * cada vez que necesita el siguiente token.
 */
int main(int argc, char *argv[])
{
    extern FILE *yyin;
    FILE *entrada = NULL;

    if (argc > 1) {
        entrada = fopen(argv[1], "r");
        if (!entrada) {
            fprintf(stderr, "ERROR: no se pudo abrir '%s'\n", argv[1]);
            return 1;
        }
        yyin = entrada;
    }

    /* yyparse() ejecuta el análisis léxico + sintáctico + semántico.
     * Al terminar, la acción en 'program' llama imprimir_reporte(). */
    int resultado = yyparse();

    if (entrada) fclose(entrada);

    return resultado;
}
