/*
 * triton.y
 * ========
 * Parser (analizador sintáctico) + Direct Syntax Translator para
 * el lenguaje Triton GPU kernel.  Implementado con UNIX yacc (bison).
 *
 * COMPILACIÓN:
 *   bison -d triton.y       -> triton.tab.c  triton.tab.h
 *   flex  triton.l          -> lex.yy.c
 *   gcc lex.yy.c triton.tab.c -lfl -o triton_compiler
 *
 * ARQUITECTURA — DIRECT SYNTAX TRANSLATOR (DST):
 *   Las acciones semánticas se ejecutan DURANTE el parsing, en cada
 *   reducción.  No se construye un AST separado.
 *
 *   fuente  →  tokens (lex)  →  parser (yacc con acciones semánticas)
 *                                         ↓
 *                               tabla de símbolos + reporte
 *
 * GRAMÁTICA — sin recursión izquierda:
 *   Técnica: A → A α | β  se convierte en  A → β A'  y  A' → α A' | ε
 *   Aplicada a todas las listas y expresiones binarias.
 */

/* =========================================================================
 * SECCIÓN 1 — DEFINICIONES
 * ========================================================================= */

%locations   /* habilita @N.first_line para rastrear números de línea */

%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* triton_next_token() esta definido en triton.l (seccion 3).
 * Drena la cola INDENT/DEDENT antes de llamar al scanner raw.
 * Redefinimos yylex para que bison use nuestro wrapper. */
extern int triton_next_token(void);
#define yylex triton_next_token
void yyerror(const char *msg);

/* ------------------------------------------------------------------
 * TABLA DE SÍMBOLOS
 * Registra funciones Triton kernel: parámetros, locales, API calls.
 * ------------------------------------------------------------------ */
#define MAX_FUNCS  256
#define MAX_ITEMS  128
#define MAX_NAME   256

typedef struct {
    char nombre[MAX_NAME];
    char anotacion[MAX_NAME];
    int  es_constexpr;
} Param;

typedef struct {
    char  nombre[MAX_NAME];
    int   linea;
    int   tiene_triton_jit;
    Param params[MAX_ITEMS];
    int   num_params;
    char  locals[MAX_ITEMS][MAX_NAME];
    int   num_locals;
    char  calls[MAX_ITEMS][MAX_NAME];
    int   num_calls;
} FuncInfo;

static FuncInfo funcs[MAX_FUNCS];
static int      num_funcs = 0;
static char     func_actual[MAX_NAME] = "";

static char errores[256][MAX_NAME];
static int  num_errores = 0;
static char avisos[256][MAX_NAME];
static int  num_avisos = 0;

/* ---- helpers ---- */
static FuncInfo *buscar_func(const char *n)
{
    int i;
    for (i = 0; i < num_funcs; i++)
        if (strcmp(funcs[i].nombre, n) == 0) return &funcs[i];
    return NULL;
}

static void declarar_func(const char *nombre, int linea, int jit)
{
    if (buscar_func(nombre)) {
        snprintf(errores[num_errores++], MAX_NAME,
                 "Linea %d: funcion '%s' ya declarada", linea, nombre);
        return;
    }
    if (num_funcs >= MAX_FUNCS) return;
    memset(&funcs[num_funcs], 0, sizeof(FuncInfo));
    strncpy(funcs[num_funcs].nombre, nombre, MAX_NAME - 1);
    funcs[num_funcs].linea            = linea;
    funcs[num_funcs].tiene_triton_jit = jit;
    num_funcs++;
    strncpy(func_actual, nombre, MAX_NAME - 1);
}

static void add_param(const char *nombre, const char *ann, int ce)
{
    FuncInfo *f = buscar_func(func_actual);
    if (!f || f->num_params >= MAX_ITEMS) return;
    strncpy(f->params[f->num_params].nombre,    nombre, MAX_NAME - 1);
    strncpy(f->params[f->num_params].anotacion, ann,    MAX_NAME - 1);
    f->params[f->num_params].es_constexpr = ce;
    f->num_params++;
}

static void add_local(const char *var)
{
    int i;
    FuncInfo *f = buscar_func(func_actual);
    if (!f) return;
    for (i = 0; i < f->num_locals; i++)
        if (strcmp(f->locals[i], var) == 0) return;
    if (f->num_locals < MAX_ITEMS)
        strncpy(f->locals[f->num_locals++], var, MAX_NAME - 1);
}

static void add_triton_call(const char *call)
{
    int i;
    FuncInfo *f = buscar_func(func_actual);
    if (!f) return;
    for (i = 0; i < f->num_calls; i++)
        if (strcmp(f->calls[i], call) == 0) return;
    if (f->num_calls < MAX_ITEMS)
        strncpy(f->calls[f->num_calls++], call, MAX_NAME - 1);
}

static void imprimir_reporte(void)
{
    int i, j;
    printf("\n");
    printf("============================================================\n");
    printf("  REPORTE DE TRADUCCION — Triton Kernel Validator\n");
    printf("============================================================\n");

    if (num_funcs == 0) {
        printf("  (no se encontraron definiciones de funcion)\n");
    } else {
        for (i = 0; i < num_funcs; i++) {
            FuncInfo *f = &funcs[i];
            printf("\n  %s %s\n",
                   f->tiene_triton_jit ? "[KERNEL]" : "[FUNC]  ",
                   f->nombre);
            printf("       Linea       : %d\n", f->linea);
            printf("       @triton.jit : %s\n",
                   f->tiene_triton_jit ? "SI" : "NO");
            if (f->num_params > 0) {
                printf("       Parametros  : ");
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
                    printf("%s", f->calls[j]);
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
    if (num_avisos > 0) {
        printf("\n  -- ADVERTENCIAS --\n");
        for (i = 0; i < num_avisos; i++)
            printf("  ! %s\n", avisos[i]);
    }
    printf("\n  Estado: %s\n", num_errores == 0 ? "VALIDO" : "INVALIDO");
    printf("============================================================\n");
}
%}

/* ------------------------------------------------------------------
 * TIPO SEMÁNTICO
 * sval: string para lexemas e identificadores
 * ival: entero para flags booleanos
 * ------------------------------------------------------------------ */
%union {
    char sval[512];
    int  ival;
}

/* ------------------------------------------------------------------
 * TOKENS CON VALOR STRING
 * ------------------------------------------------------------------ */
%token <sval> NAME STRING
%token <sval> NUMBER_INT NUMBER_FLOAT NUMBER_HEX NUMBER_BIN NUMBER_OCT

/* ------------------------------------------------------------------
 * TOKENS SIN VALOR (keywords y operadores)
 * ------------------------------------------------------------------ */
%token AND AS ASSERT BREAK CLASS CONTINUE DEF DEL ELIF ELSE EXCEPT
%token FALSE FINALLY FOR FROM GLOBAL IF IMPORT IN IS LAMBDA NONE
%token NONLOCAL NOT OR PASS RAISE RETURN TRUE TRY WHILE WITH YIELD
%token DOUBLESTAREQ DOUBLESLASHEQ LSHIFTEQ RSHIFTEQ
%token PLUSEQ MINUSEQ STAREQ SLASHEQ PERCENTEQ AMPEQ PIPEEQ CARETEQ
%token EQEQ NOTEQ LTEQ GTEQ
%token DOUBLESTAR DOUBLESLASH LSHIFT RSHIFT
%token ARROW WALRUS ELLIPSIS
%token PLUS MINUS STAR SLASH PERCENT AMP PIPE CARET TILDE LT GT EQ AT
%token LPAREN RPAREN LBRACKET RBRACKET LBRACE RBRACE
%token COLON COMMA DOT SEMI
%token NEWLINE INDENT DEDENT

/* ------------------------------------------------------------------
 * TIPOS DE NO-TERMINALES QUE PROPAGAN STRINGS
 * Todos los no-terminales que usan $$ o $N como cadena necesitan
 * declaración explícita de tipo.
 * ------------------------------------------------------------------ */
%type <sval> decorator_seq decorator decorator_str decorator_str_tail
%type <sval> target_expr
%type <sval> expr cond_expr or_expr and_expr not_expr comparison
%type <sval> bitor bitxor bitand shift sum term factor power primary atom
%type <sval> trailer_seq trailer
%type <sval> expr_list

/* ------------------------------------------------------------------
 * PRECEDENCIA (menor → mayor)
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
%left     STAR SLASH DOUBLESLASH PERCENT AT
%right    UMINUS UTILDE UPLUS
%right    DOUBLESTAR
%left     DOT LPAREN LBRACKET

%start program

%%

/* =========================================================================
 * SECCIÓN 2 — GRAMÁTICA Y ACCIONES SEMÁNTICAS
 * ========================================================================= */

program
    : stmt_seq
      { imprimir_reporte(); }
    ;

/* ---- Secuencia de sentencias (recursión derecha) ----------------------- */

stmt_seq
    : stmt stmt_seq
    | NEWLINE stmt_seq
    | /* vacío */
    ;

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

/* ---- Asignación ------------------------------------------------------- */

assign_stmt
    : target_expr EQ expr_list
      {
          if (func_actual[0]) {
              add_local($1);
              /* Detectar llamadas a la API de Triton en el lado derecho */
              if (strncmp($3, "tl.", 3) == 0) add_triton_call($3);
          }
      }
    | target_expr EQ assign_stmt
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

return_stmt
    : RETURN
    | RETURN expr_list
    ;

assert_stmt
    : ASSERT expr
    | ASSERT expr COMMA expr
    ;

pass_stmt : PASS ;

expr_stmt
    : expr
      {
          /* Sentencias de expresion que son llamadas tl.* (ej. tl.store) */
          if (func_actual[0] && strncmp($1, "tl.", 3) == 0)
              add_triton_call($1);
      }
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
/*
 * IMPORTANTE: se usa mid-rule action justo despues de ver DEF NAME.
 * Esto declara la funcion (seteando func_actual) ANTES de parsear
 * param_seq, de modo que add_param() encuentre func_actual configurado.
 *
 * Sin esto, add_param() se ejecuta durante param_seq pero func_actual
 * aun esta vacio (el action final del funcdef no ha corrido aun).
 */
funcdef
    : decorator_seq DEF NAME
      {
          /* mid-rule: declarar funcion apenas vemos DEF NAME */
          int jit = (strstr($1, "triton.jit") != NULL) ? 1 : 0;
          declarar_func($3, @3.first_line, jit);
          if (!jit)
              snprintf(avisos[num_avisos++], MAX_NAME,
                       "Funcion '%s' (linea %d) sin @triton.jit",
                       $3, @3.first_line);
      }
      LPAREN param_seq RPAREN COLON suite
      { func_actual[0] = '\0'; }

    | DEF NAME
      {
          /* mid-rule: funciones sin decorador */
          declarar_func($2, @2.first_line, 0);
          snprintf(avisos[num_avisos++], MAX_NAME,
                   "Funcion '%s' (linea %d) sin @triton.jit",
                   $2, @2.first_line);
      }
      LPAREN param_seq RPAREN COLON suite
      { func_actual[0] = '\0'; }

    | decorator_seq DEF NAME
      {
          /* mid-rule: funciones con tipo de retorno (->) */
          int jit = (strstr($1, "triton.jit") != NULL) ? 1 : 0;
          declarar_func($3, @3.first_line, jit);
      }
      LPAREN param_seq RPAREN ARROW expr COLON suite
      { func_actual[0] = '\0'; }
    ;

/* ---- Decoradores ------------------------------------------------------- */

decorator_seq
    : decorator decorator_seq
      { snprintf($$, sizeof($$), "%s %s", $1, $2); }
    | decorator
      { strncpy($$, $1, sizeof($$) - 1); }
    ;

decorator
    : AT decorator_str NEWLINE
      { strncpy($$, $2, sizeof($$) - 1); }
    ;

/* Sin recursion izquierda: A → NAME A'  y  A' → .NAME A' | (args) A' | ε */
decorator_str
    : NAME decorator_str_tail
      { snprintf($$, sizeof($$), "%s%s", $1, $2); }
    ;

decorator_str_tail
    : DOT NAME decorator_str_tail
      { snprintf($$, sizeof($$), ".%s%s", $2, $3); }
    | LPAREN arg_seq RPAREN decorator_str_tail
      { snprintf($$, sizeof($$), "(...)%s", $4); }
    | /* vacio */
      { $$[0] = '\0'; }
    ;

/* ---- Suite (bloque indentado) ----------------------------------------- */
/*
 * PREGUNTA ORAL: "¿Cómo representas los bloques en la gramática?"
 * RESPUESTA: El lexer emite INDENT al entrar y DEDENT al salir.
 * La regla suite siempre es: NEWLINE INDENT stmt_seq DEDENT
 */
suite
    : NEWLINE INDENT stmt_seq DEDENT
    ;

/* ---- If / For / While / With ------------------------------------------ */

if_stmt
    : IF expr COLON suite else_part
    ;

else_part
    : ELIF expr COLON suite else_part
    | ELSE COLON suite
    | /* vacío */
    ;

for_stmt
    : FOR target_expr IN expr COLON suite
    | FOR target_expr IN expr COLON suite ELSE COLON suite
    ;

while_stmt
    : WHILE expr COLON suite
    | WHILE expr COLON suite ELSE COLON suite
    ;

with_stmt
    : WITH with_item COLON suite
    ;

with_item
    : expr
    | expr AS target_expr
    ;

/* ---- Parámetros de función -------------------------------------------- */

/* nl_opt: newline opcional dentro de paréntesis (implicit line joining) */
nl_opt
    : NEWLINE
    | /* vacío */
    ;

param_seq
    : nl_opt param param_seq_tail
    | /* vacío */
    ;

param_seq_tail
    : COMMA nl_opt param param_seq_tail
    | COMMA nl_opt
    | /* vacío */
    ;

param
    : NAME
      { add_param($1, "", 0); }
    | NAME COLON expr
      {
          int ce = (strstr($3, "constexpr") != NULL) ? 1 : 0;
          add_param($1, $3, ce);
      }
    | NAME EQ expr
      { add_param($1, "", 0); }
    | NAME COLON expr EQ expr
      {
          int ce = (strstr($3, "constexpr") != NULL) ? 1 : 0;
          add_param($1, $3, ce);
      }
    | STAR NAME
    | DOUBLESTAR NAME
    ;

/* ---- Lista de expresiones --------------------------------------------- */

expr_list
    : expr expr_list_tail
      { strncpy($$, $1, sizeof($$) - 1); }
    ;

expr_list_tail
    : COMMA expr expr_list_tail
    | COMMA
    | /* vacío */
    ;

target_expr
    : primary
      { strncpy($$, $1, sizeof($$) - 1); }
    ;

/* ---- Jerarquía de expresiones (sin recursión izquierda) --------------- */

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
    : PLUS  factor %prec UPLUS  { strncpy($$, "uplus",  sizeof($$) - 1); }
    | MINUS factor %prec UMINUS { strncpy($$, "uminus", sizeof($$) - 1); }
    | TILDE factor %prec UTILDE { strncpy($$, "invert", sizeof($$) - 1); }
    | power
      { strncpy($$, $1, sizeof($$) - 1); }
    ;

power
    : primary DOUBLESTAR factor
      { strncpy($$, "pow", sizeof($$) - 1); }
    | primary
      { strncpy($$, $1, sizeof($$) - 1); }
    ;

/* ---- Primary: atom + trailers ----------------------------------------- */
/*
 * primary propaga la cadena completa del acceso (ej. "tl.program_id").
 * Esto permite detectar tl.constexpr en anotaciones de parametros y
 * registrar llamadas a la API de Triton como tl.load, tl.store, etc.
 */
primary
    : atom trailer_seq
      { snprintf($$, sizeof($$), "%s%s", $1, $2); }
    ;

trailer_seq
    : trailer trailer_seq
      { snprintf($$, sizeof($$), "%s%s", $1, $2); }
    | /* vacío */
      { $$[0] = '\0'; }
    ;

trailer
    : DOT NAME
      {
          snprintf($$, sizeof($$), ".%s", $2);
      }
    | LPAREN arg_seq RPAREN
      {
          /*
           * Llamada a funcion: si la expresion base tiene prefijo "tl."
           * registramos el nombre como API call de Triton.
           * El valor $$ se pasa como sufijo al primary.
           */
          strncpy($$, "(...)", sizeof($$) - 1);
      }
    | LBRACKET slice_seq RBRACKET
      { strncpy($$, "[...]", sizeof($$) - 1); }
    ;

/* ---- Argumentos ------------------------------------------------------- */

arg_seq
    : nl_opt arg arg_seq_tail
    | /* vacío */
    ;

arg_seq_tail
    : COMMA nl_opt arg arg_seq_tail
    | COMMA nl_opt
    | /* vacío */
    ;

arg
    : expr
    | NAME EQ expr
    | STAR expr
    | DOUBLESTAR expr
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

/* ---- Átomos ------------------------------------------------------------ */

atom
    : NAME
      {
          strncpy($$, $1, sizeof($$) - 1);
      }
    | NUMBER_INT   { strncpy($$, $1, sizeof($$) - 1); }
    | NUMBER_FLOAT { strncpy($$, $1, sizeof($$) - 1); }
    | NUMBER_HEX   { strncpy($$, $1, sizeof($$) - 1); }
    | NUMBER_BIN   { strncpy($$, $1, sizeof($$) - 1); }
    | NUMBER_OCT   { strncpy($$, $1, sizeof($$) - 1); }
    | STRING       { strncpy($$, $1, sizeof($$) - 1); }
    | TRUE         { strncpy($$, "True",    sizeof($$) - 1); }
    | FALSE        { strncpy($$, "False",   sizeof($$) - 1); }
    | NONE         { strncpy($$, "None",    sizeof($$) - 1); }
    | ELLIPSIS     { strncpy($$, "...",     sizeof($$) - 1); }
    | LPAREN RPAREN             { strncpy($$, "tuple()", sizeof($$) - 1); }
    | LPAREN expr_list RPAREN   { strncpy($$, "paren",   sizeof($$) - 1); }
    | LBRACKET RBRACKET         { strncpy($$, "list()",  sizeof($$) - 1); }
    | LBRACKET expr_list RBRACKET { strncpy($$, "list",  sizeof($$) - 1); }
    | LBRACE RBRACE             { strncpy($$, "dict()",  sizeof($$) - 1); }
    ;

%%

/* =========================================================================
 * SECCIÓN 3 — CÓDIGO DE USUARIO
 * ========================================================================= */

void yyerror(const char *msg)
{
    extern int yylineno;
    snprintf(errores[num_errores++], MAX_NAME,
             "Error sintactico en linea %d: %s", yylineno, msg);
    fprintf(stderr, "ERROR SINTACTICO (linea %d): %s\n", yylineno, msg);
}

int main(int argc, char *argv[])
{
    extern FILE *yyin;
    FILE *f = NULL;

    if (argc > 1) {
        f = fopen(argv[1], "r");
        if (!f) { fprintf(stderr, "No se pudo abrir '%s'\n", argv[1]); return 1; }
        yyin = f;
    }

    int r = yyparse();
    if (f) fclose(f);
    return r;
}
