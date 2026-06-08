/*
 * triton.l
 * ========
 * Scanner (analizador léxico) para el lenguaje Triton GPU kernel.
 * Implementado con UNIX lex (flex). Genera código C mediante:
 *
 *   flex triton.l        -> produce lex.yy.c
 *   gcc lex.yy.c -lfl -o triton_scanner
 *
 * SALIDAS:
 *   1. Secuencia de tokens: <TIPO, lexema, línea>
 *   2. Tabla de símbolos: identificadores y literales con su ID único
 *
 * RESTRICCIONES CUMPLIDAS:
 *   - Implementado con UNIX lex, sin librerías de RE externas.
 *   - No usa Python ni ast de Python.
 *   - No usa APIs de expresiones regulares de ningún lenguaje.
 *
 * ALFABETO DEL LENGUAJE TRITON:
 *   Letras:      A-Z  a-z  _
 *   Dígitos:     0-9
 *   Operadores:  + - * / % ** // << >> & | ^ ~ < > = ! @
 *   Delimitadores: ( ) [ ] { } : , . ;
 *   Espacio/Tab: ignorado fuera de inicio de línea
 *   Newline:     \n  delimitador lógico de sentencias
 */

/* =========================================================================
 * SECCIÓN 1 — DEFINICIONES
 * Aquí se declaran: opciones de flex, código C de cabecera,
 * macros de patrones reutilizables, y condiciones de inicio.
 * ========================================================================= */

%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------
 * TABLA DE SÍMBOLOS
 * Almacena identificadores y literales encontrados en el fuente.
 * Capacidad estática de MAX_SYMBOLS entradas; en producción se
 * ampliaría con una lista dinámica o tabla hash.
 * ------------------------------------------------------------------ */
#define MAX_SYMBOLS 4096

typedef struct {
    char  lexema[256];   /* texto del token                  */
    char  tipo[32];      /* "ID", "NUM_INT", "NUM_FLOAT", "STR" */
    int   id;            /* ID único dentro de su categoría  */
    int   linea;         /* primera aparición en el fuente   */
} Simbolo;

static Simbolo tabla[MAX_SYMBOLS];
static int     num_simbolos = 0;

/*
 * buscar_o_insertar_simbolo
 * -------------------------
 * Busca el lexema en la tabla. Si ya existe retorna su ID.
 * Si no existe, lo inserta y retorna el nuevo ID.
 * Esta búsqueda lineal es correcta para fines didácticos;
 * una tabla hash sería O(1) en producción.
 */
static int buscar_o_insertar_simbolo(const char *lex, const char *tipo, int linea)
{
    int i;
    for (i = 0; i < num_simbolos; i++) {
        if (strcmp(tabla[i].lexema, lex) == 0 &&
            strcmp(tabla[i].tipo,   tipo) == 0)
            return tabla[i].id;
    }
    if (num_simbolos >= MAX_SYMBOLS) {
        fprintf(stderr, "ERROR: tabla de símbolos llena\n");
        exit(1);
    }
    strncpy(tabla[num_simbolos].lexema, lex,  sizeof(tabla[0].lexema) - 1);
    strncpy(tabla[num_simbolos].tipo,   tipo, sizeof(tabla[0].tipo)   - 1);
    tabla[num_simbolos].id    = num_simbolos + 1;
    tabla[num_simbolos].linea = linea;
    return tabla[num_simbolos++].id;
}

/*
 * imprimir_token
 * --------------
 * Imprime una línea de la secuencia de tokens en el formato
 * estándar visto en clase:  < TIPO , lexema , línea >
 */
static void imprimir_token(const char *tipo, const char *lexema, int linea)
{
    printf("< %-18s , %-30s , %d >\n", tipo, lexema, linea);
}

/*
 * imprimir_tabla_simbolos
 * -----------------------
 * Llamada al final del escaneo; imprime toda la tabla de símbolos.
 */
static void imprimir_tabla_simbolos(void)
{
    int i;
    printf("\n");
    printf("==============================================================\n");
    printf("  TABLA DE SÍMBOLOS\n");
    printf("==============================================================\n");
    printf("  %-6s %-12s %-30s %s\n", "ID", "TIPO", "LEXEMA", "LÍNEA");
    printf("  %-6s %-12s %-30s %s\n", "------", "------------",
           "------------------------------", "-----");
    for (i = 0; i < num_simbolos; i++) {
        printf("  %-6d %-12s %-30s %d\n",
               tabla[i].id, tabla[i].tipo,
               tabla[i].lexema, tabla[i].linea);
    }
    printf("==============================================================\n");
}

%}

/* Opciones de flex:
 *   noyywrap  — no necesitamos yywrap() para entrada de un solo archivo.
 *   yylineno  — flex mantiene yylineno automáticamente.
 */
%option noyywrap
%option yylineno

/* ------------------------------------------------------------------
 * MACROS DE PATRONES
 * Definen fragmentos reutilizables de expresiones regulares.
 * El nombre en mayúsculas es convención; flex los expande con {NOMBRE}.
 * ------------------------------------------------------------------ */

/* Letras y dígitos básicos */
LETRA       [A-Za-z_]
DIGITO      [0-9]
ALFANUM     [A-Za-z0-9_]

/* Identificador: empieza con letra o _, seguido de alfanuméricos */
IDENT       {LETRA}{ALFANUM}*

/* Números enteros en distintas bases */
HEX         0[xX][0-9a-fA-F]+
BIN         0[bB][01]+
OCT         0[oO][0-7]+
DEC         {DIGITO}+

/* Números flotantes (varias formas) */
EXP         [eE][+-]?{DIGITO}+
FLOAT1      {DIGITO}+\.{DIGITO}*{EXP}?
FLOAT2      \.{DIGITO}+{EXP}?
FLOAT3      {DIGITO}+{EXP}
FLOAT       {FLOAT1}|{FLOAT2}|{FLOAT3}

/* Espacios y tabulaciones (en medio de línea se ignoran) */
ESPACIO     [ \t]+

/* =========================================================================
 * SECCIÓN 2 — REGLAS
 * Cada regla: patrón  { acción en C }
 * Las reglas más largas y más específicas van PRIMERO para tomar prioridad
 * (flex aplica la regla que produce el lexema más largo; en caso de empate,
 *  la primera regla en el archivo).
 * ========================================================================= */
%%

    /* ----------------------------------------------------------------
     * COMENTARIOS
     * Se descartan completamente; no generan token ni entrada en tabla.
     * El patrón #[^\n]* consume desde # hasta el fin de línea.
     * ---------------------------------------------------------------- */
#[^\n]*     { /* comentario — ignorar */ }


    /* ----------------------------------------------------------------
     * STRINGS
     * Triple-quoted ANTES que single-quoted para que flex tome el
     * lexema más largo.  Los prefijos opcionales (f, r, b, u) van
     * antes de las comillas.
     * ---------------------------------------------------------------- */

[fFrRbBuU]*\"\"\"([^"\\]|\\.|\"([^"\\]|\\.)|\"\"([^"\\]|\\.))*\"\"\" {
    int sid = buscar_o_insertar_simbolo(yytext, "STR", yylineno);
    imprimir_token("STRING", yytext, yylineno);
}

[fFrRbBuU]*\'\'\'([^'\\]|\\.|\'([^'\\]|\\.)|\'\'([^'\\]|\\.))*\'\'\' {
    int sid = buscar_o_insertar_simbolo(yytext, "STR", yylineno);
    imprimir_token("STRING", yytext, yylineno);
}

[fFrRbBuU]*\"([^\"\\\n]|\\.)*\" {
    int sid = buscar_o_insertar_simbolo(yytext, "STR", yylineno);
    imprimir_token("STRING", yytext, yylineno);
}

[fFrRbBuU]*\'([^\'\\\n]|\\.)*\' {
    int sid = buscar_o_insertar_simbolo(yytext, "STR", yylineno);
    imprimir_token("STRING", yytext, yylineno);
}


    /* ----------------------------------------------------------------
     * NÚMEROS — float ANTES que entero para evitar que "3.14"
     * se tokenice como entero "3" + punto + entero "14".
     * ---------------------------------------------------------------- */

{FLOAT}     {
    int sid = buscar_o_insertar_simbolo(yytext, "NUM_FLOAT", yylineno);
    imprimir_token("NUMBER_FLOAT", yytext, yylineno);
}

{HEX}       {
    int sid = buscar_o_insertar_simbolo(yytext, "NUM_INT", yylineno);
    imprimir_token("NUMBER_HEX", yytext, yylineno);
}

{BIN}       {
    int sid = buscar_o_insertar_simbolo(yytext, "NUM_INT", yylineno);
    imprimir_token("NUMBER_BIN", yytext, yylineno);
}

{OCT}       {
    int sid = buscar_o_insertar_simbolo(yytext, "NUM_INT", yylineno);
    imprimir_token("NUMBER_OCT", yytext, yylineno);
}

{DEC}       {
    int sid = buscar_o_insertar_simbolo(yytext, "NUM_INT", yylineno);
    imprimir_token("NUMBER_INT", yytext, yylineno);
}


    /* ----------------------------------------------------------------
     * IDENTIFICADORES Y PALABRAS RESERVADAS
     * La regla IDENT captura cualquier nombre; luego el bloque if/else
     * clasifica las palabras reservadas antes de tratarlo como NAME.
     * Esto replica la función t_NAME de PLY.
     * ---------------------------------------------------------------- */

{IDENT}     {
    /* Palabras reservadas de Python/Triton */
    if      (strcmp(yytext, "and")      == 0) imprimir_token("AND",      yytext, yylineno);
    else if (strcmp(yytext, "as")       == 0) imprimir_token("AS",       yytext, yylineno);
    else if (strcmp(yytext, "assert")   == 0) imprimir_token("ASSERT",   yytext, yylineno);
    else if (strcmp(yytext, "break")    == 0) imprimir_token("BREAK",    yytext, yylineno);
    else if (strcmp(yytext, "class")    == 0) imprimir_token("CLASS",    yytext, yylineno);
    else if (strcmp(yytext, "continue") == 0) imprimir_token("CONTINUE", yytext, yylineno);
    else if (strcmp(yytext, "def")      == 0) imprimir_token("DEF",      yytext, yylineno);
    else if (strcmp(yytext, "del")      == 0) imprimir_token("DEL",      yytext, yylineno);
    else if (strcmp(yytext, "elif")     == 0) imprimir_token("ELIF",     yytext, yylineno);
    else if (strcmp(yytext, "else")     == 0) imprimir_token("ELSE",     yytext, yylineno);
    else if (strcmp(yytext, "except")   == 0) imprimir_token("EXCEPT",   yytext, yylineno);
    else if (strcmp(yytext, "False")    == 0) imprimir_token("FALSE",    yytext, yylineno);
    else if (strcmp(yytext, "finally")  == 0) imprimir_token("FINALLY",  yytext, yylineno);
    else if (strcmp(yytext, "for")      == 0) imprimir_token("FOR",      yytext, yylineno);
    else if (strcmp(yytext, "from")     == 0) imprimir_token("FROM",     yytext, yylineno);
    else if (strcmp(yytext, "global")   == 0) imprimir_token("GLOBAL",   yytext, yylineno);
    else if (strcmp(yytext, "if")       == 0) imprimir_token("IF",       yytext, yylineno);
    else if (strcmp(yytext, "import")   == 0) imprimir_token("IMPORT",   yytext, yylineno);
    else if (strcmp(yytext, "in")       == 0) imprimir_token("IN",       yytext, yylineno);
    else if (strcmp(yytext, "is")       == 0) imprimir_token("IS",       yytext, yylineno);
    else if (strcmp(yytext, "lambda")   == 0) imprimir_token("LAMBDA",   yytext, yylineno);
    else if (strcmp(yytext, "None")     == 0) imprimir_token("NONE",     yytext, yylineno);
    else if (strcmp(yytext, "nonlocal") == 0) imprimir_token("NONLOCAL", yytext, yylineno);
    else if (strcmp(yytext, "not")      == 0) imprimir_token("NOT",      yytext, yylineno);
    else if (strcmp(yytext, "or")       == 0) imprimir_token("OR",       yytext, yylineno);
    else if (strcmp(yytext, "pass")     == 0) imprimir_token("PASS",     yytext, yylineno);
    else if (strcmp(yytext, "raise")    == 0) imprimir_token("RAISE",    yytext, yylineno);
    else if (strcmp(yytext, "return")   == 0) imprimir_token("RETURN",   yytext, yylineno);
    else if (strcmp(yytext, "True")     == 0) imprimir_token("TRUE",     yytext, yylineno);
    else if (strcmp(yytext, "try")      == 0) imprimir_token("TRY",      yytext, yylineno);
    else if (strcmp(yytext, "while")    == 0) imprimir_token("WHILE",    yytext, yylineno);
    else if (strcmp(yytext, "with")     == 0) imprimir_token("WITH",     yytext, yylineno);
    else if (strcmp(yytext, "yield")    == 0) imprimir_token("YIELD",    yytext, yylineno);
    else {
        /* Identificador de usuario — va a la tabla de símbolos */
        int sid = buscar_o_insertar_simbolo(yytext, "ID", yylineno);
        imprimir_token("NAME", yytext, yylineno);
    }
}


    /* ----------------------------------------------------------------
     * OPERADORES DE ASIGNACIÓN COMPUESTA (4 chars)
     * Deben ir ANTES que los de 2 chars para que flex tome el más largo.
     * ---------------------------------------------------------------- */
"**="    { imprimir_token("DOUBLESTAREQ",  yytext, yylineno); }
"//="    { imprimir_token("DOUBLESLASHEQ", yytext, yylineno); }
"<<="    { imprimir_token("LSHIFTEQ",      yytext, yylineno); }
">>="    { imprimir_token("RSHIFTEQ",      yytext, yylineno); }

    /* ----------------------------------------------------------------
     * OPERADORES DE ASIGNACIÓN COMPUESTA (2+1 chars)
     * ---------------------------------------------------------------- */
"+="     { imprimir_token("PLUSEQ",    yytext, yylineno); }
"-="     { imprimir_token("MINUSEQ",   yytext, yylineno); }
"*="     { imprimir_token("STAREQ",    yytext, yylineno); }
"/="     { imprimir_token("SLASHEQ",   yytext, yylineno); }
"%="     { imprimir_token("PERCENTEQ", yytext, yylineno); }
"&="     { imprimir_token("AMPEQ",     yytext, yylineno); }
"|="     { imprimir_token("PIPEEQ",    yytext, yylineno); }
"^="     { imprimir_token("CARETEQ",   yytext, yylineno); }

    /* ----------------------------------------------------------------
     * OPERADORES DE COMPARACIÓN (2 chars)
     * ---------------------------------------------------------------- */
"=="     { imprimir_token("EQEQ",  yytext, yylineno); }
"!="     { imprimir_token("NOTEQ", yytext, yylineno); }
"<="     { imprimir_token("LTEQ",  yytext, yylineno); }
">="     { imprimir_token("GTEQ",  yytext, yylineno); }

    /* ----------------------------------------------------------------
     * OPERADORES ARITMÉTICOS DOBLES (2 chars)
     * ---------------------------------------------------------------- */
"**"     { imprimir_token("DOUBLESTAR",  yytext, yylineno); }
"//"     { imprimir_token("DOUBLESLASH", yytext, yylineno); }
"<<"     { imprimir_token("LSHIFT",      yytext, yylineno); }
">>"     { imprimir_token("RSHIFT",      yytext, yylineno); }

    /* ----------------------------------------------------------------
     * OTROS OPERADORES MULTI-CHAR
     * ---------------------------------------------------------------- */
"->"     { imprimir_token("ARROW",    yytext, yylineno); }
":="     { imprimir_token("WALRUS",   yytext, yylineno); }
"..."    { imprimir_token("ELLIPSIS", yytext, yylineno); }

    /* ----------------------------------------------------------------
     * OPERADORES SIMPLES (1 char)
     * ---------------------------------------------------------------- */
"+"      { imprimir_token("PLUS",    yytext, yylineno); }
"-"      { imprimir_token("MINUS",   yytext, yylineno); }
"*"      { imprimir_token("STAR",    yytext, yylineno); }
"/"      { imprimir_token("SLASH",   yytext, yylineno); }
"%"      { imprimir_token("PERCENT", yytext, yylineno); }
"&"      { imprimir_token("AMP",     yytext, yylineno); }
"|"      { imprimir_token("PIPE",    yytext, yylineno); }
"^"      { imprimir_token("CARET",   yytext, yylineno); }
"~"      { imprimir_token("TILDE",   yytext, yylineno); }
"<"      { imprimir_token("LT",      yytext, yylineno); }
">"      { imprimir_token("GT",      yytext, yylineno); }
"="      { imprimir_token("EQ",      yytext, yylineno); }
"@"      { imprimir_token("AT",      yytext, yylineno); }

    /* ----------------------------------------------------------------
     * DELIMITADORES (1 char)
     * ---------------------------------------------------------------- */
"("      { imprimir_token("LPAREN",   yytext, yylineno); }
")"      { imprimir_token("RPAREN",   yytext, yylineno); }
"["      { imprimir_token("LBRACKET", yytext, yylineno); }
"]"      { imprimir_token("RBRACKET", yytext, yylineno); }
"{"      { imprimir_token("LBRACE",   yytext, yylineno); }
"}"      { imprimir_token("RBRACE",   yytext, yylineno); }
":"      { imprimir_token("COLON",    yytext, yylineno); }
","      { imprimir_token("COMMA",    yytext, yylineno); }
"."      { imprimir_token("DOT",      yytext, yylineno); }
";"      { imprimir_token("SEMI",     yytext, yylineno); }

    /* ----------------------------------------------------------------
     * NEWLINE — delimitador lógico de sentencias
     * flex incrementa yylineno automáticamente con %option yylineno,
     * por lo que aquí solo necesitamos emitir el token.
     * ---------------------------------------------------------------- */
\n       { imprimir_token("NEWLINE", "\\n", yylineno); }

    /* ----------------------------------------------------------------
     * CONTINUACIÓN DE LÍNEA CON BACKSLASH — se ignora
     * ---------------------------------------------------------------- */
\\\n     { /* backslash + newline: continuación lógica — ignorar */ }

    /* ----------------------------------------------------------------
     * ESPACIOS Y TABULACIONES EN MEDIO DE LÍNEA — se ignoran
     * (la indentación significativa es manejada por el parser)
     * ---------------------------------------------------------------- */
{ESPACIO}  { /* espacios en medio de línea — ignorar */ }

    /* ----------------------------------------------------------------
     * CARACTERES INVÁLIDOS — manejo de errores léxicos
     * Cualquier carácter que no coincida con ninguna regla anterior
     * es un error. Se imprime el mensaje y se continúa el escaneo.
     * ---------------------------------------------------------------- */
.        {
    fprintf(stderr,
            "ERROR LÉXICO (línea %d): carácter ilegal '%s'\n",
            yylineno, yytext);
}

%%

/* =========================================================================
 * SECCIÓN 3 — CÓDIGO DE USUARIO
 * Contiene main() y cualquier función auxiliar adicional.
 * Esta sección se copia literalmente al final de lex.yy.c.
 * ========================================================================= */

/*
 * main
 * ----
 * Punto de entrada del scanner.
 * Uso:  ./triton_scanner < archivo.py
 *   o:  ./triton_scanner archivo.py
 *
 * Si se provee un argumento, se abre como archivo de entrada.
 * Si no, se lee de stdin.
 *
 * Proceso:
 *   1. Configurar entrada.
 *   2. Llamar yylex() que consume el fuente token a token,
 *      imprimiendo cada token al encontrarlo (acción en la Sección 2).
 *   3. Al terminar yylex(), imprimir la tabla de símbolos completa.
 */
int main(int argc, char *argv[])
{
    FILE *entrada = NULL;

    if (argc > 1) {
        entrada = fopen(argv[1], "r");
        if (!entrada) {
            fprintf(stderr, "ERROR: no se pudo abrir '%s'\n", argv[1]);
            return 1;
        }
        yyin = entrada;
    }
    /* Si argc == 1, yyin ya apunta a stdin por defecto */

    printf("==============================================================\n");
    printf("  SECUENCIA DE TOKENS — Triton Scanner\n");
    printf("==============================================================\n");
    printf("  %-18s   %-30s   %s\n", "TIPO", "LEXEMA", "LÍNEA");
    printf("  %-18s   %-30s   %s\n",
           "------------------", "------------------------------", "-----");

    /* yylex() itera sobre todo el fuente, ejecutando las acciones
     * de la Sección 2 para cada token encontrado. Retorna 0 al EOF. */
    yylex();

    /* Imprimir tabla de símbolos al finalizar el escaneo */
    imprimir_tabla_simbolos();

    if (entrada)
        fclose(entrada);

    return 0;
}
