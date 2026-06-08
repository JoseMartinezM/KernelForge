/*
 * triton.l — Scanner Triton GPU kernel
 * =====================================
 * UNIX flex (genera C). Dos modos de compilacion:
 *
 *   Standalone:  flex triton.l && gcc lex.yy.c -lfl -o triton_scanner
 *   Con parser:  bison -d triton.y && flex triton.l
 *                gcc -DWITH_PARSER lex.yy.c triton.tab.c -lfl -o triton_compiler
 *
 * INDENTACION:
 *   Implementa el algoritmo oficial de Python (ref. §2.1.8 Language Reference).
 *   El patron \n[ \t]* captura el salto de linea mas los espacios del inicio
 *   de la siguiente linea, determinando el nuevo nivel de indentacion.
 *
 *   Algoritmo:
 *     pila_indent = [0]
 *     Al ver \n + espacios:
 *       nuevo = contar_espacios()
 *       si nuevo > pila.top  → emitir NEWLINE, luego INDENT, push nuevo
 *       si nuevo < pila.top  → emitir NEWLINE, luego DEDENT(s) hasta igualar
 *       si nuevo == pila.top → solo emitir NEWLINE
 *     NEWLINE dentro de ( ) [ ] { } → ignorado (implicit line joining)
 *     Al EOF → NEWLINE + DEDENT por cada nivel abierto
 *
 *   Como una regla de flex solo puede retornar UN token, usamos una cola
 *   circular para encolar INDENT/DEDENT y drenarla en llamadas siguientes.
 */

/* =========================================================================
 * SECCION 1 — DEFINICIONES
 * ========================================================================= */

%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------
 * Modo parser: incluir cabecera de bison para YYSTYPE y token IDs
 * ------------------------------------------------------------------ */
#ifdef WITH_PARSER
  #include "triton.tab.h"
#else
/* ------------------------------------------------------------------
 * Modo standalone: definir IDs de token localmente
 * ------------------------------------------------------------------ */
  enum standalone_tokens {
    NEWLINE=257, INDENT, DEDENT,
    NAME, STRING,
    NUMBER_INT, NUMBER_FLOAT, NUMBER_HEX, NUMBER_BIN, NUMBER_OCT,
    AND, AS, ASSERT, BREAK, CLASS, CONTINUE, DEF, DEL, ELIF, ELSE,
    EXCEPT, FALSE, FINALLY, FOR, FROM, GLOBAL, IF, IMPORT, IN, IS,
    LAMBDA, NONE, NONLOCAL, NOT, OR, PASS, RAISE, RETURN, TRUE, TRY,
    WHILE, WITH, YIELD,
    DOUBLESTAREQ, DOUBLESLASHEQ, LSHIFTEQ, RSHIFTEQ,
    PLUSEQ, MINUSEQ, STAREQ, SLASHEQ, PERCENTEQ, AMPEQ, PIPEEQ, CARETEQ,
    EQEQ, NOTEQ, LTEQ, GTEQ,
    DOUBLESTAR, DOUBLESLASH, LSHIFT, RSHIFT,
    ARROW, WALRUS, ELLIPSIS,
    PLUS, MINUS, STAR, SLASH, PERCENT, AMP, PIPE, CARET, TILDE,
    LT, GT, EQ, AT,
    LPAREN, RPAREN, LBRACKET, RBRACKET, LBRACE, RBRACE,
    COLON, COMMA, DOT, SEMI
  };
#endif

/* ------------------------------------------------------------------
 * TABLA DE SIMBOLOS
 * ------------------------------------------------------------------ */
#define MAX_SYMBOLS 4096
typedef struct { char lexema[256]; char tipo[32]; int id; int linea; } Simbolo;
static Simbolo tabla[MAX_SYMBOLS];
static int     num_simbolos = 0;

static int buscar_o_insertar_simbolo(const char *lex, const char *tipo, int linea)
{
    int i;
    for (i = 0; i < num_simbolos; i++)
        if (strcmp(tabla[i].lexema, lex) == 0 && strcmp(tabla[i].tipo, tipo) == 0)
            return tabla[i].id;
    if (num_simbolos >= MAX_SYMBOLS) return -1;
    strncpy(tabla[num_simbolos].lexema, lex,  255);
    strncpy(tabla[num_simbolos].tipo,   tipo, 31);
    tabla[num_simbolos].id    = num_simbolos + 1;
    tabla[num_simbolos].linea = linea;
    return tabla[num_simbolos++].id;
}

static void imprimir_token(const char *tipo, const char *lexema, int linea)
{
    printf("< %-18s , %-30s , %d >\n", tipo, lexema, linea);
}

static void imprimir_tabla_simbolos(void)
{
    int i;
    printf("\n============================================================\n");
    printf("  TABLA DE SIMBOLOS\n");
    printf("============================================================\n");
    printf("  %-6s %-12s %-30s %s\n", "ID", "TIPO", "LEXEMA", "LINEA");
    printf("  %-6s %-12s %-30s %s\n", "------","------------",
           "------------------------------","-----");
    for (i = 0; i < num_simbolos; i++)
        printf("  %-6d %-12s %-30s %d\n",
               tabla[i].id, tabla[i].tipo, tabla[i].lexema, tabla[i].linea);
    printf("============================================================\n");
}

/* ------------------------------------------------------------------
 * PILA DE INDENTACION
 * ------------------------------------------------------------------ */
#define MAX_INDENT 64
static int indent_stack[MAX_INDENT] = {0};
static int indent_top = 0;

/* ------------------------------------------------------------------
 * COLA DE TOKENS PENDIENTES (circular)
 * Permite encolar INDENT/DEDENT y devolverlos uno a uno.
 * ------------------------------------------------------------------ */
#define QUEUE_CAP 128
static int  tok_queue[QUEUE_CAP];
static char tok_lex  [QUEUE_CAP][256];
static int  q_head = 0, q_tail = 0;

static int  queue_empty(void)  { return q_head == q_tail; }
static void queue_push(int t, const char *lex) {
    tok_queue[q_tail % QUEUE_CAP] = t;
    strncpy(tok_lex[q_tail % QUEUE_CAP], lex, 255);
    q_tail++;
}

/* Saca un token de la cola, actualiza yylval en modo parser,
 * imprime en modo standalone, y lo retorna. */
static int queue_pop_tok(void) {
    int tok         = tok_queue[q_head % QUEUE_CAP];
    const char *lex = tok_lex  [q_head % QUEUE_CAP];
    q_head++;
#ifdef WITH_PARSER
    strncpy(yylval.sval, lex, sizeof(yylval.sval) - 1);
#else
    if      (tok == NEWLINE) imprimir_token("NEWLINE", "\\n", yylineno);
    else if (tok == INDENT)  imprimir_token("INDENT",  "",    yylineno);
    else if (tok == DEDENT)  imprimir_token("DEDENT",  "",    yylineno);
#endif
    return tok;
}

/* ------------------------------------------------------------------
 * CONTADOR DE PARENTESIS
 * Dentro de ( ) [ ] { } los NEWLINE se suprimen.
 * ------------------------------------------------------------------ */
static int paren_depth = 0;

/* ------------------------------------------------------------------
 * MACROS DE EMISION DE TOKENS
 * Ambos modos retornan el token ID para que el loop de tokens funcione.
 * ------------------------------------------------------------------ */
#ifdef WITH_PARSER
  #define TOK(id, tipo) \
      do { strncpy(yylval.sval, yytext, sizeof(yylval.sval)-1); \
           buscar_o_insertar_simbolo(yytext, tipo, yylineno); \
           return (id); } while(0)
  #define TOK_OP(id) return (id)
#else
  #define TOK(id, tipo) \
      do { buscar_o_insertar_simbolo(yytext, tipo, yylineno); \
           imprimir_token(#id, yytext, yylineno); \
           return (id); } while(0)
  #define TOK_OP(id) \
      do { imprimir_token(#id, yytext, yylineno); \
           return (id); } while(0)
#endif

%}

%option noyywrap
%option yylineno

/* ---- Macros de patrones ---- */
LETRA    [A-Za-z_]
ALFANUM  [A-Za-z0-9_]
IDENT    {LETRA}{ALFANUM}*
HEX      0[xX][0-9a-fA-F]+
BIN      0[bB][01]+
OCT      0[oO][0-7]+
DEC      [0-9]+
EXP      [eE][+-]?[0-9]+
FLOAT    ({DEC}\.{DEC}?|\.{DEC}){EXP}?|{DEC}{EXP}
ESP      [ \t]+

/* =========================================================================
 * SECCION 2 — REGLAS
 * ========================================================================= */
%%

    /* ---- Comentarios ---- */
#[^\n]*  { /* ignorar */ }

    /* ---- Strings triple-quote (patron mas largo va primero) ---- */
[fFrRbBuU]*\"\"\"([^"\\]|\\.|\"[^"\\]|\"\"[^"\\])*\"\"\"  { TOK(STRING,"STR"); }
[fFrRbBuU]*\'\'\'([^'\\]|\\.|\'[^'\\]|\'\'[^'\\])*\'\'\'  { TOK(STRING,"STR"); }
[fFrRbBuU]*\"([^\"\\\n]|\\.)*\"  { TOK(STRING,"STR"); }
[fFrRbBuU]*\'([^\'\\\n]|\\.)*\'  { TOK(STRING,"STR"); }

    /* ---- Numeros ---- */
{FLOAT}  { TOK(NUMBER_FLOAT,"NUM_FLOAT"); }
{HEX}    { TOK(NUMBER_HEX,  "NUM_INT");   }
{BIN}    { TOK(NUMBER_BIN,  "NUM_INT");   }
{OCT}    { TOK(NUMBER_OCT,  "NUM_INT");   }
{DEC}    { TOK(NUMBER_INT,  "NUM_INT");   }

    /* ---- Identificadores y palabras reservadas ---- */
{IDENT}  {
    if      (!strcmp(yytext,"and"))      TOK_OP(AND);
    else if (!strcmp(yytext,"as"))       TOK_OP(AS);
    else if (!strcmp(yytext,"assert"))   TOK_OP(ASSERT);
    else if (!strcmp(yytext,"break"))    TOK_OP(BREAK);
    else if (!strcmp(yytext,"class"))    TOK_OP(CLASS);
    else if (!strcmp(yytext,"continue")) TOK_OP(CONTINUE);
    else if (!strcmp(yytext,"def"))      TOK_OP(DEF);
    else if (!strcmp(yytext,"del"))      TOK_OP(DEL);
    else if (!strcmp(yytext,"elif"))     TOK_OP(ELIF);
    else if (!strcmp(yytext,"else"))     TOK_OP(ELSE);
    else if (!strcmp(yytext,"except"))   TOK_OP(EXCEPT);
    else if (!strcmp(yytext,"False"))    TOK_OP(FALSE);
    else if (!strcmp(yytext,"finally"))  TOK_OP(FINALLY);
    else if (!strcmp(yytext,"for"))      TOK_OP(FOR);
    else if (!strcmp(yytext,"from"))     TOK_OP(FROM);
    else if (!strcmp(yytext,"global"))   TOK_OP(GLOBAL);
    else if (!strcmp(yytext,"if"))       TOK_OP(IF);
    else if (!strcmp(yytext,"import"))   TOK_OP(IMPORT);
    else if (!strcmp(yytext,"in"))       TOK_OP(IN);
    else if (!strcmp(yytext,"is"))       TOK_OP(IS);
    else if (!strcmp(yytext,"lambda"))   TOK_OP(LAMBDA);
    else if (!strcmp(yytext,"None"))     TOK_OP(NONE);
    else if (!strcmp(yytext,"nonlocal")) TOK_OP(NONLOCAL);
    else if (!strcmp(yytext,"not"))      TOK_OP(NOT);
    else if (!strcmp(yytext,"or"))       TOK_OP(OR);
    else if (!strcmp(yytext,"pass"))     TOK_OP(PASS);
    else if (!strcmp(yytext,"raise"))    TOK_OP(RAISE);
    else if (!strcmp(yytext,"return"))   TOK_OP(RETURN);
    else if (!strcmp(yytext,"True"))     TOK_OP(TRUE);
    else if (!strcmp(yytext,"try"))      TOK_OP(TRY);
    else if (!strcmp(yytext,"while"))    TOK_OP(WHILE);
    else if (!strcmp(yytext,"with"))     TOK_OP(WITH);
    else if (!strcmp(yytext,"yield"))    TOK_OP(YIELD);
    else { TOK(NAME, "ID"); }
}

    /* ---- Operadores de asignacion compuesta (mas largos primero) ---- */
"**="  { TOK_OP(DOUBLESTAREQ);  }
"//="  { TOK_OP(DOUBLESLASHEQ); }
"<<="  { TOK_OP(LSHIFTEQ);      }
">>="  { TOK_OP(RSHIFTEQ);      }
"+="   { TOK_OP(PLUSEQ);    }
"-="   { TOK_OP(MINUSEQ);   }
"*="   { TOK_OP(STAREQ);    }
"/="   { TOK_OP(SLASHEQ);   }
"%="   { TOK_OP(PERCENTEQ); }
"&="   { TOK_OP(AMPEQ);     }
"|="   { TOK_OP(PIPEEQ);    }
"^="   { TOK_OP(CARETEQ);   }
    /* ---- Operadores de comparacion ---- */
"=="   { TOK_OP(EQEQ);  }
"!="   { TOK_OP(NOTEQ); }
"<="   { TOK_OP(LTEQ);  }
">="   { TOK_OP(GTEQ);  }
    /* ---- Operadores dobles ---- */
"**"   { TOK_OP(DOUBLESTAR);  }
"//"   { TOK_OP(DOUBLESLASH); }
"<<"   { TOK_OP(LSHIFT);      }
">>"   { TOK_OP(RSHIFT);      }
"->"   { TOK_OP(ARROW);  }
":="   { TOK_OP(WALRUS); }
"..."  { TOK_OP(ELLIPSIS); }
    /* ---- Operadores simples ---- */
"+"    { TOK_OP(PLUS);    }
"-"    { TOK_OP(MINUS);   }
"*"    { TOK_OP(STAR);    }
"/"    { TOK_OP(SLASH);   }
"%"    { TOK_OP(PERCENT); }
"&"    { TOK_OP(AMP);     }
"|"    { TOK_OP(PIPE);    }
"^"    { TOK_OP(CARET);   }
"~"    { TOK_OP(TILDE);   }
"<"    { TOK_OP(LT);      }
">"    { TOK_OP(GT);      }
"="    { TOK_OP(EQ);      }
"@"    { TOK_OP(AT);      }

    /* ---- Delimitadores (actualizan profundidad de parentesis) ---- */
"("  { paren_depth++; TOK_OP(LPAREN);   }
"["  { paren_depth++; TOK_OP(LBRACKET); }
"{"  { paren_depth++; TOK_OP(LBRACE);   }
")"  { if(paren_depth>0)paren_depth--; TOK_OP(RPAREN);   }
"]"  { if(paren_depth>0)paren_depth--; TOK_OP(RBRACKET); }
"}"  { if(paren_depth>0)paren_depth--; TOK_OP(RBRACE);   }
":"  { TOK_OP(COLON); }
","  { TOK_OP(COMMA); }
"."  { TOK_OP(DOT);   }
";"  { TOK_OP(SEMI);  }

    /* ----------------------------------------------------------------
     * NEWLINE + INDENTACION
     *
     * Dos reglas, orden importa (flex usa la mas larga primero):
     *
     * 1) Linea en blanco: \n[ \t]* seguido de otro \n (o #).
     *    En Python las lineas vacias dentro de un bloque NO cambian
     *    el nivel de indentacion.  Solo emitimos NEWLINE.
     *
     * 2) Linea real: \n[ \t]* seguido de contenido.
     *    Calculamos el nuevo nivel y emitimos INDENT/DEDENT segun
     *    el algoritmo de Python §2.1.8.
     * ---------------------------------------------------------------- */

    /* Linea en blanco (solo espacios entre dos newlines o antes de EOF) */
\n[ \t]*/[\n#] {
    if (paren_depth == 0) {
        queue_push(NEWLINE, "\\n");
        return queue_pop_tok();
    }
}

    /* Linea con contenido real */
\n[ \t]* {
    if (paren_depth > 0) {
        /* Implicit line joining: newline invisible al parser */
    } else {
        int i, nivel = 0;
        for (i = 1; i < yyleng; i++) {
            if      (yytext[i] == ' ')  nivel++;
            else if (yytext[i] == '\t') nivel = ((nivel / 4) + 1) * 4;
        }

        queue_push(NEWLINE, "\\n");

        if (nivel > indent_stack[indent_top]) {
            if (indent_top + 1 < MAX_INDENT) {
                indent_stack[++indent_top] = nivel;
                queue_push(INDENT, "");
            }
        } else {
            while (indent_top > 0 && indent_stack[indent_top] > nivel) {
                indent_top--;
                queue_push(DEDENT, "");
            }
        }

        return queue_pop_tok();
    }
}

    /* ---- Continuacion de linea explicita (yylineno auto) ---- */
\\\n   { /* yylineno incrementado por flex */ }

    /* ---- Espacios en medio de linea ---- */
{ESP}  { /* ignorar */ }

    /* ----------------------------------------------------------------
     * EOF: emitir NEWLINE final + DEDENT por cada bloque abierto.
     * La regla <<EOF>> se llama repetidamente hasta retornar 0.
     * ---------------------------------------------------------------- */
<<EOF>> {
    if (!queue_empty()) return queue_pop_tok();
    if (indent_top > 0) {
        queue_push(NEWLINE, "\\n");
        while (indent_top > 0) { indent_top--; queue_push(DEDENT, ""); }
        return queue_pop_tok();
    }
    return 0;
}

    /* ---- Error lexico ---- */
.   {
    fprintf(stderr, "ERROR LEXICO (linea %d): caracter '%s'\n",
            yylineno, yytext);
}

%%

/* =========================================================================
 * SECCION 3 — CODIGO DE USUARIO
 * ========================================================================= */

/*
 * triton_next_token
 * -----------------
 * Funcion que bison llama en lugar de yylex() (via #define en triton.y).
 * Drena la cola de INDENT/DEDENT pendientes antes de pedir al scanner.
 * Actualiza yylloc para que @N.first_line funcione en las reglas de bison.
 */
#ifdef WITH_PARSER
int triton_next_token(void)
{
    int tok;
    if (!queue_empty()) tok = queue_pop_tok();
    else                tok = yylex();
    /* Propagar numero de linea al mecanismo de locations de bison */
    yylloc.first_line  = yylineno;
    yylloc.last_line   = yylineno;
    yylloc.first_column = 0;
    yylloc.last_column  = 0;
    return tok;
}
#endif

/* ---- Modo standalone ---- */
#ifndef WITH_PARSER

static int next_token_standalone(void)
{
    if (!queue_empty()) return queue_pop_tok();
    return yylex();
}

int main(int argc, char *argv[])
{
    int tok;
    FILE *f = NULL;

    if (argc > 1) {
        f = fopen(argv[1], "r");
        if (!f) { fprintf(stderr, "No se pudo abrir '%s'\n", argv[1]); return 1; }
        yyin = f;
    }

    printf("============================================================\n");
    printf("  SECUENCIA DE TOKENS — Triton Scanner\n");
    printf("============================================================\n");
    printf("  %-18s   %-30s   %s\n", "TIPO", "LEXEMA", "LINEA");
    printf("  %-18s   %-30s   %s\n",
           "------------------","------------------------------","-----");

    while ((tok = next_token_standalone()) != 0)
        ;

    imprimir_tabla_simbolos();
    if (f) fclose(f);
    return 0;
}
#endif
