/*
 * triton.l -- Flex scanner for the KernelForge Triton JIT subset.
 *
 * The scanner has two lexical modes:
 *   1. HOST mode skips ordinary Python wrapper code.
 *   2. JIT mode emits tokens only for top-level @triton.jit function blocks.
 *
 * This keeps generated files usable as input without relying on Python's ast
 * module or any external preprocessing step.
 */

%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef WITH_PARSER
  #include "triton.tab.h"
#else
  enum standalone_tokens {
    NAME = 258, STRING, NUMBER_INT, NUMBER_FLOAT, NUMBER_HEX, NUMBER_BIN, NUMBER_OCT,
    DEF, IF, ELIF, ELSE, FOR, IN, WHILE, WITH, AS, RETURN, ASSERT, PASS,
    AND, OR, NOT, IS, TRUE, FALSE, NONE,
    NEWLINE, INDENT, DEDENT,
    DOUBLESTAREQ, DOUBLESLASHEQ, LSHIFTEQ, RSHIFTEQ,
    PLUSEQ, MINUSEQ, STAREQ, SLASHEQ, PERCENTEQ, AMPEQ, PIPEEQ, CARETEQ,
    EQEQ, NOTEQ, LTEQ, GTEQ,
    DOUBLESTAR, DOUBLESLASH, LSHIFT, RSHIFT, ARROW, ELLIPSIS,
    PLUS, MINUS, STAR, SLASH, PERCENT, AMP, PIPE, CARET, TILDE, AT,
    LT, GT, EQ, LPAREN, RPAREN, LBRACKET, RBRACKET, LBRACE, RBRACE,
    COLON, COMMA, DOT, SEMI
  };
#endif

int compiler_lexical_errors = 0;

typedef struct {
    int id;
    char token[32];
    char lexeme[128];
    int line;
} Symbol;

#define MAX_SYMBOLS 4096
static Symbol symbols[MAX_SYMBOLS];
static int symbol_count = 0;

#ifdef WITH_PARSER
static char *copy_text(const char *text)
{
    size_t length = strlen(text);
    char *copy = (char *)malloc(length + 1);
    if (!copy) {
        fprintf(stderr, "fatal: out of memory\n");
        exit(2);
    }
    memcpy(copy, text, length + 1);
    return copy;
}
#endif

static int add_symbol(const char *token, const char *lexeme, int line)
{
    if (!lexeme || lexeme[0] == '\0') {
        return 0;
    }

    for (int i = 0; i < symbol_count; i++) {
        if (strcmp(symbols[i].token, token) == 0 && strcmp(symbols[i].lexeme, lexeme) == 0) {
            return symbols[i].id;
        }
    }

    if (symbol_count >= MAX_SYMBOLS) {
        return -1;
    }

    Symbol *symbol = &symbols[symbol_count];
    symbol->id = symbol_count + 1;
    snprintf(symbol->token, sizeof(symbol->token), "%s", token);
    snprintf(symbol->lexeme, sizeof(symbol->lexeme), "%s", lexeme);
    symbol->line = line;
    symbol_count++;
    return symbol->id;
}

static const char *token_name(int token)
{
    switch (token) {
        case NAME: return "NAME";
        case STRING: return "STRING";
        case NUMBER_INT: return "NUMBER_INT";
        case NUMBER_FLOAT: return "NUMBER_FLOAT";
        case NUMBER_HEX: return "NUMBER_HEX";
        case NUMBER_BIN: return "NUMBER_BIN";
        case NUMBER_OCT: return "NUMBER_OCT";
        case DEF: return "DEF";
        case IF: return "IF";
        case ELIF: return "ELIF";
        case ELSE: return "ELSE";
        case FOR: return "FOR";
        case IN: return "IN";
        case WHILE: return "WHILE";
        case WITH: return "WITH";
        case AS: return "AS";
        case RETURN: return "RETURN";
        case ASSERT: return "ASSERT";
        case PASS: return "PASS";
        case AND: return "AND";
        case OR: return "OR";
        case NOT: return "NOT";
        case IS: return "IS";
        case TRUE: return "TRUE";
        case FALSE: return "FALSE";
        case NONE: return "NONE";
        case NEWLINE: return "NEWLINE";
        case INDENT: return "INDENT";
        case DEDENT: return "DEDENT";
        case DOUBLESTAREQ: return "DOUBLESTAREQ";
        case DOUBLESLASHEQ: return "DOUBLESLASHEQ";
        case LSHIFTEQ: return "LSHIFTEQ";
        case RSHIFTEQ: return "RSHIFTEQ";
        case PLUSEQ: return "PLUSEQ";
        case MINUSEQ: return "MINUSEQ";
        case STAREQ: return "STAREQ";
        case SLASHEQ: return "SLASHEQ";
        case PERCENTEQ: return "PERCENTEQ";
        case AMPEQ: return "AMPEQ";
        case PIPEEQ: return "PIPEEQ";
        case CARETEQ: return "CARETEQ";
        case EQEQ: return "EQEQ";
        case NOTEQ: return "NOTEQ";
        case LTEQ: return "LTEQ";
        case GTEQ: return "GTEQ";
        case DOUBLESTAR: return "DOUBLESTAR";
        case DOUBLESLASH: return "DOUBLESLASH";
        case LSHIFT: return "LSHIFT";
        case RSHIFT: return "RSHIFT";
        case ARROW: return "ARROW";
        case ELLIPSIS: return "ELLIPSIS";
        case PLUS: return "PLUS";
        case MINUS: return "MINUS";
        case STAR: return "STAR";
        case SLASH: return "SLASH";
        case PERCENT: return "PERCENT";
        case AMP: return "AMP";
        case PIPE: return "PIPE";
        case CARET: return "CARET";
        case TILDE: return "TILDE";
        case AT: return "AT";
        case LT: return "LT";
        case GT: return "GT";
        case EQ: return "EQ";
        case LPAREN: return "LPAREN";
        case RPAREN: return "RPAREN";
        case LBRACKET: return "LBRACKET";
        case RBRACKET: return "RBRACKET";
        case LBRACE: return "LBRACE";
        case RBRACE: return "RBRACE";
        case COLON: return "COLON";
        case COMMA: return "COMMA";
        case DOT: return "DOT";
        case SEMI: return "SEMI";
        default: return "TOKEN";
    }
}

#define QUEUE_CAPACITY 256
typedef struct {
    int token;
    char lexeme[128];
    int line;
} PendingToken;

static PendingToken pending[QUEUE_CAPACITY];
static int pending_head = 0;
static int pending_tail = 0;

static int queue_empty(void)
{
    return pending_head == pending_tail;
}

static void queue_push(int token, const char *lexeme, int line)
{
    int index = pending_tail % QUEUE_CAPACITY;
    pending[index].token = token;
    snprintf(pending[index].lexeme, sizeof(pending[index].lexeme), "%s", lexeme ? lexeme : "");
    pending[index].line = line;
    pending_tail++;
}

static int queue_pop(void)
{
    int index = pending_head % QUEUE_CAPACITY;
    PendingToken item = pending[index];
    pending_head++;

#ifdef WITH_PARSER
    if (item.token == NAME || item.token == STRING ||
        item.token == NUMBER_INT || item.token == NUMBER_FLOAT || item.token == NUMBER_HEX ||
        item.token == NUMBER_BIN || item.token == NUMBER_OCT) {
        yylval.text = copy_text(item.lexeme);
    }
#else
    printf("< %-16s , %-24s , line %d >\n", token_name(item.token), item.lexeme, item.line);
#endif

    return item.token;
}

#ifndef WITH_PARSER
static void print_symbol_table(void)
{
    printf("\n=== Symbol table ===\n");
    printf("%-5s %-16s %-28s %s\n", "ID", "TOKEN", "LEXEME", "LINE");
    for (int i = 0; i < symbol_count; i++) {
        printf("%-5d %-16s %-28s %d\n",
               symbols[i].id, symbols[i].token, symbols[i].lexeme, symbols[i].line);
    }
}
#endif

static int emit_token(int token, const char *lexeme)
{
    const char *name = token_name(token);
    add_symbol(name, lexeme, yylineno);
#ifdef WITH_PARSER
    if (token == NAME || token == STRING ||
        token == NUMBER_INT || token == NUMBER_FLOAT || token == NUMBER_HEX ||
        token == NUMBER_BIN || token == NUMBER_OCT) {
        yylval.text = copy_text(lexeme);
    }
#else
    printf("< %-16s , %-24s , line %d >\n", name, lexeme, yylineno);
#endif
    return token;
}

static int paren_depth = 0;
static int host_column = 0;
static int host_line_has_only_space = 1;

#define MAX_INDENT 64
static int indent_stack[MAX_INDENT] = {0};
static int indent_top = 0;
static int saw_jit_body_indent = 0;

static int indentation_width(const char *text, int length)
{
    int width = 0;
    for (int i = 1; i < length; i++) {
        if (text[i] == ' ') {
            width++;
        } else if (text[i] == '\t') {
            width = ((width / 4) + 1) * 4;
        }
    }
    return width;
}

static void reset_jit_state(void)
{
    paren_depth = 0;
    indent_top = 0;
    indent_stack[0] = 0;
    saw_jit_body_indent = 0;
}
%}

%option noyywrap
%option yylineno
%option nodefault
%option noinput
%option nounput

%x JIT

DIGIT      [0-9]
LETTER     [A-Za-z_]
NAME_RE    {LETTER}({LETTER}|{DIGIT})*
HEX        0[xX][0-9a-fA-F]+
BIN        0[bB][01]+
OCT        0[oO][0-7]+
INT        {DIGIT}+
EXP        [eE][+-]?{DIGIT}+
FLOAT      (({DIGIT}+\.{DIGIT}*|\.{DIGIT}+){EXP}?|{DIGIT}+{EXP})
SPACE      [ \t]+

%%

<INITIAL>@triton[ \t]*\.[ \t]*jit {
    if (host_column == 0) {
        reset_jit_state();
        BEGIN(JIT);
        yyless(0);
    } else if (host_line_has_only_space) {
        fprintf(stderr, "LEXICAL ERROR line %d: nested @triton.jit blocks are outside the supported subset\n", yylineno);
        compiler_lexical_errors++;
        host_column += yyleng;
        host_line_has_only_space = 0;
    } else {
        host_column += yyleng;
    }
}
<INITIAL>[ \t]+ {
    for (int i = 0; i < yyleng; i++) {
        if (yytext[i] == ' ') {
            host_column++;
        } else {
            host_column = ((host_column / 4) + 1) * 4;
        }
    }
}
<INITIAL>\n                              { host_column = 0; host_line_has_only_space = 1; }
<INITIAL>[^@\n \t]+                     { host_column += yyleng; host_line_has_only_space = 0; }
<INITIAL>@                               { host_column++; host_line_has_only_space = 0; }
<INITIAL><<EOF>>                         { return 0; }

<JIT>#[^\n]*                             { /* skip comments; newline is handled separately */ }

<JIT>[fFrRbBuU]*\"\"\"([^"\\]|\\.|\"[^"\\]|\"\"[^"\\])*\"\"\" { return emit_token(STRING, yytext); }
<JIT>[fFrRbBuU]*\'\'\'([^'\\]|\\.|\'[^'\\]|\'\'[^'\\])*\'\'\' { return emit_token(STRING, yytext); }
<JIT>[fFrRbBuU]*\"([^\"\\\n]|\\.)*\"     { return emit_token(STRING, yytext); }
<JIT>[fFrRbBuU]*\'([^\'\\\n]|\\.)*\'     { return emit_token(STRING, yytext); }

<JIT>{FLOAT}                              { return emit_token(NUMBER_FLOAT, yytext); }
<JIT>{HEX}                                { return emit_token(NUMBER_HEX, yytext); }
<JIT>{BIN}                                { return emit_token(NUMBER_BIN, yytext); }
<JIT>{OCT}                                { return emit_token(NUMBER_OCT, yytext); }
<JIT>{INT}                                { return emit_token(NUMBER_INT, yytext); }

<JIT>{NAME_RE} {
    if      (strcmp(yytext, "def") == 0)    return emit_token(DEF, yytext);
    else if (strcmp(yytext, "if") == 0)     return emit_token(IF, yytext);
    else if (strcmp(yytext, "elif") == 0)   return emit_token(ELIF, yytext);
    else if (strcmp(yytext, "else") == 0)   return emit_token(ELSE, yytext);
    else if (strcmp(yytext, "for") == 0)    return emit_token(FOR, yytext);
    else if (strcmp(yytext, "in") == 0)     return emit_token(IN, yytext);
    else if (strcmp(yytext, "while") == 0)  return emit_token(WHILE, yytext);
    else if (strcmp(yytext, "with") == 0)   return emit_token(WITH, yytext);
    else if (strcmp(yytext, "as") == 0)     return emit_token(AS, yytext);
    else if (strcmp(yytext, "return") == 0) return emit_token(RETURN, yytext);
    else if (strcmp(yytext, "assert") == 0) return emit_token(ASSERT, yytext);
    else if (strcmp(yytext, "pass") == 0)   return emit_token(PASS, yytext);
    else if (strcmp(yytext, "and") == 0)    return emit_token(AND, yytext);
    else if (strcmp(yytext, "or") == 0)     return emit_token(OR, yytext);
    else if (strcmp(yytext, "not") == 0)    return emit_token(NOT, yytext);
    else if (strcmp(yytext, "is") == 0)     return emit_token(IS, yytext);
    else if (strcmp(yytext, "True") == 0)   return emit_token(TRUE, yytext);
    else if (strcmp(yytext, "False") == 0)  return emit_token(FALSE, yytext);
    else if (strcmp(yytext, "None") == 0)   return emit_token(NONE, yytext);
    return emit_token(NAME, yytext);
}

<JIT>"**="                                { return emit_token(DOUBLESTAREQ, yytext); }
<JIT>"//="                                { return emit_token(DOUBLESLASHEQ, yytext); }
<JIT>"<<="                                { return emit_token(LSHIFTEQ, yytext); }
<JIT>">>="                                { return emit_token(RSHIFTEQ, yytext); }
<JIT>"+="                                 { return emit_token(PLUSEQ, yytext); }
<JIT>"-="                                 { return emit_token(MINUSEQ, yytext); }
<JIT>"*="                                 { return emit_token(STAREQ, yytext); }
<JIT>"/="                                 { return emit_token(SLASHEQ, yytext); }
<JIT>"%="                                 { return emit_token(PERCENTEQ, yytext); }
<JIT>"&="                                 { return emit_token(AMPEQ, yytext); }
<JIT>"|="                                 { return emit_token(PIPEEQ, yytext); }
<JIT>"^="                                 { return emit_token(CARETEQ, yytext); }

<JIT>"=="                                 { return emit_token(EQEQ, yytext); }
<JIT>"!="                                 { return emit_token(NOTEQ, yytext); }
<JIT>"<="                                 { return emit_token(LTEQ, yytext); }
<JIT>">="                                 { return emit_token(GTEQ, yytext); }
<JIT>"**"                                 { return emit_token(DOUBLESTAR, yytext); }
<JIT>"//"                                 { return emit_token(DOUBLESLASH, yytext); }
<JIT>"<<"                                 { return emit_token(LSHIFT, yytext); }
<JIT>">>"                                 { return emit_token(RSHIFT, yytext); }
<JIT>"->"                                 { return emit_token(ARROW, yytext); }
<JIT>"..."                                { return emit_token(ELLIPSIS, yytext); }

<JIT>"+"                                  { return emit_token(PLUS, yytext); }
<JIT>"-"                                  { return emit_token(MINUS, yytext); }
<JIT>"*"                                  { return emit_token(STAR, yytext); }
<JIT>"/"                                  { return emit_token(SLASH, yytext); }
<JIT>"%"                                  { return emit_token(PERCENT, yytext); }
<JIT>"&"                                  { return emit_token(AMP, yytext); }
<JIT>"|"                                  { return emit_token(PIPE, yytext); }
<JIT>"^"                                  { return emit_token(CARET, yytext); }
<JIT>"~"                                  { return emit_token(TILDE, yytext); }
<JIT>"@"                                  { return emit_token(AT, yytext); }
<JIT>"<"                                  { return emit_token(LT, yytext); }
<JIT>">"                                  { return emit_token(GT, yytext); }
<JIT>"="                                  { return emit_token(EQ, yytext); }

<JIT>"("                                  { paren_depth++; return emit_token(LPAREN, yytext); }
<JIT>"["                                  { paren_depth++; return emit_token(LBRACKET, yytext); }
<JIT>"{"                                  { paren_depth++; return emit_token(LBRACE, yytext); }
<JIT>")"                                  { if (paren_depth > 0) paren_depth--; return emit_token(RPAREN, yytext); }
<JIT>"]"                                  { if (paren_depth > 0) paren_depth--; return emit_token(RBRACKET, yytext); }
<JIT>"}"                                  { if (paren_depth > 0) paren_depth--; return emit_token(RBRACE, yytext); }
<JIT>":"                                  { return emit_token(COLON, yytext); }
<JIT>","                                  { return emit_token(COMMA, yytext); }
<JIT>"."                                  { return emit_token(DOT, yytext); }
<JIT>";"                                  { return emit_token(SEMI, yytext); }

<JIT>\n[ \t]*/[\n#] {
    if (paren_depth == 0) {
        queue_push(NEWLINE, "\\n", yylineno);
        return queue_pop();
    }
}

<JIT>\n[ \t]* {
    if (paren_depth == 0) {
        int width = indentation_width(yytext, yyleng);

        queue_push(NEWLINE, "\\n", yylineno);

        if (width > indent_stack[indent_top]) {
            if (indent_top + 1 >= MAX_INDENT) {
                fprintf(stderr, "LEXICAL ERROR line %d: indentation nesting is too deep\n", yylineno);
                compiler_lexical_errors++;
            } else {
                indent_stack[++indent_top] = width;
                saw_jit_body_indent = 1;
                queue_push(INDENT, "", yylineno);
            }
        } else {
            while (indent_top > 0 && indent_stack[indent_top] > width) {
                indent_top--;
                queue_push(DEDENT, "", yylineno);
            }
            if (indent_stack[indent_top] != width) {
                fprintf(stderr, "LEXICAL ERROR line %d: inconsistent indentation\n", yylineno);
                compiler_lexical_errors++;
            }
        }

        if (saw_jit_body_indent && indent_top == 0 && width == 0) {
            host_column = width;
            host_line_has_only_space = 1;
            BEGIN(INITIAL);
            saw_jit_body_indent = 0;
        }

        return queue_pop();
    }
}

<JIT>\\\n                                { /* explicit line continuation */ }
<JIT>{SPACE}                             { /* skip inline whitespace */ }

<JIT><<EOF>> {
    if (!queue_empty()) {
        return queue_pop();
    }
    if (indent_top > 0) {
        queue_push(NEWLINE, "\\n", yylineno);
        while (indent_top > 0) {
            indent_top--;
            queue_push(DEDENT, "", yylineno);
        }
        return queue_pop();
    }
    return 0;
}

<JIT>. {
    fprintf(stderr, "LEXICAL ERROR line %d: unsupported character '%s'\n", yylineno, yytext);
    compiler_lexical_errors++;
}

%%

#ifdef WITH_PARSER
int triton_next_token(void)
{
    int token;
    if (!queue_empty()) {
        token = queue_pop();
    } else {
        token = yylex();
    }

    yylloc.first_line = yylineno;
    yylloc.last_line = yylineno;
    yylloc.first_column = 0;
    yylloc.last_column = 0;
    return token;
}
#endif

#ifndef WITH_PARSER
static int next_token(void)
{
    if (!queue_empty()) {
        return queue_pop();
    }
    return yylex();
}

int main(int argc, char **argv)
{
    FILE *input = NULL;

    if (argc > 1) {
        input = fopen(argv[1], "r");
        if (!input) {
            fprintf(stderr, "error: could not open '%s'\n", argv[1]);
            return 1;
        }
        yyin = input;
    }

    printf("=== Token sequence (Triton JIT blocks only) ===\n");
    while (next_token() != 0) {
        /* printing is done when each token is emitted */
    }
    print_symbol_table();

    if (input) {
        fclose(input);
    }
    return compiler_lexical_errors ? 1 : 0;
}
#endif
