%{
#include <stdio.h>
%}

%option noyywrap

DIGIT      [0-9]
LETTER     [A-Za-z_]
IDENT      {LETTER}({LETTER}|{DIGIT})*

%%

"kernel"          { printf("KEYWORD(kernel)\n"); }
"return"          { printf("KEYWORD(return)\n"); }
{IDENT}            { printf("IDENT(%s)\n", yytext); }
{DIGIT}+           { printf("NUMBER(%s)\n", yytext); }
"=="|"!="|"<="|">=" { printf("OP(%s)\n", yytext); }
[+\-*/=<>]        { printf("OP(%s)\n", yytext); }
[(){}:,;]          { printf("PUNCT(%s)\n", yytext); }
[ \t\r\n]+        { /* skip whitespace */ }
.                  { printf("UNKNOWN(%s)\n", yytext); }

%%

int main(void) {
    return yylex();
}
