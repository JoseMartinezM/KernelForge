/*
 * triton.y -- Bison parser for the KernelForge Triton JIT subset.
 *
 * The compiler intentionally validates only top-level @triton.jit function
 * blocks.  Host Python around those blocks is skipped by the Flex scanner, so
 * generated samples may include imports, wrappers, launch code, and tests
 * without making this parser a full Python parser.
 */

%locations

%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int triton_next_token(void);
extern int yylineno;
extern FILE *yyin;
extern int compiler_lexical_errors;

#define yylex triton_next_token

void yyerror(const char *message);

#define MAX_KERNELS 256
#define MAX_ITEMS 256

typedef struct {
    char name[128];
    int is_constexpr;
} ParameterInfo;

typedef struct {
    char name[128];
    int line;
    ParameterInfo parameters[MAX_ITEMS];
    int parameter_count;
    char locals[MAX_ITEMS][128];
    int local_count;
    char calls[MAX_ITEMS][128];
    int call_count;
} KernelInfo;

static KernelInfo kernels[MAX_KERNELS];
static int kernel_count = 0;
static KernelInfo *current_kernel = NULL;
static int semantic_errors = 0;
static int syntax_errors = 0;

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

static char *join_text(const char *left, const char *middle, const char *right)
{
    size_t length = strlen(left) + strlen(middle) + strlen(right);
    char *result = (char *)malloc(length + 1);
    if (!result) {
        fprintf(stderr, "fatal: out of memory\n");
        exit(2);
    }
    snprintf(result, length + 1, "%s%s%s", left, middle, right);
    return result;
}

static int text_starts_with(const char *text, const char *prefix)
{
    return strncmp(text, prefix, strlen(prefix)) == 0;
}

static KernelInfo *find_kernel(const char *name)
{
    for (int i = 0; i < kernel_count; i++) {
        if (strcmp(kernels[i].name, name) == 0) {
            return &kernels[i];
        }
    }
    return NULL;
}

static void begin_kernel(const char *name, int line)
{
    if (find_kernel(name)) {
        fprintf(stderr, "SEMANTIC ERROR line %d: duplicate Triton kernel '%s'\n", line, name);
        semantic_errors++;
        current_kernel = NULL;
        return;
    }

    if (kernel_count >= MAX_KERNELS) {
        fprintf(stderr, "SEMANTIC ERROR line %d: too many Triton kernels\n", line);
        semantic_errors++;
        current_kernel = NULL;
        return;
    }

    KernelInfo *kernel = &kernels[kernel_count++];
    memset(kernel, 0, sizeof(*kernel));
    snprintf(kernel->name, sizeof(kernel->name), "%s", name);
    kernel->line = line;
    current_kernel = kernel;
}

static void end_kernel(void)
{
    current_kernel = NULL;
}

static void add_parameter(const char *name, int is_constexpr)
{
    if (!current_kernel || current_kernel->parameter_count >= MAX_ITEMS) {
        return;
    }

    ParameterInfo *parameter = &current_kernel->parameters[current_kernel->parameter_count++];
    snprintf(parameter->name, sizeof(parameter->name), "%s", name);
    parameter->is_constexpr = is_constexpr;
}

static void add_local(const char *name)
{
    if (!current_kernel || !name || name[0] == '\0' || current_kernel->local_count >= MAX_ITEMS) {
        return;
    }

    for (int i = 0; i < current_kernel->local_count; i++) {
        if (strcmp(current_kernel->locals[i], name) == 0) {
            return;
        }
    }

    snprintf(current_kernel->locals[current_kernel->local_count++], 128, "%s", name);
}

static void add_tl_call(const char *name)
{
    if (!current_kernel || !name || !text_starts_with(name, "tl.") || current_kernel->call_count >= MAX_ITEMS) {
        return;
    }

    for (int i = 0; i < current_kernel->call_count; i++) {
        if (strcmp(current_kernel->calls[i], name) == 0) {
            return;
        }
    }

    snprintf(current_kernel->calls[current_kernel->call_count++], 128, "%s", name);
}

static void print_report(void)
{
    printf("\n=== Triton JIT parser report ===\n");
    printf("Kernels parsed: %d\n", kernel_count);

    for (int i = 0; i < kernel_count; i++) {
        KernelInfo *kernel = &kernels[i];
        printf("\n[KERNEL] %s (line %d)\n", kernel->name, kernel->line);

        printf("  Parameters:");
        if (kernel->parameter_count == 0) {
            printf(" none");
        }
        for (int j = 0; j < kernel->parameter_count; j++) {
            printf("%s%s%s",
                   j == 0 ? " " : ", ",
                   kernel->parameters[j].name,
                   kernel->parameters[j].is_constexpr ? ": tl.constexpr" : "");
        }
        printf("\n");

        printf("  Local identifiers:");
        if (kernel->local_count == 0) {
            printf(" none");
        }
        for (int j = 0; j < kernel->local_count; j++) {
            printf("%s%s", j == 0 ? " " : ", ", kernel->locals[j]);
        }
        printf("\n");

        printf("  Triton calls:");
        if (kernel->call_count == 0) {
            printf(" none");
        }
        for (int j = 0; j < kernel->call_count; j++) {
            printf("%s%s", j == 0 ? " " : ", ", kernel->calls[j]);
        }
        printf("\n");
    }

    if (kernel_count == 0) {
        printf("\nNo top-level @triton.jit kernels were found.\n");
    }

    printf("\nErrors: lexical=%d, syntax=%d, semantic=%d\n",
           compiler_lexical_errors, syntax_errors, semantic_errors);
}
%}

%union {
    char *text;
    int flag;
}

%token <text> NAME STRING NUMBER_INT NUMBER_FLOAT NUMBER_HEX NUMBER_BIN
%token DEF IF ELIF ELSE FOR IN WHILE WITH AS RETURN ASSERT PASS
%token AND OR NOT IS TRUE FALSE NONE
%token NEWLINE INDENT DEDENT
%token DOUBLESTAREQ DOUBLESLASHEQ LSHIFTEQ RSHIFTEQ
%token PLUSEQ MINUSEQ STAREQ SLASHEQ PERCENTEQ AMPEQ PIPEEQ CARETEQ
%token EQEQ NOTEQ LTEQ GTEQ
%token DOUBLESTAR DOUBLESLASH LSHIFT RSHIFT ARROW ELLIPSIS
%token PLUS MINUS STAR SLASH PERCENT AMP PIPE CARET TILDE AT
%token LT GT EQ LPAREN RPAREN LBRACKET RBRACKET LBRACE RBRACE
%token COLON COMMA DOT SEMI

%type <text> expr primary atom for_target for_target_list maybe_annotation
%type <flag> annotation_opt

%nonassoc IFX
%right IF ELSE
%left OR
%left AND
%right NOT
%nonassoc EQEQ NOTEQ LT GT LTEQ GTEQ IN IS
%left PIPE
%left CARET
%left AMP
%left LSHIFT RSHIFT
%left PLUS MINUS
%left STAR SLASH DOUBLESLASH PERCENT AT
%right UPLUS UMINUS UTILDE
%right DOUBLESTAR
%left DOT LPAREN LBRACKET

%start program

%%

program
    : jit_blocks
      { print_report(); }
    ;

jit_blocks
    : jit_blocks jit_block
    | /* empty */
    ;

jit_block
    : decorator DEF NAME
      { begin_kernel($3, @3.first_line); }
      LPAREN parameters_opt RPAREN return_annotation_opt COLON suite
      { end_kernel(); }
    ;

decorator
    : AT NAME DOT NAME decorator_call_opt NEWLINE
    ;

decorator_call_opt
    : LPAREN arguments_opt RPAREN
    | /* empty */
    ;

return_annotation_opt
    : ARROW expr
    | /* empty */
    ;

parameters_opt
    : parameter_list comma_opt
    | /* empty */
    ;

parameter_list
    : parameter
    | parameter_list COMMA parameter
    ;

parameter
    : NAME annotation_opt default_opt
      { add_parameter($1, $2); }
    | STAR NAME
      { add_parameter($2, 0); }
    | DOUBLESTAR NAME
      { add_parameter($2, 0); }
    ;

annotation_opt
    : COLON maybe_annotation
      { $$ = strcmp($2, "tl.constexpr") == 0; }
    | /* empty */
      { $$ = 0; }
    ;

maybe_annotation
    : primary
      { $$ = $1; }
    ;

default_opt
    : EQ expr
    | /* empty */
    ;

suite
    : newlines INDENT block DEDENT
    ;

newlines
    : NEWLINE
    | newlines NEWLINE
    ;

block
    : block_item
    | block block_item
    ;

block_item
    : NEWLINE
    | simple_stmt NEWLINE
    | compound_stmt
    ;

simple_stmt
    : small_stmt semi_tail
    ;

semi_tail
    : SEMI small_stmt semi_tail
    | SEMI
    | /* empty */
    ;

small_stmt
    : expr
    | expr EQ expr
      { add_local($1); }
    | expr aug_op expr
      { add_local($1); }
    | expr COLON expr annotated_default_opt
      { add_local($1); }
    | return_stmt
    | assert_stmt
    | PASS
    ;

annotated_default_opt
    : EQ expr
    | /* empty */
    ;

return_stmt
    : RETURN
    | RETURN expr_list
    ;

assert_stmt
    : ASSERT expr
    | ASSERT expr COMMA expr
    ;

compound_stmt
    : if_stmt
    | for_stmt
    | while_stmt
    | with_stmt
    ;

if_stmt
    : IF expr COLON suite elif_parts else_part %prec IFX
    ;

elif_parts
    : elif_parts ELIF expr COLON suite
    | /* empty */
    ;

else_part
    : ELSE COLON suite
    | /* empty */
    ;

for_stmt
    : FOR for_target_list IN expr COLON suite else_part %prec IFX
    ;

while_stmt
    : WHILE expr COLON suite else_part %prec IFX
    ;

with_stmt
    : WITH with_items COLON suite
    ;

with_items
    : with_item
    | with_items COMMA with_item
    ;

with_item
    : expr
    | expr AS for_target
    ;

for_target_list
    : for_target
      { $$ = $1; }
    | for_target_list COMMA for_target
      { $$ = $1; }
    | for_target_list COMMA
      { $$ = $1; }
    ;

for_target
    : NAME
      { $$ = $1; }
    | LPAREN for_target_list_opt RPAREN
      { $$ = copy_text(""); }
    | LBRACKET for_target_list_opt RBRACKET
      { $$ = copy_text(""); }
    ;

for_target_list_opt
    : for_target_list
    | /* empty */
    ;

expr_list
    : expr
    | expr_list COMMA expr
    | expr_list COMMA
    ;

expr_list_opt
    : expr_list
    | /* empty */
    ;

expr
    : expr IF expr ELSE expr
      { $$ = copy_text("conditional"); }
    | expr OR expr
      { $$ = $1; }
    | expr AND expr
      { $$ = $1; }
    | NOT expr
      { $$ = copy_text("not"); }
    | expr EQEQ expr
      { $$ = $1; }
    | expr NOTEQ expr
      { $$ = $1; }
    | expr LT expr
      { $$ = $1; }
    | expr GT expr
      { $$ = $1; }
    | expr LTEQ expr
      { $$ = $1; }
    | expr GTEQ expr
      { $$ = $1; }
    | expr IN expr
      { $$ = $1; }
    | expr IS expr
      { $$ = $1; }
    | expr PIPE expr
      { $$ = $1; }
    | expr CARET expr
      { $$ = $1; }
    | expr AMP expr
      { $$ = $1; }
    | expr LSHIFT expr
      { $$ = $1; }
    | expr RSHIFT expr
      { $$ = $1; }
    | expr PLUS expr
      { $$ = $1; }
    | expr MINUS expr
      { $$ = $1; }
    | expr STAR expr
      { $$ = $1; }
    | expr SLASH expr
      { $$ = $1; }
    | expr DOUBLESLASH expr
      { $$ = $1; }
    | expr PERCENT expr
      { $$ = $1; }
    | expr AT expr
      { $$ = $1; }
    | expr DOUBLESTAR expr
      { $$ = $1; }
    | PLUS expr %prec UPLUS
      { $$ = copy_text("uplus"); }
    | MINUS expr %prec UMINUS
      { $$ = copy_text("uminus"); }
    | TILDE expr %prec UTILDE
      { $$ = copy_text("invert"); }
    | primary
      { $$ = $1; }
    ;

primary
    : atom
      { $$ = $1; }
    | primary DOT NAME
      { $$ = join_text($1, ".", $3); }
    | primary LPAREN arguments_opt RPAREN
      {
          add_tl_call($1);
          $$ = join_text($1, "", "(...)");
      }
    | primary LBRACKET slices_opt RBRACKET
      { $$ = join_text($1, "", "[...]"); }
    ;

atom
    : NAME
      { $$ = $1; }
    | NUMBER_INT
      { $$ = $1; }
    | NUMBER_FLOAT
      { $$ = $1; }
    | NUMBER_HEX
      { $$ = $1; }
    | NUMBER_BIN
      { $$ = $1; }
    | STRING
      { $$ = $1; }
    | TRUE
      { $$ = copy_text("True"); }
    | FALSE
      { $$ = copy_text("False"); }
    | NONE
      { $$ = copy_text("None"); }
    | ELLIPSIS
      { $$ = copy_text("..."); }
    | LPAREN expr_list_opt RPAREN
      { $$ = copy_text("tuple"); }
    | LBRACKET expr_list_opt RBRACKET
      { $$ = copy_text("list"); }
    ;

arguments_opt
    : arguments
    | /* empty */
    ;

arguments
    : argument
    | arguments COMMA argument
    | arguments COMMA
    ;

argument
    : expr
    | NAME EQ expr
    | STAR expr
    | DOUBLESTAR expr
    ;

slices_opt
    : slices
    | /* empty */
    ;

slices
    : slice_item
    | slices COMMA slice_item
    | slices COMMA
    ;

slice_item
    : expr
    | expr_opt COLON expr_opt slice_step_opt
    ;

expr_opt
    : expr
    | /* empty */
    ;

slice_step_opt
    : COLON expr_opt
    | /* empty */
    ;

aug_op
    : PLUSEQ
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
    | DOUBLESTAREQ
    ;

comma_opt
    : COMMA
    | /* empty */
    ;

%%

void yyerror(const char *message)
{
    fprintf(stderr, "SYNTAX ERROR line %d: %s\n", yylineno, message);
    syntax_errors++;
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

    int parse_result = yyparse();

    if (input) {
        fclose(input);
    }

    if (parse_result != 0 || compiler_lexical_errors || syntax_errors || semantic_errors || kernel_count == 0) {
        return 1;
    }
    return 0;
}
