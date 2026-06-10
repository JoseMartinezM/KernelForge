LEX := flex
YACC := bison
CC ?= cc
CFLAGS ?= -Wall -Wextra -g

LEX_FILE := triton.l
YACC_FILE := triton.y
LEX_C := lex.yy.c
YACC_C := triton.tab.c
YACC_H := triton.tab.h
SCANNER := triton_scanner
COMPILER := triton_compiler

.PHONY: all scanner compiler run test clean

all: compiler

scanner: $(SCANNER)

compiler: $(COMPILER)

$(SCANNER): $(LEX_C)
	$(CC) $(CFLAGS) -o $@ $<

$(COMPILER): $(YACC_C) $(YACC_H) $(LEX_C)
	$(CC) $(CFLAGS) -DWITH_PARSER -o $@ $(YACC_C) $(LEX_C)

$(LEX_C): $(LEX_FILE)
	$(LEX) $(LEX_FILE)

$(YACC_C) $(YACC_H): $(YACC_FILE)
	$(YACC) -d $(YACC_FILE)

run: $(COMPILER)
	./$(COMPILER) ../tests/compiler/fixtures/valid/vector_add.py

test: $(COMPILER) $(SCANNER)
	./$(COMPILER) ../tests/compiler/fixtures/valid/vector_add.py
	./$(SCANNER) ../tests/compiler/fixtures/valid/vector_add.py >/dev/null

clean:
	rm -f $(SCANNER) $(COMPILER) $(LEX_C) $(YACC_C) $(YACC_H) parser.output
