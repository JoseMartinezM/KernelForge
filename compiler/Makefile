# Makefile — Triton Compiler
#
# FASE 1 — Scanner standalone:
#   make scanner   -> compila triton.l y produce ./triton_scanner
#   make test      -> corre los casos de prueba
#
# FASE 2 — Parser completo (siguiente fase del proyecto):
#   make compiler  -> compila triton.l + triton.y y produce ./triton_compiler
#
# make clean       -> elimina archivos generados

CC      = gcc
CFLAGS  = -Wall -g
LDFLAGS = -lfl

# ── FASE 1: Scanner (solo triton.l) ───────────────────────────────
#
# Flujo:
#   triton.l  --flex-->  lex.yy.c  --gcc-->  triton_scanner

scanner: lex.yy.c
	$(CC) $(CFLAGS) lex.yy.c $(LDFLAGS) -o triton_scanner

lex.yy.c: triton.l
	flex triton.l

# ── FASE 2: Compiler completo (triton.l + triton.y) ───────────────
#
# Flujo:
#   triton.y  --bison-d-->  triton.tab.c + triton.tab.h
#   triton.l  --flex-->     lex.yy.c
#   gcc triton.tab.c lex.yy.c --> triton_compiler

compiler: triton.tab.c lex.yy.c
	$(CC) $(CFLAGS) lex.yy.c triton.tab.c $(LDFLAGS) -o triton_compiler

triton.tab.c triton.tab.h: triton.y
	bison -d triton.y

# ── Tests ─────────────────────────────────────────────────────────
test: scanner
	@echo "\n=== TEST 1: vector_add ==="
	./triton_scanner fixtures/valid/vector_add.py

	@echo "\n=== TEST 2: softmax ==="
	./triton_scanner fixtures/valid/softmax_kernel.py

	@echo "\n=== TEST 3: bad_syntax (debe mostrar error léxico) ==="
	./triton_scanner fixtures/invalid/bad_syntax.py || true

	@echo "\n=== TEST 4: missing_jit ==="
	./triton_scanner fixtures/invalid/missing_jit.py

# ── Limpieza ──────────────────────────────────────────────────────
clean:
	rm -f lex.yy.c triton.tab.c triton.tab.h
	rm -f triton_scanner triton_compiler parser.out
