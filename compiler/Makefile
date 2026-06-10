LEX := flex
CC ?= cc
CFLAGS ?= -O2

LEX_FILE := triton.l
LEX_C := lex.yy.c
TARGET := triton_lexer

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(LEX_C)
	$(CC) $(CFLAGS) -o $@ $<

$(LEX_C): $(LEX_FILE)
	$(LEX) $<

run: $(TARGET)
	printf 'kernel add(x: 42) { return x + 1; }\n' | ./$(TARGET)

clean:
	rm -f $(TARGET) $(LEX_C)
