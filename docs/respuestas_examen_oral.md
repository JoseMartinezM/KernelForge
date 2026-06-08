# Respuestas — Banco de Preguntas Examen Oral
## Direct Syntax Translator para Código Triton-GPU (KernelForge)

> **Nota:** Las preguntas del banco mencionan "Tryton-oriented code". En nuestro proyecto, eso equivale a **Triton-GPU oriented code** — la misma idea aplicada a kernels de GPU escritos con la librería `triton` de OpenAI.

---

## 1. Diseño General del Compilador

**¿Cuál es el objetivo principal de tu compilador?**
El objetivo es *validar* código Python que contiene kernels Triton-GPU. El compilador lee el código fuente, lo tokeniza, lo parsea con una gramática BNF, construye una tabla de símbolos durante el parsing, y genera un reporte indicando si el código es un kernel Triton válido (tiene `@triton.jit`, parámetros `tl.constexpr`, etc.) o tiene errores.

**¿Qué significa "Triton-oriented code" en tu proyecto?**
Son funciones Python decoradas con `@triton.jit` que siguen el patrón de programación de kernels GPU: reciben punteros a memoria, usan `tl.program_id` para paralelismo, calculan offsets con `tl.arange`, y leen/escriben memoria con `tl.load`/`tl.store`. Los parámetros de tamaño de bloque se anotan como `tl.constexpr`.

**¿Qué tipo de código acepta como entrada?**
Código Python 3 con las extensiones de Triton-GPU. El lenguaje es un subconjunto de Python: definiciones de función, decoradores, asignaciones, estructuras de control (`if/elif/else`, `for`, `while`), expresiones con operadores aritméticos, de comparación y bit-a-bit, e imports. La indentación tiene semántica (como Python).

**¿Qué tipo de salida genera?**
No genera código ejecutable. Genera un **reporte de validación** en texto plano que lista:
- Cada función encontrada (`[KERNEL]` si tiene `@triton.jit`, `[FUNC]` si no)
- Sus parámetros y cuáles son `tl.constexpr`
- Variables locales asignadas
- Llamadas a la API de Triton (`tl.load`, `tl.store`, etc.)
- Errores semánticos (función duplicada, falta `@triton.jit`)

**¿Es un compilador completo, un intérprete o un Direct Syntax Translator?**
Es un **Direct Syntax Translator (DST)**. Un compilador completo construye primero un AST completo y luego lo recorre para generar código máquina. Un intérprete ejecuta el código directamente. Nuestro DST ejecuta acciones de traducción *durante* el parsing, en cada regla gramatical que se reduce. No hay un AST separado; la "salida" se genera de forma incremental conforme el parser reconoce estructuras del lenguaje.

**¿Cuáles son las fases principales?**
1. **Análisis léxico** (`triton_lexer.py`) — convierte el texto en tokens
2. **Análisis sintáctico + traducción** (`triton_parser.py`) — valida la gramática y ejecuta acciones semánticas simultáneamente
3. **Reporte** (`triton_validator.py`) — consolida la tabla de símbolos en un `ValidationResult` estructurado

**¿Qué fase fue más difícil de implementar y por qué?**
El manejo de indentación en el lexer. Python usa espacios para delimitar bloques en vez de `{}`; el lexer debe convertir eso en tokens `INDENT`/`DEDENT`. El caso difícil son las **líneas en blanco dentro de bloques**: una línea vacía tiene indent=0, lo que falsamente dispararía `DEDENT` al interior de una función. Se resolvió con un algoritmo de *lookahead*: si un `NEWLINE` va seguido inmediatamente de otro `NEWLINE`, es una línea en blanco y se ignora para efectos de indentación.

**¿Qué suposiciones hiciste sobre el programa de entrada?**
- La indentación usa espacios (4 por nivel) o tabulaciones consistentes, no mezclas.
- Las líneas lógicas que continúan entre paréntesis `( )` o corchetes `[ ]` son válidas.
- El archivo es Python 3; no se soporta Python 2.
- Los kernels Triton siempre están dentro de funciones con `def`.

---

## 2. Análisis Léxico

**¿Cuáles son los tokens del lenguaje?**
La lista completa está en `compiler/triton_lexer.py`. Los principales son:
- `INDENT`, `DEDENT`, `NEWLINE` — control de bloques
- `NAME` — identificadores genéricos
- `NUMBER` — enteros (decimal, hexadecimal, binario, octal) y floats
- `STRING` — literales de cadena (simple, doble, triple)
- `AT` — símbolo `@` para decoradores
- Todas las **palabras reservadas de Python**: `DEF`, `IF`, `ELSE`, `ELIF`, `FOR`, `WHILE`, `RETURN`, `IMPORT`, `FROM`, `CLASS`, `AND`, `OR`, `NOT`, `IN`, `IS`, etc.
- **Operadores**: `PLUS`, `MINUS`, `STAR`, `SLASH`, `DOUBLESTAR` (`**`), `EQEQ` (`==`), `LTEQ` (`<=`), `LSHIFT` (`<<`), `RSHIFT` (`>>`), etc.
- **Delimitadores**: `LPAREN`, `RPAREN`, `LBRACKET`, `RBRACKET`, `COLON`, `COMMA`, `DOT`

**¿Qué tokens son específicos al código Triton-oriented?**
No hay tokens *exclusivos* de Triton a nivel léxico porque Triton se expresa como Python normal. Sin embargo, el lexer reconoce el token `AT` (símbolo `@`) que el parser luego usa para decoradores. El identificador `triton`, `tl`, `jit`, etc., todos son `NAME` — el parser y las acciones semánticas saben que `tl.load` es una llamada Triton, no el lexer.

**¿Cómo distingues entre identificadores, keywords y palabras reservadas?**
El lexer tiene una función `t_NAME` con la regex `[A-Za-z_][A-Za-z0-9_]*`. Después de capturar el string, consulta un diccionario `_KEYWORDS`. Si el string está en el diccionario, cambia el tipo del token al keyword correspondiente (`"def"` → `DEF`, `"if"` → `IF`, etc.). Si no está, queda como `NAME`. Esto es el patrón estándar en PLY.

```python
def t_NAME(self, t):
    r"[A-Za-z_][A-Za-z0-9_]*"
    t.type = _KEYWORDS.get(t.value, "NAME")  # default: NAME
    return t
```

**¿Qué errores léxicos puede detectar?**
- Carácter ilegal (símbolo no reconocido): `LexError` con número de línea.
- Error de indentación (nivel que no coincide con ningún bloque abierto): se registra en `_errors`.

**¿Qué pasa si el input tiene un símbolo desconocido?**
La función `t_error` lanza una excepción `LexError` con el mensaje `"Carácter ilegal 'X' en línea N"`. El parser la captura y la reporta como error de parseo.

**¿Cómo manejas comentarios, espacios y saltos de línea?**
- **Comentarios** (`#...`): la regla `t_COMMENT` consume el texto hasta el fin de línea y no retorna nada (PLY ignora los tokens que retornan `None`).
- **Espacios/tabs en medio de línea**: la regla `t_WHITESPACE` los ignora. Solo son significativos al *inicio* de una línea, donde el token `NEWLINE` captura `\n[ \t]*` y el `IndentLexer` mide esos espacios.
- **Saltos de línea**: se convierten en `NEWLINE`. Dentro de paréntesis se ignoran (continuación implícita de línea lógica, variable `_paren_depth`).

**¿Puedes mostrar un ejemplo de código y los tokens generados?**
Para el código:
```python
@triton.jit
def add(x, n: tl.constexpr):
    pass
```
Los tokens son:
```
AT  NAME('triton')  DOT  NAME('jit')  NEWLINE
DEF  NAME('add')  LPAREN  NAME('x')  COMMA  NAME('n')  COLON  NAME('tl')  DOT  NAME('constexpr')  RPAREN  COLON  NEWLINE
INDENT  PASS  NEWLINE  DEDENT
```

**¿Qué herramienta usaste para el análisis léxico?**
**PLY** (Python Lex-Yacc), versión 3.11. Es una implementación pura en Python del algoritmo de Lex (usando expresiones regulares con el módulo `re` de Python). No es Flex, pero implementa el mismo algoritmo: las reglas son expresiones regulares y se selecciona la más larga que hace match al inicio del input.

---

## 3. Análisis Sintáctico y Gramática

**¿Qué gramática definiste para tu lenguaje?**
Una gramática BNF para un subconjunto de Python que cubre:
- Definiciones de función con decoradores, parámetros y anotaciones de tipo
- Instrucciones: asignación, asignación compuesta (`+=`, etc.), asignación anotada (`:=`), `return`, `pass`, `assert`, `import`, `from...import`
- Control de flujo: `if/elif/else`, `for`, `while`
- Expresiones: aritmética completa con jerarquía de precedencia (11 niveles), comparaciones, operaciones lógicas, acceso a atributos (`tl.load`), llamadas a función, subscripts (`x[i]`), slices, operadores ternarios
- Bloques: `suite → NEWLINE INDENT stmt_seq DEDENT`

**¿Tu gramática es ambigua? ¿Cómo lo sabes?**
No, no es ambigua. PLY usa el algoritmo **LALR(1)** que solo funciona con gramáticas no ambiguas (o con reglas de desempate explícitas). Si hubiera ambigüedad, PLY reportaría conflictos shift/reduce o reduce/reduce irresolubles. Al compilar la gramática, PLY no reportó ningún conflicto que no hayamos resuelto explícitamente con la tabla de precedencia.

**¿Tenías recursión izquierda? ¿Cómo la eliminaste?**
Sí. Las expresiones binarias naturalmente tienen recursión izquierda. Por ejemplo:
```
expr → expr '+' term | term         ← recursión izquierda
```

La técnica de eliminación (requerida por el profesor) transforma `A → A α | β` en:
```
A  → β A'
A' → α A' | ε
```

Aplicado a suma:
```
sum_expr  → term_expr sum_tail
sum_tail  → PLUS term_expr sum_tail
           | MINUS term_expr sum_tail
           | ε
```

Esto se repite para **cada nivel** de la jerarquía de expresiones: `or_expr`, `and_expr`, `bitor_expr`, `bitxor_expr`, `bitand_expr`, `shift_expr`, `sum_expr`, `term_expr`. La asociatividad izquierda se reconstruye con la función `_fold_tail(base, tail)` en las acciones semánticas.

**¿Qué estrategia de parsing usaste?**
**LALR(1)** — el algoritmo de PLY Yacc. LALR(1) es un parser *bottom-up* que construye una tabla de estados basándose en los "items LR" del autómata. Lee un token de *lookahead* (1 símbolo) para decidir si hacer *shift* (leer más) o *reduce* (aplicar una producción).

**¿Qué errores de sintaxis puede detectar?**
- Token inesperado (p.ej., `def` sin `:` al final)
- Bloque vacío sin `pass`
- Paréntesis sin cerrar
- Expresión incompleta

**¿Qué pasa si el usuario olvida dos puntos, paréntesis o delimitador de bloque?**
PLY llama a la función `p_error(p)` que registra el error con número de línea en `_symtab.errors`. El parser intenta continuar pero el resultado será un árbol incompleto. El reporte final marca `is_valid = False`.

**¿Puedes explicar una regla de producción de tu gramática?**
La regla para definición de función con decorador:
```python
def p_funcdef_decorated(p):
    """funcdef : decorator_list DEF NAME LPAREN param_list RPAREN COLON suite"""
    name = p[3]
    params = p[5]
    decorators = p[1]
    _symtab.declare_function(name, p.lineno(3), params, decorators)
    p[0] = ("funcdef", name, params, decorators)
```
La lista de parámetros ya fue reducida antes (reglas `param_list`), y los decoradores también. Esta regla simplemente ensambla todo y registra la función en la tabla de símbolos.

**¿Tenías conflictos shift/reduce o reduce/reduce? ¿Cómo los resolviste?**
Sí, conflictos shift/reduce de **precedencia de operadores**. Por ejemplo, `a + b * c` puede reducir `a+b` y luego multiplicar, o primero multiplicar y luego sumar. PLY tiene el mecanismo `precedence`:

```python
precedence = (
    ("left",  "OR"),
    ("left",  "AND"),
    ("right", "NOT"),
    ("left",  "LT", "GT", "EQEQ", "NOTEQ", "LTEQ", "GTEQ"),
    ("left",  "PIPE"),
    ("left",  "CARET"),
    ("left",  "AMP"),
    ("left",  "LSHIFT", "RSHIFT"),
    ("left",  "PLUS", "MINUS"),
    ("left",  "STAR", "SLASH", "DOUBLESLASH", "PERCENT"),
    ("right", "TILDE", "UMINUS"),
    ("left",  "DOUBLESTAR"),
    ("left",  "DOT", "LPAREN", "LBRACKET"),
)
```
Los niveles más bajos tienen menor precedencia. `("left", "PLUS", "MINUS")` dice que `+` y `-` son left-associative (resuelve el conflicto shift/reduce a favor de reduce primero). Sin embargo, **para cumplir la exigencia del profesor de no left recursion**, nuestras reglas gramáticales tampoco son izquierdo-recursivas — la tabla de precedencia sirve como capa extra de seguridad.

---

## 4. Análisis Semántico

**¿Tu compilador verifica errores semánticos además de sintácticos?**
Sí. Los errores semánticos que detecta:
- Función declarada más de una vez (nombre duplicado)
- (Advertencia) Función sin decorador `@triton.jit` — puede ser intencional pero se avisa

**¿Usas tabla de símbolos?**
Sí, la clase `SymbolTable` en `triton_parser.py`.

**¿Qué información guardas en la tabla de símbolos?**
Por cada función:
- `linea`: número de línea de la declaración
- `tiene_triton_jit`: booleano — si el decorador `@triton.jit` está presente
- `parametros`: lista de `{nombre, anotacion, es_constexpr}` — `es_constexpr` es `True` si la anotación contiene `constexpr`
- `variables_locales`: nombres de variables asignadas dentro del cuerpo
- `llamadas_triton`: lista de llamadas a `tl.*` detectadas (p.ej. `tl.load`, `tl.program_id`)

**¿Cómo detectas variables no declaradas?**
No implementamos detección completa de variables no declaradas (eso requeriría scoping completo). Sí rastreamos las variables *locales* (del lado izquierdo de una asignación). Para un compilador real se haría en una segunda pasada sobre el AST.

**¿Cómo detectas declaraciones duplicadas?**
En `declare_function()`:
```python
if name in self.functions:
    self.errors.append(f"Línea {lineno}: función '{name}' ya declarada")
    return
```

**¿Verificas tipos de datos?**
No. Python es dinámicamente tipado y la inferencia de tipos completa requeriría un sistema mucho más complejo. Solo verificamos las anotaciones de tipo explícitas (como `tl.constexpr`) para registrarlas en la tabla de símbolos.

**¿Qué pasa si el programa es sintácticamente correcto pero semánticamente inválido?**
El parser lo acepta (no hay error de sintaxis), pero las acciones semánticas añaden entradas a `_symtab.errors` o `_symtab.warnings`. La función `validate()` retorna `is_valid = False` si hay errores.

**¿Qué reglas semánticas son específicas a código Triton?**
- Si una función tiene `@triton.jit`, se marca como kernel; si no lo tiene, se genera una advertencia.
- Parámetros con anotación que contiene `constexpr` se marcan `es_constexpr = True` (patrón obligatorio de Triton para constantes en tiempo de compilación GPU).
- Las llamadas a funciones que empiezan con `tl.` se registran como "llamadas Triton" en la tabla.

---

## 5. Direct Syntax Translation

**¿Qué es un Direct Syntax Translator?**
Es un compilador cuya *salida se produce directamente durante el parsing*, sin construir un árbol de sintaxis abstracta (AST) separado. Las reglas gramaticales llevan **acciones semánticas** que se ejecutan en el momento en que esa producción se reduce (en parsers bottom-up) o se termina de reconocer (en top-down). La "traducción" ocurre en paralelo con el reconocimiento.

**¿En qué se diferencia de un compilador tradicional?**

| Compilador tradicional | Direct Syntax Translator |
|------------------------|--------------------------|
| Fuente → tokens → **AST** → IR → código | Fuente → tokens → traducción **durante parsing** |
| Dos o más pasadas sobre el árbol | Una sola pasada |
| Flexible para optimizaciones | Simple, menos código |
| Necesario para recursión arbitraria en el árbol | Limitado a lo que se puede hacer en una pasada |

Nuestro DST hace todo en una pasada: parsea y llena la tabla de símbolos simultáneamente.

**¿Dónde adjuntas las acciones de traducción en tu gramática?**
En las funciones de reglas de PLY. Cada función `p_algo(p)` tiene una acción al final que:
1. Asigna `p[0]` (el valor del símbolo en el lado izquierdo de la producción)
2. Opcionalmente llama métodos de `_symtab` para registrar información

Por ejemplo en `p_funcdef_decorated` se llama `_symtab.declare_function(...)` y en `p_assign_simple` se llama `_symtab.add_local_var(...)`.

**¿Generas la salida durante el parsing o después de construir una representación intermedia?**
Durante el parsing. Los valores de `p[0]` forman sub-árboles simples (tuplas Python), pero la tabla de símbolos se llena conforme el parser reduce. Al final del parsing, `_symtab.report()` genera el reporte de texto sin necesidad de recorrer un árbol separado.

**¿Cuáles son tus acciones semánticas?**
- `p_funcdef_decorated`: `_symtab.declare_function(nombre, linea, parámetros, decoradores)`
- `p_decorator`: convierte el nodo AST del decorador a string (`_expr_to_str`) para comparar `"triton.jit"`
- `p_assign_simple`: `_symtab.add_local_var(función_actual, variable)` — registra variables locales
- `p_call_expr`: `_symtab.add_triton_call(función_actual, "tl.función")` — detecta llamadas `tl.*`
- `p_error`: registra el error con línea en `_symtab.errors`

**¿Puedes mostrar una regla con su acción de traducción?**
```python
def p_decorator(p):
    """decorator : AT expr NEWLINE"""
    # Convertimos el nodo AST del decorador a string legible.
    # p[2] puede ser ("name","triton") o ("access",("name","triton"),("attr","jit"))
    p[0] = _expr_to_str(p[2])  # → "triton" o "triton.jit"

def _expr_to_str(node) -> str:
    if isinstance(node, tuple):
        if node[0] == "name":
            return node[1]
        if node[0] == "access":
            return _expr_to_str(node[1]) + "." + node[2][1]
        if node[0] == "call":
            return _expr_to_str(node[1]) + "()"
    return str(node)
```

**¿Cómo garantizas que la traducción preserva el significado?**
El reporte es *descriptivo*, no ejecutable — lista lo que encontró, no genera código. Por lo tanto "preservar significado" equivale a: si el parser reconoció exitosamente una función, la información registrada en la tabla de símbolos es correcta. Esto se valida con pruebas unitarias en `tests/compiler/test_triton_parser.py` (41 pruebas).

**¿Qué pasa si el input es válido sintácticamente pero no puede traducirse a Triton?**
El reporte lo marca con el tag `[FUNC]` (en vez de `[KERNEL]`) y genera una advertencia: `"Función 'x' sin @triton.jit"`. Esto es una distinción semántica, no de sintaxis.

---

## 6. Detección de Código Triton-Oriented

**¿Qué características de Triton intentas detectar?**
1. Decorador `@triton.jit` (señal de que es un kernel GPU)
2. Parámetros con anotación `tl.constexpr` (constantes en tiempo de compilación)
3. Llamadas a `tl.program_id` (índice de hilo paralelo)
4. Llamadas a `tl.arange` (generación de vectores de índices)
5. Llamadas a `tl.load` y `tl.store` (lectura/escritura de memoria GPU)

**¿Qué partes del lenguaje mapean a conceptos Triton?**
- El **decorador** `@triton.jit` → marca la función como un kernel que corre en GPU
- La **anotación de tipo** `tl.constexpr` en parámetros → valores fijados en compile time del GPU
- Las **llamadas a función** `tl.program_id(0)` → obtiene el ID del bloque paralelo (como `blockIdx.x` en CUDA)
- `tl.load(ptr + offsets, mask=mask)` → carga vectorizada de memoria con predicado
- `tl.store(ptr + offsets, value, mask=mask)` → escritura vectorizada

**¿Detectas modelos, campos, métodos, flujos de trabajo?**
Detectamos **kernels** (funciones decoradas con `@triton.jit`), sus **parámetros** con anotaciones de tipo, sus **variables locales**, y sus **llamadas a la API** `tl.*`. No manejamos clases ni herencia.

**¿Cómo representas estructuras Triton-específicas en tu gramática?**
No hay reglas gramaticales especiales para Triton — a nivel sintáctico todo es Python normal. La especificidad de Triton se captura en las **acciones semánticas**: la regla `p_call_expr` examina si la función llamada empieza con `tl.` y la registra como llamada Triton.

**¿Qué hace que un código sea "Triton-oriented"?**
Que tenga al menos una función decorada con `@triton.jit`. Adicionalmente, el código Triton-oriented típicamente:
- Usa `tl.program_id` para obtener el ID del hilo
- Usa `tl.arange` para vectorizar operaciones
- Usa `tl.load`/`tl.store` para leer/escribir memoria
- Tiene parámetros de tipo `tl.constexpr` para el tamaño del bloque

**¿Puede tu compilador rechazar código que no es Triton-oriented?**
No rechaza (no genera error de parseo), pero sí avisa: genera una advertencia `"Función sin @triton.jit"` y la marca como `[FUNC]` en vez de `[KERNEL]`. `is_valid = True` pero `warnings` contiene la advertencia.

**¿Puede detectar patrones incorrectos de Triton?**
Detecta funciones duplicadas (error) y funciones sin `@triton.jit` (advertencia). Hay un módulo separado, `src/kernelforge/benchmark/semantic_checker.py`, que usa AST de Python para detectar patrones más complejos: `tl.load` sin `mask` cuando hay offsets vectorizados, `BLOCK_SIZE` sin `tl.constexpr`, falta de `tl.program_id`.

**¿Cuáles son las limitaciones de tu detección Triton?**
- No verifica que `tl.load` tenga una máscara cuando es necesaria (eso está en el semantic_checker separado)
- No valida que `tl.program_id` esté presente en cada kernel
- No verifica la aridad de las llamadas `tl.*`
- No detecta accesos fuera de límites o carreras de datos

**¿Genera el translator código compatible con Triton?**
No. El output es un reporte de validación en texto, no código ejecutable. Para generar código Triton funcional hay un módulo separado (`scripts/generate_kernel.py`) que usa LLMs.

---

## 7. Generación de Código / Salida

**¿Puedes mostrar un ejemplo de entrada y salida?**
Entrada (`compiler/fixtures/valid/vector_add.py`):
```python
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    tl.store(output_ptr + offsets, x + y)
```

Salida del validator:
```
════════════════════════════════════════════════════════════
  REPORTE DE TRADUCCIÓN — Triton Kernel Validator
════════════════════════════════════════════════════════════
[KERNEL] vector_add_kernel  (línea 4)
  Parámetros : x_ptr, y_ptr, output_ptr, n: tl.constexpr [constexpr], BLOCK_SIZE: tl.constexpr [constexpr]
  Vars locales: pid, offsets, x, y
  Triton API  : tl.program_id, tl.arange, tl.load, tl.store
════════════════════════════════════════════════════════════
✓ Sin errores   │   Advertencias: 0
```

**¿El código generado es ejecutable?**
No, es solo un reporte de texto. La salida es descriptiva, no código fuente.

**¿Cómo validas que el output es correcto?**
Con 41 pruebas unitarias en `tests/compiler/test_triton_parser.py` que verifican el contenido de la tabla de símbolos, los flags detectados, y que los fixtures válidos/inválidos se clasifican correctamente.

**¿Qué limitaciones tiene la fase de generación?**
Solo genera texto descriptivo. No puede generar código GPU ejecutable desde la validación.

---

## 8. Manejo de Errores

**¿Qué tipos de errores puede detectar?**
- **Léxicos**: carácter ilegal, error de indentación inconsistente
- **Sintácticos**: token inesperado (falta `:`, `)`, palabra clave fuera de lugar)
- **Semánticos**: función declarada dos veces

**¿Qué tan claros son los mensajes de error?**
Todos incluyen número de línea. Ejemplos:
- `"Línea 3: token inesperado 'NAME' (valor: 'mi_var')"`
- `"Línea 5: función 'kernel' ya declarada"`
- `"Carácter ilegal '$' en línea 8"`

**¿Los mensajes incluyen número de línea?**
Sí. PLY con `tracking=True` propaga el número de línea a través de todas las producciones. `p.lineno(i)` da la línea del i-ésimo símbolo de la producción. El `IndentLexer` expone la propiedad `lineno` que delega al lexer raw interno.

**¿Qué pasa después del primer error?**
El parser continúa (modo de recuperación de errores de PLY). Los errores adicionales se acumulan en `_symtab.errors`. El reporte final muestra todos los errores encontrados.

**¿Tu compilador se detiene inmediatamente o intenta recuperarse?**
Intenta recuperarse — el parser PLY tiene soporte para eso con el token especial `error` en las reglas. En la práctica, la recuperación de PLY es limitada y para código muy malformado el reporte puede ser incompleto.

**¿Cuál fue el caso de error más difícil de manejar?**
Las líneas en blanco dentro de funciones. Sin el lookahead de NEWLINE, una línea en blanco (nivel de indentación 0) disparaba DEDENTs en medio de una función, haciendo que el parser creyera que el bloque había terminado cuando no era así.

---

## 10. Implementación y Demo

**¿Qué lenguaje de programación usaste?**
Python 3.13.

**¿Qué herramientas o librerías usaste?**
- **PLY 3.11** (Python Lex-Yacc): implementa los algoritmos de Lex y LALR(1) Yacc en Python
- **pytest**: para las 41 pruebas unitarias + de integración
- **dataclasses**: para `ValidationResult`

**¿Cómo está organizado el proyecto?**
```
compiler/
├── __init__.py
├── triton_lexer.py      # Análisis léxico (IndentLexer + _RawLexer)
├── triton_parser.py     # Gramática LALR(1) + DST + SymbolTable
├── triton_validator.py  # API pública: validate() / validate_file()
├── README.md            # Documentación detallada
└── fixtures/
    ├── valid/           # vector_add.py, softmax_kernel.py
    └── invalid/         # missing_jit.py, bad_syntax.py
tests/compiler/
└── test_triton_parser.py  # 41 pruebas
conftest.py              # Agrega raíz del proyecto al sys.path
```

**¿Cuál es el rol de cada archivo?**
- `triton_lexer.py`: convierte texto en tokens, maneja INDENT/DEDENT
- `triton_parser.py`: define la gramática BNF, la tabla de símbolos, y las acciones semánticas
- `triton_validator.py`: API de alto nivel que une lexer+parser y devuelve `ValidationResult`
- `conftest.py`: configuración de pytest para imports

**¿Cómo compilas y corres el proyecto?**
```bash
# Instalar dependencias
pip install ply pytest

# Validar un archivo
python -m compiler.triton_validator ruta/al/kernel.py

# Correr todas las pruebas
python -m pytest tests/compiler/ -v
```

**¿Cuál fue el mayor reto de implementación?**
El manejo de indentación con lookahead descrito antes, y hacer que el decorador `@triton.jit` (que se parsea como `expr` genérica) fuera reconocido correctamente. El parser retorna `("access", ("name","triton"), ("attr","jit"))` para `triton.jit`, y la función `_expr_to_str()` convierte eso al string `"triton.jit"` para compararlo.

**Durante la demo, ¿puedes mostrar el flujo completo?**
```
código fuente (str)
    │
    ▼ IndentLexer.input()
    │
    ├→ [token] token() → [NEWLINE, INDENT, DEF, NAME, ...]
    │
    ▼ parser.parse() — LALR(1), una reducción a la vez
    │
    ├→ cada reducción ejecuta acción semántica
    │       └→ _symtab.declare_function() / add_local_var() / add_triton_call()
    │
    ▼ _symtab.report()
    │
    ▼ ValidationResult(is_valid, report, errors, warnings, symbol_table)
```

---

## 11. Preguntas de Comprensión Profunda

**¿Por qué diseñaste la gramática de esta manera?**
La gramática sigue la estructura de Python por dos razones: (1) Triton usa Python como lenguaje anfitrión, entonces el compilador debe aceptar Python válido; (2) el curso requiere una gramática sin recursión izquierda, así que en vez de las reglas listas naturales (`expr → expr '+' term`) usamos el patrón `A→βA'` que es equivalente pero compatible con parsers top-down aunque usemos LALR.

**¿Qué pasaría si agregáramos bloques anidados más profundos?**
Nada, ya funciona. La regla `suite → NEWLINE INDENT stmt_seq DEDENT` es recursiva: un `stmt` puede ser un `funcdef` o un `if_stmt`, cada uno con su propio `suite`. La pila de indentación del lexer maneja cualquier nivel de profundidad.

**¿Qué pasaría si agregáramos llamadas a función anidadas?**
Ya están implementadas. La regla `call_expr → primary LPAREN arg_list RPAREN` y `primary` puede ser otra `call_expr` o un acceso de atributo, permitiendo `tl.load(ptr + tl.arange(0, N))`.

**¿Qué tan difícil sería agregar verificación de tipos?**
Muy difícil en PLY puro. Requeriría: (a) un sistema de tipos para Python (dinámico), (b) inferencia de tipos para variables locales, (c) compatibilidad de tipos en operaciones. Triton tiene tipos parcialmente anotados (`tl.constexpr`), pero el resto es dinámico. Sería más práctico integrar mypy o basedpyright.

**¿Qué parte del compilador depende más de la gramática?**
El parser completo. Si cambia la gramática (p.ej., agregamos un nuevo tipo de statement), hay que agregar reglas en `triton_parser.py` y las acciones semánticas correspondientes. El lexer y el validator son más estables.

**¿Qué parte se rompería si cambia la gramática?**
Las pruebas de parser y las acciones semánticas. Si una regla cambia de forma, su acción semántica (`p[1]`, `p[2]`, etc.) indexará posiciones incorrectas.

**¿Por qué la ambigüedad es un problema en el diseño de compiladores?**
Porque si una secuencia de tokens puede ser reconocida por dos árboles de derivación diferentes, el compilador podría generar dos traducciones distintas para el mismo programa. Ejemplo clásico: `a - b - c` podría ser `(a-b)-c` o `a-(b-c)` (distintos resultados). La ambigüedad hace que el compilador sea no determinista e incorrecto.

**¿Por qué la recursión izquierda es un problema para algunos parsers?**
Para parsers **top-down (LL)**, la recursión izquierda causa un bucle infinito: para expandir `A`, el parser intenta expandir `A` otra vez inmediatamente, sin consumir ningún token. Para parsers **bottom-up (LR/LALR)** como PLY, la recursión izquierda sí funciona correctamente. Sin embargo, el profesor exige gramáticas sin recursión izquierda como ejercicio de comprensión del algoritmo de eliminación.

**¿Cuál es la diferencia entre sintaxis y semántica?**
- **Sintaxis**: la *forma* del programa. ¿Las reglas gramaticales se cumplen? `def foo:` es un error sintáctico (faltan paréntesis).
- **Semántica**: el *significado* del programa. `def foo(): pass` y `def foo(): pass` dos veces es un error semántico (función duplicada) aunque cada declaración individualmente sea sintácticamente correcta.

**¿Cuál es la diferencia entre reconocer código y traducirlo?**
- **Reconocer** (parser puro): solo decide si el input pertenece al lenguaje definido por la gramática. Responde "sí/no".
- **Traducir** (DST/compiler): además de reconocer, produce una salida — en nuestro caso, la tabla de símbolos y el reporte. La traducción agrega semántica al reconocimiento.

---

## Resumen de Comandos para la Demo

```bash
# Validar el kernel de suma de vectores
python -m compiler.triton_validator compiler/fixtures/valid/vector_add.py

# Detectar error (función sin @triton.jit)
python -m compiler.triton_validator compiler/fixtures/invalid/missing_jit.py

# Detectar error de sintaxis
python -m compiler.triton_validator compiler/fixtures/invalid/bad_syntax.py

# Correr suite completa de pruebas
python -m pytest tests/compiler/ -v

# Usar la API programáticamente
python -c "
from compiler.triton_validator import validate
r = validate(open('compiler/fixtures/valid/vector_add.py').read())
print(r.report)
"
```
