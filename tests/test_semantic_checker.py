import textwrap

from notebooks.benchmark.semantic_checker import check_kernel


def test_valid_kernel_no_warnings():
    src = textwrap.dedent("""
    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(in_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(in_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x, mask=mask)
    """)

    warnings = check_kernel(src)
    assert warnings == []


def test_load_missing_mask_detected():
    src = textwrap.dedent("""
    import triton
    import triton.language as tl

    @triton.jit
    def k(in_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(in_ptr + offsets)
        tl.store(in_ptr + offsets, x, mask=offsets < n)
    """)

    warnings = check_kernel(src)
    assert any("tl.load call missing mask" in w for w in warnings)


def test_scalar_load_without_mask_no_warning():
    src = textwrap.dedent("""
    import triton
    import triton.language as tl

    @triton.jit
    def k(freqency_penalty, cur_batch):
        # scalar load per-batch: should NOT warn
        cur_freqency = tl.load(freqency_penalty + cur_batch)
        return cur_freqency
    """)

    warnings = check_kernel(src)
    assert not any("tl.load call missing mask" in w or "vectorized load" in w for w in warnings)


def test_store_missing_mask_detected():
    src = textwrap.dedent("""
    import triton
    import triton.language as tl

    @triton.jit
    def k(in_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(in_ptr + offsets, mask=offsets < n)
        tl.store(out_ptr + offsets, x)
    """)

    warnings = check_kernel(src)
    assert any("tl.store call missing mask" in w for w in warnings)


def test_missing_decorator_detected():
    src = textwrap.dedent("""
    import triton
    import triton.language as tl

    def k(in_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(in_ptr + offsets, mask=offsets < n)
        tl.store(out_ptr + offsets, x, mask=offsets < n)
    """)

    warnings = check_kernel(src)
    assert any("missing @triton.jit" in w for w in warnings)


def test_block_size_without_constexpr_detected():
    src = textwrap.dedent("""
    import triton
    import triton.language as tl

    @triton.jit
    def k(in_ptr, out_ptr, n, BLOCK_SIZE):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(in_ptr + offsets, mask=offsets < n)
        tl.store(out_ptr + offsets, x, mask=offsets < n)
    """)

    warnings = check_kernel(src)
    assert any("BLOCK_SIZE used" in w for w in warnings)


def test_missing_program_id_detected():
    src = textwrap.dedent("""
    import triton
    import triton.language as tl

    @triton.jit
    def k(in_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        x = tl.load(in_ptr + offsets, mask=offsets < n)
        tl.store(out_ptr + offsets, x, mask=offsets < n)
    """)

    warnings = check_kernel(src)
    assert any("missing tl.program_id" in w for w in warnings)
