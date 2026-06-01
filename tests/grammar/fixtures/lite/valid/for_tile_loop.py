for offset in range(0, n, BLOCK_SIZE):
    x = tl.load(ptr + offset)
    tl.store(out + offset, x)
