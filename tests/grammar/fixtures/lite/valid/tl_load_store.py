offsets = tl.arange(0, BLOCK_SIZE)
x = tl.load(ptr + offsets, mask=offsets < n, other=0.0)
tl.store(out + offsets, x, mask=offsets < n)
