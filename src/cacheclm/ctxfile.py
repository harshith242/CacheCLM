"""The working context as one text: a pinned task block, then blocks in the paper's [[CTX_TURN i role=...]] format."""
import re

CHARS_PER_TOKEN = 4  # local estimate; the gate's inequality is scale-free and billing uses provider counts
HEADER = re.compile(r"^\[\[CTX_TURN (\d+) role=([\w-]+)( pinned)?\]\]$", re.M)


def tokens(text):
    return len(text) // CHARS_PER_TOKEN


def block(i, role, text, pinned=False):
    return f"[[CTX_TURN {i} role={role}{' pinned' if pinned else ''}]]\n{text.rstrip()}\n"


def new_context(task):
    return block(0, "task", task, pinned=True)


def blocks(ctx):
    """[(start, end, index, role, pinned)] for each block, in file order."""
    heads = list(HEADER.finditer(ctx))
    ends = [h.start() for h in heads[1:]] + [len(ctx)]
    return [(h.start(), end, int(h.group(1)), h.group(2), bool(h.group(3))) for h, end in zip(heads, ends)]


def next_index(ctx):
    return max((b[2] for b in blocks(ctx)), default=-1) + 1


def append(ctx, role, text):
    """ctx with a new block at the end, numbered after the highest index."""
    sep = "" if ctx.endswith("\n") else "\n"
    return ctx + sep + block(next_index(ctx), role, text)


def drop_oldest(ctx):
    """Remove the first unpinned block (forced truncation); unchanged if there is none."""
    for start, end, _, _, pinned in blocks(ctx):
        if not pinned:
            return ctx[:start] + ctx[end:]
    return ctx


def split_for_summary(ctx, keep):
    """(head, middle, tail): the pinned block, the blocks to compact, the last `keep` chunk blocks; None if nothing to compact."""
    bs = blocks(ctx)
    head_end = bs[0][1]
    chunks = [b for b in bs if b[3] == "chunk"]
    if len(chunks) <= keep or chunks[-keep][0] <= head_end:
        return None
    tail_start = chunks[-keep][0]
    return ctx[:head_end], ctx[head_end:tail_start], ctx[tail_start:]


def pinned_intact(old, new):
    """True when new still starts with old's pinned first block, unchanged."""
    start, end = blocks(old)[0][:2]
    return new.startswith(old[start:end])
