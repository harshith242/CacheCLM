"""The working context: a pinned task block (kept apart, never in ctx.txt) and an editable body of blocks in the
paper's [[CTX_TURN i role=...]] format. The model edits only the body."""
import re

CHARS_PER_TOKEN = 4  # local estimate; the gate's inequality is scale-free and billing uses provider counts
HEADER = re.compile(r"^\[\[CTX_TURN (\d+) role=([\w-]+)( pinned)?\]\]$", re.M)
PINNED = re.compile(r"^(\[\[CTX_TURN \d+ role=[\w-]+) pinned\]\]$", re.M)


def tokens(text):
    return len(text) // CHARS_PER_TOKEN


def block(i, role, text, pinned=False):
    return f"[[CTX_TURN {i} role={role}{' pinned' if pinned else ''}]]\n{text.rstrip()}\n"


def new_context(task):
    """The pinned task block; it is sent before the body in every request but never written to ctx.txt."""
    return block(0, "task", task, pinned=True)


def blocks(text):
    """[(start, end, index, role)] for each block, in order."""
    heads = list(HEADER.finditer(text))
    ends = [h.start() for h in heads[1:]] + [len(text)]
    return [(h.start(), end, int(h.group(1)), h.group(2)) for h, end in zip(heads, ends)]


def next_index(body):
    """Body blocks are numbered from 1; the task block is 0."""
    return max([b[2] for b in blocks(body)] + [0]) + 1


def append(body, role, text):
    sep = "" if not body or body.endswith("\n") else "\n"
    return body + sep + block(next_index(body), role, text)


def normalize(body, role="notes"):
    """The body without model-written 'pinned' marks, starting with a header (loose leading text gets one)."""
    body = PINNED.sub(r"\1]]", body).lstrip("\n")
    if body and not HEADER.match(body):
        body = f"[[CTX_TURN {next_index(body)} role={role}]]\n" + body
    return body


def cut_oldest_lines(body, chars):
    """The body without its oldest whole lines (at least `chars` characters of them), re-headed as a chunk."""
    lines = body.splitlines(keepends=True)
    cut = 0
    while lines and cut < chars:
        line = lines.pop(0)
        cut += 0 if HEADER.match(line) else len(line)  # a header is re-added below, so it frees nothing
    return normalize("".join(lines), "chunk")


def split_for_summary(body, keep):
    """(older, tail): everything before the last `keep` chunk blocks, and those blocks; None if nothing is older."""
    chunks = [b for b in blocks(body) if b[3] == "chunk"]
    if len(chunks) <= keep or not body[:chunks[-keep][0]].strip():
        return None
    start = chunks[-keep][0]
    return body[:start], body[start:]
