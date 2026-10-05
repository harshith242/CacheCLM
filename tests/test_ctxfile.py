from cacheclm.ctxfile import append, blocks, drop_oldest, new_context, next_index, split_for_summary, split_pinned


def build(n):
    ctx = new_context("TASK")
    for i in range(n):
        ctx = append(ctx, "chunk", f"part {i}")
    return ctx


def test_blocks_are_numbered_in_order():
    ctx = build(3)
    assert [(b[2], b[3], b[4]) for b in blocks(ctx)] == [(0, "task", True), (1, "chunk", False), (2, "chunk", False),
                                                       (3, "chunk", False)]
    assert next_index(ctx) == 4


def test_append_after_an_edit_without_trailing_newline():
    ctx = build(1).rstrip("\n") + "\n[[CTX_TURN 7 role=notes]]\nmy note"
    ctx = append(ctx, "chunk", "next")
    assert [b[2] for b in blocks(ctx)] == [0, 1, 7, 8]


def test_drop_oldest_keeps_the_pinned_block():
    ctx = drop_oldest(build(2))
    assert "part 0" not in ctx and "part 1" in ctx and ctx.startswith(new_context("TASK"))
    assert drop_oldest(new_context("TASK")) == new_context("TASK")


def test_split_for_summary_keeps_the_last_chunks():
    head, middle, tail = split_for_summary(build(4), keep=2)
    assert head == new_context("TASK")
    assert "part 0" in middle and "part 1" in middle and "part 2" not in middle
    assert "part 2" in tail and "part 3" in tail
    assert split_for_summary(build(2), keep=2) is None


def test_only_the_blocks_after_the_task_are_editable():
    pinned, editable = split_pinned(build(2))
    assert pinned == new_context("TASK") and editable.startswith("[[CTX_TURN 1 role=chunk]]")
    assert pinned + editable == build(2)
