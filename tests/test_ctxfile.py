from cacheclm.ctxfile import append, blocks, cut_oldest_lines, next_index, normalize, split_for_summary


def build(n):
    body = ""
    for i in range(n):
        body = append(body, "chunk", f"part {i}")
    return body


def test_body_blocks_are_numbered_from_one():
    assert next_index("") == 1
    assert [(b[2], b[3]) for b in blocks(build(3))] == [(1, "chunk"), (2, "chunk"), (3, "chunk")]


def test_append_after_an_edit_without_trailing_newline():
    body = build(1) + "[[CTX_TURN 7 role=notes]]\nmy note"
    assert [b[2] for b in blocks(append(body, "chunk", "next"))] == [1, 7, 8]


def test_normalize_heads_loose_text_and_strips_model_pinned_marks():
    assert normalize("\n\n12. a fact\n").startswith("[[CTX_TURN 1 role=notes]]\n12. a fact")
    assert normalize("[[CTX_TURN 4 role=notes pinned]]\nx\n") == "[[CTX_TURN 4 role=notes]]\nx\n"
    assert normalize("") == "" and normalize(build(2)) == build(2)


def test_cut_oldest_lines_keeps_the_newest_lines_under_a_header():
    body = "".join(f"{i}. fact {i}\n" for i in range(10))
    cut = cut_oldest_lines(body, 30)
    assert cut.startswith("[[CTX_TURN 1 role=chunk]]\n") and cut.endswith("9. fact 9\n")
    assert "0. fact 0" not in cut and len(cut) < len(body)


def test_split_for_summary_keeps_the_last_chunks():
    older, tail = split_for_summary(build(4), keep=2)
    assert "part 0" in older and "part 1" in older and "part 2" not in older
    assert tail.startswith("[[CTX_TURN 3 role=chunk]]") and "part 3" in tail
    assert split_for_summary(build(2), keep=2) is None


def test_cutting_always_removes_content_even_when_the_header_is_first():
    body = "[[CTX_TURN 3 role=chunk]]\nshort line\nnewest line\n"
    cut = cut_oldest_lines(body, 5)
    assert len(cut) < len(body) and "short line" not in cut and cut.endswith("newest line\n")
