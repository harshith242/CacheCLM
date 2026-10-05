"""Adversarial cases: hostile model commands, degenerate budgets and inputs, broken provider replies."""
import signal
import time
from types import SimpleNamespace

import pytest

from cacheclm.arms import fit
from cacheclm.ctxfile import new_context, tokens
from cacheclm.gate import decide
from cacheclm.llm import LLM
from cacheclm.mab import Sample, chunk_text
from cacheclm.files import read_jsonl
from cacheclm.run import run_log, run_sample
from cacheclm.sandbox import run
from test_run import CFG, PRICE, FakeLLM, answers, events, sample


@pytest.fixture(autouse=True)
def no_hangs():
    """Fail a test that runs over 20 s instead of letting it hang the suite."""
    def stop(*_):
        raise TimeoutError("test hung")
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(20)
    yield
    signal.alarm(0)


def bash(script):
    return f"```bash\n{script}\n```"


# Sandbox: hostile file tricks

def test_ctx_replaced_by_a_symlink_to_an_endless_device_returns_quickly():
    start = time.monotonic()
    new, _ = run("python3 -c \"import os; os.remove('ctx.txt'); os.symlink('/dev/zero', 'ctx.txt')\"", "keep\n")
    assert time.monotonic() - start < 15 and len(new) < 1_000_000


def test_ctx_replaced_by_a_fifo_does_not_block():
    start = time.monotonic()
    run("python3 -c \"import os; os.remove('ctx.txt'); os.mkfifo('ctx.txt')\"", "keep\n")
    assert time.monotonic() - start < 15


def test_a_background_job_is_killed_with_the_command():
    start = time.monotonic()
    _, out = run("python3 -c 'import time; time.sleep(30)' &", "x\n", timeout=1)
    assert time.monotonic() - start < 10


def test_invalid_utf8_written_by_the_model_is_survivable():
    new, _ = run("python3 -c \"open('ctx.txt','wb').write(bytes([0xff, 0xfe, 0x41]))\"", "x\n")
    assert new.endswith("A")


def test_a_huge_output_is_capped():
    _, out = run("python3 -c \"print('y' * 5_000_000)\"", "x\n")
    assert len(out) <= 2000


# Context management under degenerate sizes

def test_a_giant_single_line_part_still_fits():
    task = new_context("TASK")
    body = "[[CTX_TURN 1 role=chunk]]\n" + "z" * 200_000 + "\n"
    out = fit(task, body, 10, 400, lambda r: None)
    assert tokens(task + out) + 10 <= 400


def test_a_budget_smaller_than_the_task_block_terminates():
    log = []
    out = fit(new_context("T" * 4000), "[[CTX_TURN 1 role=chunk]]\nabc\n", 50, 10, log.append)
    assert out == ""


def test_parts_larger_than_the_budget_do_not_loop(tmp_path):
    big = Sample("T/1", "factconsolidation_sh_test", "".join(f"{i}. fact {i}.\n" for i in range(400)),
                 ["q1"], [["fact 399"]])
    cfg = {**CFG, "context_budget": 150, "chunk_tokens": 300}  # each part alone is twice the budget
    run_sample(big, "clm", 0, FakeLLM(lambda t: answers(t) or "READY"), cfg, PRICE, tmp_path)
    assert any(r["event"] == "done" for r in read_jsonl(run_log(tmp_path, "clm", big, 0)))


def test_a_model_that_is_always_cut_off_still_finishes(tmp_path):
    llm = FakeLLM(lambda t: answers(t) or ("THOUGHT: " + "long " * 50, "length"))
    run_sample(sample(), "clm", 0, llm, CFG, PRICE, tmp_path)
    assert events(tmp_path, "clm", "done")


def test_a_model_that_inflates_ctx_is_cut_back_to_budget(tmp_path):
    inflate = bash("python3 -c \"open('ctx.txt','a').write('w' * 50000)\"")
    run_sample(sample(), "clm", 0, FakeLLM(lambda t: answers(t) or inflate), CFG, PRICE, tmp_path)
    assert all(s["ctx_tokens"] <= CFG["context_budget"] for s in events(tmp_path, "clm", "step"))


def test_a_model_that_moves_ctx_away_is_rolled_back(tmp_path):
    commands = iter(["READY", bash("mv ctx.txt elsewhere.txt")])
    run_sample(sample(), "clm", 0, FakeLLM(lambda t: answers(t) or next(commands, "READY")), CFG, PRICE, tmp_path)
    assert any(e["emptied"] and not e["allowed"] for e in events(tmp_path, "clm", "edit"))


def test_gate_with_no_turns_left_allows_only_free_edits():
    old = "A" * 4000 + "B" * 4000
    assert decide(old, "A" * 4000, 0, PRICE)[0]  # deleting the tail re-bills nothing: allowed
    assert not decide(old, "B" * 4000, 0, PRICE)[0]  # deleting the head re-bills the rest: rejected


def test_chunking_text_with_no_newlines():
    assert "".join(chunk_text("q" * 10_001, 1000)) == "q" * 10_001


# Provider: broken replies

class OddClient:
    """Replies with no choices (an error body sent with HTTP 200) on its first calls, then a normal reply."""

    def __init__(self, bad_calls):
        self.bad, self.calls = bad_calls, 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls += 1
        if self.calls <= self.bad:
            return SimpleNamespace(model="m", choices=[], usage=None)
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=1, prompt_cache_hit_tokens=0)
        return SimpleNamespace(model="m", usage=usage,
                               choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")])


def test_an_empty_provider_reply_is_retried(tmp_path, monkeypatch):
    monkeypatch.setattr("cacheclm.llm.time.sleep", lambda s: None)
    reply = LLM("m", OddClient(bad_calls=2), tmp_path, 0.3, PRICE).chat([{"role": "user", "content": "q"}], 8, 0)
    assert reply["content"] == "ok"


def test_a_provider_that_never_answers_fails_only_the_sample(tmp_path, monkeypatch):
    monkeypatch.setattr("cacheclm.llm.time.sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        LLM("m", OddClient(bad_calls=99), tmp_path, 0.3, PRICE).chat([{"role": "user", "content": "q"}], 8, 0)
