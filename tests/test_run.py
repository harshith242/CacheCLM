from cacheclm.arms import control, parse_command
from cacheclm.files import read_jsonl
from cacheclm.mab import Sample
from cacheclm.run import run_log, run_sample

PRICE = {"cache_read": 0.006, "cache_write": 0.30, "output": 1.20}
CFG = {"context_budget": 400, "chunk_tokens": 40, "max_edits_per_chunk": 3, "max_condense_tries": 3,
       "summary_at": 0.75, "keep_recent_chunks": 2, "nudges": [0.25, 0.5, 0.75], "query_workers": 2}
DROP_FIRST = ("```bash\npython3 -c \"t=open('ctx.txt').read(); b=t.split('[[CTX_TURN '); "
              "open('ctx.txt','w').write('[[CTX_TURN '.join(b[:2]+b[3:]))\"\n```")


def sample():
    facts = "".join(f"{i}. fact number {i}.\n" for i in range(60))
    return Sample("T/0", "factconsolidation_sh_test", facts, ["q1", "q2"], [["fact number 59"], ["nope"]])


class FakeLLM:
    def __init__(self, script):
        self.script, self.calls = script, 0

    def chat(self, messages, max_tokens, repeat):
        self.calls += 1
        text = messages[-1]["content"]
        return {"content": self.script(text), "model": "fake", "prompt_tokens": len(text) // 4,
                "cache_hit_tokens": 0, "completion_tokens": 10, "latency_s": 0.0, "cached": False}


def answers(text):
    return "fact number 59" if "Question:" in text else None


def events(tmp_path, arm, name):
    return [r for r in read_jsonl(run_log(tmp_path, arm, sample(), 0)) if r.get("event") == name]


def test_parse_command_and_control_note():
    assert parse_command("READY") is None
    assert parse_command("```bash\ncat ctx.txt\n```") == "cat ctx.txt"
    assert "OVER LIMIT" in control("x" * 760, 200, 40, [0.25], "")


def test_summary_arm_compacts_and_scores(tmp_path):
    llm = FakeLLM(lambda t: answers(t) or "SUMMARY OF EARLIER FACTS")
    acc = run_sample(sample(), "summary", 0, llm, CFG, PRICE, tmp_path)
    assert acc == 0.5
    assert events(tmp_path, "summary", "summary")
    assert all(s["ctx_tokens"] <= CFG["context_budget"] for s in events(tmp_path, "summary", "step"))


def test_clm_arm_makes_room_itself(tmp_path):
    llm = FakeLLM(lambda t: answers(t) or (DROP_FIRST if "OVER LIMIT" in t else "READY"))
    run_sample(sample(), "clm", 0, llm, CFG, PRICE, tmp_path)
    edits = events(tmp_path, "clm", "edit")
    assert edits and all(e["allowed"] for e in edits)
    assert not events(tmp_path, "clm", "forced_truncation")


def test_gate_rejects_costly_edits_but_allows_them_when_over_the_limit(tmp_path):
    # deletes the oldest part whenever a newer part sits after it: a mid-context edit
    llm = FakeLLM(lambda t: answers(t) or (DROP_FIRST if t.count("role=chunk") >= 2 else "READY"))
    run_sample(sample(), "gate", 0, llm, CFG, PRICE, tmp_path)
    edits = events(tmp_path, "gate", "edit")
    assert any(not e["allowed"] and "rejected" in e["reason"] for e in edits)
    assert any(e["allowed"] and "over the limit" in e["reason"] for e in edits)


def test_damaging_the_pinned_block_is_rolled_back(tmp_path):
    bad = "```bash\npython3 -c \"t=open('ctx.txt').read(); open('ctx.txt','w').write(t[10:])\"\n```"
    llm = FakeLLM(lambda t: answers(t) or bad)
    run_sample(sample(), "clm", 0, llm, CFG, PRICE, tmp_path)
    assert all(not e["allowed"] and "pinned" in e["reason"] for e in events(tmp_path, "clm", "edit"))


def test_finished_run_is_skipped(tmp_path):
    llm = FakeLLM(lambda t: answers(t) or "READY")
    first = run_sample(sample(), "clm", 0, llm, CFG, PRICE, tmp_path)
    calls = llm.calls
    assert run_sample(sample(), "clm", 0, llm, CFG, PRICE, tmp_path) == first and llm.calls == calls
