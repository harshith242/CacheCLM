import threading
import time

import pytest

from cacheclm.arms import ARMS, SKILL, control, parse_command, system
from cacheclm.budget import BudgetExceeded
from cacheclm.files import read_jsonl
from cacheclm.mab import Sample
from cacheclm.run import Recorder, answer_all, run_log, run_sample

PRICE = {"cache_read": 0.006, "cache_write": 0.30, "output": 1.20}
CFG = {"context_budget": 400, "chunk_tokens": 40, "max_edits_per_chunk": 3, "max_condense_tries": 3,
       "summary_at": 0.75, "keep_recent_chunks": 2, "summary_words": 60, "summary_max_tokens": 100, "edit_max_tokens": 200,
       "nudges": [0.25, 0.5, 0.75], "query_workers": 2}
DROP_FIRST = ("```bash\npython3 -c \"t=open('ctx.txt').read(); b=t.split('[[CTX_TURN '); "
              "open('ctx.txt','w').write('[[CTX_TURN '.join(b[:1]+b[2:]))\"\n```")  # ctx.txt starts at the oldest part


def sample():
    facts = "".join(f"{i}. fact number {i}.\n" for i in range(60))
    return Sample("T/0", "factconsolidation_sh_test", facts, ["q1", "q2"], [["fact number 59"], ["nope"]])


class FakeLLM:
    def __init__(self, script):
        self.script, self.calls = script, 0

    def chat(self, messages, max_tokens, repeat):
        self.calls += 1
        text = messages[-1]["content"]
        out = self.script(text)
        content, finish = out if isinstance(out, tuple) else (out, "stop")
        return {"content": content, "model": "fake", "prompt_tokens": len(text) // 4, "cache_hit_tokens": 0,
                "completion_tokens": 10, "latency_s": 0.0, "cached": False, "finish_reason": finish}


def answers(text):
    return "fact number 59" if "Question:" in text else None


def events(tmp_path, arm, name):
    return [r for r in read_jsonl(run_log(tmp_path, arm, sample(), 0)) if r.get("event") == name]


def test_parse_command_and_control_note():
    assert parse_command("READY") is None
    assert parse_command("```bash\ncat ctx.txt\n```") == "cat ctx.txt"
    assert parse_command("```text\nnote\n```\n```bash\ncat new.txt\n```") == "cat new.txt"
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


KEEP_FACT_LINES = "```bash\ngrep -E '^\\[\\[|^[0-9]+\\.' ctx.txt > t && mv t ctx.txt\n```"  # drops every other line


def test_a_line_filter_cannot_damage_the_task(tmp_path):
    prompts = []

    class Spy(FakeLLM):
        def chat(self, messages, max_tokens, repeat):
            prompts.append(messages[-1]["content"])
            return super().chat(messages, max_tokens, repeat)

    llm = Spy(lambda t: answers(t) or (KEEP_FACT_LINES if "OVER LIMIT" in t else "READY"))
    run_sample(sample(), "clm", 0, llm, CFG, PRICE, tmp_path)
    edits = [e for e in events(tmp_path, "clm", "edit") if e["changed"]]
    assert edits and all(e["allowed"] for e in edits)
    assert all("When facts conflict" in p for p in prompts)  # the task text reaches every call


def test_finished_run_is_skipped(tmp_path):
    llm = FakeLLM(lambda t: answers(t) or "READY")
    first = run_sample(sample(), "clm", 0, llm, CFG, PRICE, tmp_path)
    calls = llm.calls
    assert run_sample(sample(), "clm", 0, llm, CFG, PRICE, tmp_path) == first and llm.calls == calls


def test_system_prompt_is_salted_per_repeat_and_shared_by_clm_and_gate():
    assert system("clm", 100, 1).startswith("Run r1.")
    assert system("clm", 100, 0) != system("clm", 100, 1)
    assert system("clm", 100, 1) == system("gate", 100, 1)


def test_repeat_salt_reaches_every_call(tmp_path):
    seen = []

    class Spy(FakeLLM):
        def chat(self, messages, max_tokens, repeat):
            seen.append(messages[0]["content"])
            return super().chat(messages, max_tokens, repeat)

    run_sample(sample(), "gate", 1, Spy(lambda t: answers(t) or "READY"), CFG, PRICE, tmp_path)
    assert seen and all(s.startswith("Run r1.") for s in seen)


class TokenLLM:
    """Bills 1 prompt token per 3 characters, so a chars/4 estimate would be off."""

    def chat(self, messages, max_tokens, repeat):
        text = "".join(m["content"] for m in messages)
        return {"content": "ok", "model": "m", "prompt_tokens": len(text) // 3, "cache_hit_tokens": 0,
                "completion_tokens": 1, "latency_s": 0.0, "cached": False}


def test_ideal_hit_is_scaled_to_billed_tokens(tmp_path):
    rec = Recorder(tmp_path / "log.jsonl", TokenLLM(), 0, {})
    rec.chat([{"role": "user", "content": "a" * 600}], 8, "edit")
    rec.chat([{"role": "user", "content": "a" * 600 + "b" * 300}], 8, "edit")
    calls = read_jsonl(tmp_path / "log.jsonl")
    assert calls[0]["ideal_hit_tokens"] == 0
    assert calls[1]["ideal_hit_tokens"] == 200  # 600 of 900 characters shared, 300 billed tokens


def test_queries_wait_for_a_cache_hit_and_the_first_one_is_compared_with_the_real_previous_request(tmp_path):
    lock, state, log = threading.Lock(), {"inflight": 0, "n": 0}, []

    class SlowLLM:
        def chat(self, messages, max_tokens, repeat):
            with lock:
                state["inflight"] += 1
                state["n"] += 1
                n, alone = state["n"], state["inflight"] == 1
            time.sleep(0.05)
            with lock:
                log.append(alone)
                state["inflight"] -= 1
            text = "".join(m["content"] for m in messages)
            return {"content": "x", "model": "m", "prompt_tokens": len(text) // 4,
                    "cache_hit_tokens": 100 if n >= 2 else 0, "completion_tokens": 1, "latency_s": 0.0,
                    "cached": False}

    rec = Recorder(tmp_path / "log.jsonl", SlowLLM(), 0, {})
    rec.ref = "an unrelated earlier request"
    answer_all("CTX " * 50, "summary", "factconsolidation", [f"q{i}" for i in range(6)], rec,
               {**CFG, "query_workers": 4})
    assert log[:2] == [True, True] and not all(log)  # serial until the cache hits, then parallel
    calls = read_jsonl(tmp_path / "log.jsonl")
    assert calls[0]["ideal_hit_tokens"] == 0 and calls[1]["ideal_hit_tokens"] > 0


def test_read_only_and_refused_commands_are_not_changes(tmp_path):
    commands = iter(["```bash\nwc -c ctx.txt\n```", "```bash\ncurl example.com\n```"])
    llm = FakeLLM(lambda t: answers(t) or next(commands, "READY"))
    run_sample(sample(), "clm", 0, llm, CFG, PRICE, tmp_path)
    edits = events(tmp_path, "clm", "edit")
    assert [e["changed"] for e in edits] == [False, False]
    assert [e["refused"] for e in edits] == [False, True]


def test_a_model_that_never_makes_room_is_truncated_to_fit(tmp_path):
    llm = FakeLLM(lambda t: answers(t) or "READY")
    run_sample(sample(), "clm", 0, llm, CFG, PRICE, tmp_path)
    assert events(tmp_path, "clm", "forced_truncation")
    assert all(s["ctx_tokens"] <= CFG["context_budget"] for s in events(tmp_path, "clm", "step"))


def test_an_interrupted_run_restarts_cleanly(tmp_path):
    path = run_log(tmp_path, "clm", sample(), 0)
    path.parent.mkdir(parents=True)
    path.write_text('{"event": "call", "phase": "edit", "stale": true}\n')  # killed mid-run, no done record
    run_sample(sample(), "clm", 0, FakeLLM(lambda t: answers(t) or "READY"), CFG, PRICE, tmp_path)
    records = read_jsonl(path)
    assert not any(r.get("stale") for r in records)
    assert [r["event"] for r in records].count("done") == 1


def test_a_budget_stop_leaves_the_run_resumable(tmp_path):
    class Broke(FakeLLM):
        def chat(self, messages, max_tokens, repeat):
            if self.calls == 5:
                raise BudgetExceeded("spent $5.01 of $5.00")
            return super().chat(messages, max_tokens, repeat)

    with pytest.raises(BudgetExceeded):
        run_sample(sample(), "clm", 0, Broke(lambda t: answers(t) or "READY"), CFG, PRICE, tmp_path)
    assert not events(tmp_path, "clm", "done")  # so the next run redoes this sample instead of skipping it


def test_questions_tell_the_model_that_editing_is_over(tmp_path):
    asked = []

    class Spy(FakeLLM):
        def chat(self, messages, max_tokens, repeat):
            if "Question:" in messages[-1]["content"]:
                asked.append(messages[-1]["content"])
            return super().chat(messages, max_tokens, repeat)

    run_sample(sample(), "clm", 0, Spy(lambda t: answers(t) or "READY"), CFG, PRICE, tmp_path)
    assert asked and all("Editing is over" in q for q in asked)


def test_summary_asks_for_a_length_it_can_finish(tmp_path):
    seen = []

    class Spy(FakeLLM):
        def chat(self, messages, max_tokens, repeat):
            if "Write one summary" in messages[-1]["content"]:
                seen.append((max_tokens, messages[-1]["content"]))
            return super().chat(messages, max_tokens, repeat)

    run_sample(sample(), "summary", 0, Spy(lambda t: answers(t) or "SUMMARY"), CFG, PRICE, tmp_path)
    assert seen and all(m == 100 and "at most 60 words" in text for m, text in seen)


def done(tmp_path, arm):
    return events(tmp_path, arm, "done")[-1]


def test_keeping_the_last_lines_keeps_the_newest_facts_within_budget(tmp_path):
    keep_tail = "```bash\ntail -n 12 ctx.txt > t && mv t ctx.txt\n```"  # leaves headless text at the top
    llm = FakeLLM(lambda t: answers(t) or (keep_tail if "OVER LIMIT" in t else "READY"))
    run_sample(sample(), "clm", 0, llm, CFG, PRICE, tmp_path)
    assert all(s["ctx_tokens"] <= CFG["context_budget"] for s in events(tmp_path, "clm", "step"))
    final = done(tmp_path, "clm")["final_context"]
    assert "59. fact number 59." in final
    assert final.split("]]\n", 1)[1].split("\n[[CTX_TURN", 1)[0].count("fact number") == 0  # task block holds no facts


def test_model_written_pinned_header_is_stripped(tmp_path):
    note = "```bash\nprintf '[[CTX_TURN 99 role=notes pinned]]\\nmy note\\n' >> ctx.txt\n```"
    commands = iter([note])
    llm = FakeLLM(lambda t: answers(t) or next(commands, "READY"))
    run_sample(sample(), "clm", 0, llm, CFG, PRICE, tmp_path)
    final = done(tmp_path, "clm")["final_context"]
    assert final.count(" pinned]]") == 1  # only the task block


def test_an_edit_that_empties_ctx_is_rolled_back(tmp_path):
    wipe = "```bash\npython3 -c \"open('ctx.txt', 'w')\"\n```"
    commands = iter(["READY", wipe])  # wipe once the body has a part in it
    llm = FakeLLM(lambda t: answers(t) or next(commands, "READY"))
    run_sample(sample(), "clm", 0, llm, CFG, PRICE, tmp_path)
    wiped = [e for e in events(tmp_path, "clm", "edit") if e["emptied"]]
    assert wiped and not wiped[0]["allowed"]


def test_a_cut_off_reply_ends_the_edit_phase_without_a_retry(tmp_path):
    # a retry was told "send a shorter one" and repeated the same whole rewrite, three times per step
    replies = iter([("THOUGHT: keep the newest.\n```bash\npython3 -c \"print(1)", "length")])
    llm = FakeLLM(lambda t: answers(t) or next(replies, "READY"))
    run_sample(sample(), "clm", 0, llm, CFG, PRICE, tmp_path)
    records = [r["event"] for r in read_jsonl(run_log(tmp_path, "clm", sample(), 0))]
    after = records[records.index("cut_off") + 1:]
    assert next(e for e in after if e in ("call", "step")) == "step"  # no retry in the same phase


def test_gate_explains_itself_only_on_rejection(tmp_path):
    prompts = []

    class Spy(FakeLLM):
        def chat(self, messages, max_tokens, repeat):
            prompts.append(messages[-1]["content"])
            return super().chat(messages, max_tokens, repeat)

    llm = Spy(lambda t: answers(t) or (DROP_FIRST if t.count("role=chunk") >= 2 else "READY"))
    run_sample(sample(), "gate", 0, llm, CFG, PRICE, tmp_path)
    assert any("edit rejected" in p for p in prompts)
    assert not any("allowed" in p.split("Result of your last command:")[-1] for p in prompts if "Result of" in p)


def test_skill_arm_is_clm_plus_the_skill_and_is_not_gated(tmp_path):
    assert system("skill", 100, 0) == system("clm", 100, 0) + SKILL
    assert ARMS == ("summary", "clm", "gate", "skill")
    llm = FakeLLM(lambda t: answers(t) or (DROP_FIRST if t.count("role=chunk") >= 2 else "READY"))
    run_sample(sample(), "skill", 0, llm, CFG, PRICE, tmp_path)
    edits = [e for e in events(tmp_path, "skill", "edit") if e["changed"]]
    assert edits and all(e["allowed"] for e in edits)


def test_a_cut_off_summary_is_logged(tmp_path):
    llm = FakeLLM(lambda t: answers(t) or ("SUMMARY", "length"))
    run_sample(sample(), "summary", 0, llm, CFG, PRICE, tmp_path)
    assert all(s["cut_off"] for s in events(tmp_path, "summary", "summary"))


def test_a_headers_only_file_counts_as_emptied(tmp_path):
    headers_only = "```bash\ngrep '^\\[\\[' ctx.txt > t && mv t ctx.txt\n```"
    commands = iter(["READY", headers_only])
    run_sample(sample(), "clm", 0, FakeLLM(lambda t: answers(t) or next(commands, "READY")), CFG, PRICE, tmp_path)
    wiped = [e for e in events(tmp_path, "clm", "edit") if e["emptied"]]
    assert wiped and not wiped[0]["allowed"]


def test_fact_presence_is_only_checked_for_single_hop_facts(tmp_path):
    mh = sample()
    mh.source, mh.questions = "factconsolidation_mh_test", ["Which fact number is 59?", "q2"]  # matchable on sh
    run_sample(mh, "clm", 0, FakeLLM(lambda t: answers(t) or "READY"), CFG, PRICE, tmp_path)
    answers_logged = [r for r in read_jsonl(run_log(tmp_path, "clm", mh, 0)) if r.get("event") == "answer"]
    assert answers_logged and all(r["gold_fact_present"] is None for r in answers_logged)


def test_a_text_block_reaches_the_command_as_new_txt(tmp_path):
    reply = ("THOUGHT: log the newest part.\n```text\nDebbie said \"no\" and wore a green dress.\n```\n"
             "```bash\npython3 -c \"t=open('ctx.txt').read(); i=t.rindex('[[CTX_TURN'); "
             "h=t[i:].split(chr(10))[0]; open('ctx.txt','w').write(t[:i]+h+chr(10)+open('new.txt').read())\"\n```")
    once = iter([reply])  # in the final edit phase, so no later truncation can cut the new text
    script = lambda t: answers(t) or (next(once, "READY") if "Next part: 0 tokens" in t else "READY")  # noqa: E731
    run_sample(sample(), "skill", 0, FakeLLM(script), CFG, PRICE, tmp_path)
    assert 'Debbie said "no" and wore a green dress.' in done(tmp_path, "skill")["final_context"]


def test_the_prompt_states_the_reply_cap_in_words_that_match_the_config():
    import re as _re
    import yaml
    cap = yaml.safe_load(open("configs/base.yaml"))["edit_max_tokens"]
    words = int(_re.search(r"cut off after about ([\d,]+) words", system("clm", 100)).group(1).replace(",", ""))
    assert abs(words - cap * 0.75) <= cap * 0.1  # about 0.75 words per token, so the stated limit is real
