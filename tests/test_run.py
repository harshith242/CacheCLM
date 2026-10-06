import threading
import time

import pytest

from cacheclm.arms import ARMS, REFERENCES, SKILL, control, parse_reply, system
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


def test_parse_reply_and_control_note():
    assert parse_reply("READY") == (None, {})
    assert parse_reply("```bash\ncat ctx.txt\n```") == ("cat ctx.txt", {})
    assert parse_reply("```text\nnote\n```\n```bash\ncat new.txt\n```") == ("cat new.txt", {"new.txt": "note\n"})
    assert "OVER LIMIT" in control("x" * 760, 200, 40, [0.25], "")


def test_a_command_that_only_says_ready_is_ready():
    for reply in ('```bash\necho "READY"\n```', "```bash\necho READY\n```", "```sh\nprintf 'READY'\n```"):
        assert parse_reply(reply)[0] is None  # Qwen often wraps READY in a command; it must not use up an edit
    assert parse_reply('```bash\necho "READY" >> ctx.txt\n```')[0] == 'echo "READY" >> ctx.txt'  # a real edit stays


def test_a_python_block_runs_as_a_script_and_prose_between_blocks_is_never_a_command():
    code = "open('ctx.txt', 'w').write('x')\n"
    assert parse_reply(f"THOUGHT\n```python\n{code}```") == ("python3 edit.py", {"edit.py": code})
    assert parse_reply("```python\nprint(1)\n```\nThen:\n```bash\nsed -n 1p ctx.txt\n```")[0] == "python3 edit.py"
    assert parse_reply("```json\n{}\n```\nnot a command\n")[0] is None  # an unknown block runs nothing


def test_an_edit_that_changes_nothing_is_told_so_and_never_repeats_the_same_prompt(tmp_path):
    prompts = []
    writes_new_txt = "```python\nopen('new.txt', 'w').write('kept facts')\n```"  # Qwen's mistake: ctx.txt untouched

    class Spy(FakeLLM):
        def chat(self, messages, max_tokens, repeat):
            prompts.append(messages[-1]["content"])
            return super().chat(messages, max_tokens, repeat)
    run_sample(sample(), "clm", 0, Spy(lambda t: answers(t) or writes_new_txt), CFG, PRICE, tmp_path)
    edit_prompts = [p for p in prompts if "Question:" not in p]
    assert any("ctx.txt did not change" in p for p in edit_prompts)
    assert len(set(edit_prompts)) == len(edit_prompts)  # a repeated prompt would replay the same reply from the cache


def test_summary_arm_compacts_and_scores(tmp_path):
    llm = FakeLLM(lambda t: answers(t) or "SUMMARY OF EARLIER FACTS")
    acc = run_sample(sample(), "summary", 0, llm, CFG, PRICE, tmp_path)
    assert acc == 0.5
    assert events(tmp_path, "summary", "summary")
    assert all(s["ctx_tokens"] <= CFG["context_budget"] for s in events(tmp_path, "summary", "step"))


def test_the_summary_prompt_states_its_reply_cap_like_the_edit_prompt(tmp_path):
    prompts = []
    llm = FakeLLM(lambda t: prompts.append(t) or answers(t) or "SUMMARY")
    run_sample(sample(), "summary", 0, llm, {**CFG, "summary_max_tokens": 1000}, PRICE, tmp_path)
    asks = [p for p in prompts if "Write one summary" in p]
    assert asks and all("cut off after about 750 words" in p for p in asks)  # every fact summary hit the cap unwarned


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
    assert any(not e["allowed"] and "rejected" in e["reason"] and e["outcome"] == "rejected_price" for e in edits)
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
    assert [e["outcome"] for e in edits] == ["no_change", "refused"]


def test_an_edit_arm_that_cannot_make_room_stops_reading_and_still_answers(tmp_path):
    llm = FakeLLM(lambda t: answers(t) or "READY")
    run_sample(sample(), "clm", 0, llm, CFG, PRICE, tmp_path)
    stop = events(tmp_path, "clm", "overflow_stop")
    assert len(stop) == 1 and stop[0]["unread_parts"] > 0 and stop[0]["policy"] == "CacheCLM stop-on-overflow"
    assert not events(tmp_path, "clm", "forced_truncation")  # no harness cut hides the failure
    over = [r for r in events(tmp_path, "clm", "ready") if r["over"]]
    assert len(over) == CFG["max_edits_per_chunk"] + CFG["max_condense_tries"]  # READY while over uses up a try
    assert events(tmp_path, "clm", "done")[0]["n_questions"] == 2  # the questions are still answered


def test_the_summary_arm_keeps_the_forced_cut_as_its_fallback(tmp_path):
    run_sample(sample(), "summary", 0, FakeLLM(lambda t: answers(t) or "S " * 400), CFG, PRICE, tmp_path)
    assert not events(tmp_path, "summary", "overflow_stop")
    assert all(s["ctx_tokens"] <= CFG["context_budget"] for s in events(tmp_path, "summary", "step"))


def test_every_arm_answers_under_the_same_query_prompt(tmp_path):
    systems = set()

    class Spy(FakeLLM):
        def chat(self, messages, max_tokens, repeat):
            if "Question:" in messages[-1]["content"]:
                systems.add(messages[0]["content"])
            return super().chat(messages, max_tokens, repeat)
    for arm in ARMS + REFERENCES:
        run_sample(sample(), arm, 0, Spy(lambda t: answers(t) or "READY"), CFG, PRICE, tmp_path)
    assert len(systems) == 1  # the skill's recipe or the editing protocol must not reach the answers


def test_an_edit_that_grows_past_the_limit_is_rolled_back_in_every_edit_arm(tmp_path):
    inflate = "```bash\npython3 -c \"open('ctx.txt','a').write('w' * 5000)\"\n```"
    for arm in ("clm", "gate", "skill"):
        run_sample(sample(), arm, 0, FakeLLM(lambda t: answers(t) or inflate), CFG, PRICE, tmp_path)
        edits = events(tmp_path, arm, "edit")
        assert edits and all(e["outcome"] == "rejected_growth" and not e["allowed"] for e in edits)
    note = "```bash\necho '[[CTX_TURN 99 role=notes]]' >> ctx.txt\n```"
    commands = iter([note])
    run_sample(sample(), "clm", 1, FakeLLM(lambda t: answers(t) or next(commands, "READY")), CFG, PRICE, tmp_path)
    assert [e["outcome"] for e in read_jsonl(run_log(tmp_path, "clm", sample(), 1)) if e.get("event") == "edit"] \
        == ["applied"]  # growth that still fits is kept, as in the paper's default rule


def test_the_prompt_asks_to_locate_text_with_code_instead_of_retyping_it():
    assert "never paste or retype text you keep" in system("clm", 100)


def test_nudges_are_informational_at_25_percent_and_warn_against_wiping_later():
    quarter = control("x" * 1040, 1000, 10, [0.25], "")
    assert "about 25% of your budget" in quarter and "do not delete" not in quarter
    half = control("x" * 2040, 1000, 10, [0.25, 0.5], "")
    assert "over 50% full" in half and "do not delete whole regions" in half


def test_a_nudge_fires_again_after_the_model_compacts_below_it(tmp_path):
    prompts = []

    class Spy(FakeLLM):
        def chat(self, messages, max_tokens, repeat):
            prompts.append(messages[-1]["content"])
            return super().chat(messages, max_tokens, repeat)
    run_sample(sample(), "clm", 0, Spy(lambda t: answers(t) or (DROP_FIRST if "over 50% full" in t else "READY")),
               CFG, PRICE, tmp_path)
    assert sum("over 50% full" in p for p in prompts) >= 2  # re-armed once the context dropped below 50%


def test_every_edit_attempt_has_one_outcome(tmp_path):
    commands = iter(["```bash\nwc -c ctx.txt\n```", "```bash\ncurl example.com\n```"])
    run_sample(sample(), "clm", 0, FakeLLM(lambda t: answers(t) or next(commands, "READY")), CFG, PRICE, tmp_path)
    assert [e["outcome"] for e in events(tmp_path, "clm", "edit")] == ["no_change", "refused"]
    assert events(tmp_path, "clm", "ready")  # READY is logged too


def test_the_done_record_fingerprints_the_stream(tmp_path):
    for repeat, reply in ((0, "READY"), (1, DROP_FIRST)):
        run_sample(sample(), "clm", repeat, FakeLLM(lambda t: answers(t) or reply), CFG, PRICE, tmp_path)
    hashes = [[r for r in read_jsonl(run_log(tmp_path, "clm", sample(), k)) if r.get("event") == "done"][0]
              ["stream_hash"] for k in (0, 1)]
    assert all(len(h) == 40 for h in hashes) and hashes[0] != hashes[1]


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
    assert system("skill", 100, 0) == system("clm", 100, 0) + SKILL["head"] + SKILL["eventqa"] + SKILL["factconsolidation"]
    facts = system("skill", 100, 0, 16384, "factconsolidation")
    assert "Numbered fact lists" in facts and "Book excerpts" not in facts  # the skill arm saw the book recipe on facts
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
    script = lambda t: answers(t) or (next(once, "READY") if "Next part: 0 tokens" in t  # noqa: E731
                                      else DROP_FIRST if "OVER LIMIT" in t else "READY")  # makes room, so it reads on
    run_sample(sample(), "skill", 0, FakeLLM(script), CFG, PRICE, tmp_path)
    assert 'Debbie said "no" and wore a green dress.' in done(tmp_path, "skill")["final_context"]


def test_the_prompt_states_the_reply_cap_in_words_that_match_the_config():
    import re as _re
    import yaml
    for config in ("configs/base.yaml", "configs/local.yaml"):
        cap = yaml.safe_load(open(config))["edit_max_tokens"]
        prompt = system("clm", 100, 0, cap)
        words = int(_re.search(r"cut off after about ([\d,]+) words", prompt).group(1).replace(",", ""))
        assert cap * 0.65 <= words <= cap * 0.75  # about 0.75 words per token, rounded down: the stated limit is real
    assert "about 12,000 words" in system("clm", 100, 0, 16384)  # unchanged text, so cached DeepSeek calls still replay


def test_each_budget_nudge_is_shown_once_when_crossed(tmp_path):
    prompts = []

    class Spy(FakeLLM):
        def chat(self, messages, max_tokens, repeat):
            prompts.append(messages[-1]["content"])
            return super().chat(messages, max_tokens, repeat)

    run_sample(sample(), "clm", 0, Spy(lambda t: answers(t) or "READY"), CFG, PRICE, tmp_path)
    for phrase in ("about 25% of your budget", "over 50% full", "over 75% full"):
        assert sum(phrase in p for p in prompts) == 1  # once per crossing
