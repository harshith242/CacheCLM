from cacheclm.__main__ import jobs, load, make_llm, run_config
from cacheclm.budget import Budget
from cacheclm.arms import ARMS, REFERENCES

CFG = {"repeats": [0], "samples": [{"split": "A", "row": 1}, {"split": "B", "row": 2}],
       "extra_runs": [{"split": "B", "row": 2, "repeat": 1}],
       "smoke": {"samples": [{"split": "C", "row": 1, "questions": 100, "context_budget": 12000}]}}


def test_full_run_jobs_add_references_once_and_the_extra_repeat():
    planned = jobs(CFG, smoke=False)
    assert [(s["split"], r, arms) for s, r, arms in planned] == [  # references last: a budget stop hits them first
        ("A", 0, ARMS), ("B", 0, ARMS), ("B", 1, ARMS), ("A", 0, REFERENCES), ("B", 0, REFERENCES)]


def test_smoke_jobs_run_every_arm_so_the_references_are_checked_live_once():
    assert [(s["split"], r, arms) for s, r, arms in jobs(CFG, smoke=True)] == [("C", 0, ARMS + REFERENCES)]


def test_a_local_agent_needs_no_key_and_waits_long_enough_for_a_slow_reply(tmp_path):
    agent = {"model": "qwen3.5-9b-32k", "base_url": "http://localhost:11434/v1", "timeout": 1200}
    llm = make_llm({"agent": agent, "cache_dir": tmp_path, "temperature": 0.3}, {}, Budget(tmp_path / "spend.json", 1.0))
    assert llm.client.timeout == 1200  # a 16K-token reply at ~20 tokens/s outlasts the hosted default of 300 s


def test_a_sample_spec_overrides_settings_and_picks_where_its_questions_start(monkeypatch):
    from cacheclm import __main__ as main
    from cacheclm.mab import Sample
    monkeypatch.setattr(main, "load_sample", lambda d, split, row: Sample(f"{split}/{row}", "eventqa_131072", "abcdef",
                                                                         ["q0", "q1", "q2"], [["a0"], ["a1"], ["a2"]]))
    spec = {"split": "A", "row": 12, "start": 1, "end": 4, "question_start": 1, "context_budget": 9, "chunk_tokens": 2}
    sample = load({"data_dir": "d"}, spec)
    assert (sample.context, sample.questions, sample.answers) == ("bcd", ["q1", "q2"], [["a1"], ["a2"]])
    assert sample.sid == "A/12[1:4]q1"  # its own run log, apart from a run of the same text from question 0
    cfg = run_config({"context_budget": 1, "chunk_tokens": 1, "nudges": [0.5]}, spec)
    assert cfg == {"context_budget": 9, "chunk_tokens": 2, "nudges": [0.5]}  # spec keys that are not settings stay out


def test_a_sample_spec_can_pick_its_questions_by_number(monkeypatch):
    from cacheclm import __main__ as main
    from cacheclm.mab import Sample
    monkeypatch.setattr(main, "load_sample", lambda d, split, row: Sample(f"{split}/{row}", "eventqa_131072", "abc",
                                                                         ["q0", "q1", "q2", "q3"], [["a0"], ["a1"], ["a2"], ["a3"]]))
    sample = load({"data_dir": "d"}, {"split": "A", "row": 12, "question_ids": [1, 3]})
    assert (sample.questions, sample.answers) == (["q1", "q3"], [["a1"], ["a3"]])  # only questions about the text shown
    assert sample.sid == "A/12q1-3x2"


def test_a_smoke_sample_can_run_only_some_arms():
    cfg = {"smoke": {"samples": [{"split": "A", "row": 13, "arms": ["clm", "gate", "skill"]}, {"split": "B", "row": 1}]}}
    assert [arms for _, _, arms in jobs(cfg, smoke=True)] == [("clm", "gate", "skill"), ARMS + REFERENCES]
