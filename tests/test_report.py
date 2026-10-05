import json

import pytest

from cacheclm.report import call_cost, kept_share, load_units, paired, write_report

PRICES = {"deepseek": {"cache_read": 0.006, "cache_write": 0.30, "output": 1.20},
          "openai": {"cache_read": 0.175, "cache_write": 1.75, "output": 14.0},
          "anthropic": {"cache_read": 0.30, "cache_write": 3.75, "output": 15.0}}


def test_call_cost_reprices_with_the_ideal_hit():
    call = {"prompt_tokens": 1000, "cache_hit_tokens": 600, "ideal_hit_tokens": 800, "completion_tokens": 100}
    assert call_cost(call, PRICES["anthropic"], "ideal_hit_tokens") == pytest.approx((800 * 0.3 + 200 * 3.75 + 1500) / 1e6)
    assert call_cost(call, PRICES["deepseek"], "cache_hit_tokens") == pytest.approx((600 * 0.006 + 400 * 0.3 + 120) / 1e6)


def test_paired_stats_on_consistent_gains():
    mean, low, high, p = paired([0.1] * 5)
    assert mean == pytest.approx(0.1) and low == pytest.approx(0.1) and high == pytest.approx(0.1)
    assert p == pytest.approx(2 / 32, abs=0.02)  # only all-plus or all-minus sign flips reach the mean


def test_kept_share():
    share, _, _ = kept_share([0.5, 0.5], [0.7, 0.7], [0.6, 0.6])
    assert share == pytest.approx(0.5)
    assert kept_share([0.5, 0.5], [0.5, 0.5], [0.6, 0.6]) is None
    assert kept_share([0.5, 0.5], [0.45, 0.45], [0.48, 0.48]) is None  # CLM lost accuracy: no gain to keep


def call(phase, hit, prompt=1000):
    return {"event": "call", "phase": phase, "model": "m", "prompt_tokens": prompt, "cache_hit_tokens": hit,
            "ideal_hit_tokens": hit, "completion_tokens": 10, "latency_s": 0.1, "cached": False}


def fake_run(runs, arm, sample, unit, family, accuracy, stream_hit=900, edits=0, repeat=0, wrong_present=None):
    path = runs / arm / f"{sample.replace('/', '_')}_r{repeat}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"arm": arm, "sample": sample, "source": family, "repeat": repeat}
    lines = [{**meta, **call("edit", stream_hit)}, {**meta, **call("query", 990)}]
    lines += [{**meta, "event": "edit", "command": "sed x", "allowed": i % 2 == 0, "reason": "r", "changed": True,
               "rebilled_tokens": 100, "deleted_tokens": 50} for i in range(edits)]
    if wrong_present is not None:
        lines.append({**meta, "event": "answer", "qi": 0, "prediction": "x", "correct": False,
                      "gold_fact_present": wrong_present})
    lines.append({**meta, "event": "done", "accuracy": accuracy, "n_questions": 100, "ctx_tokens_final": 5000,
                  "unit": unit, "family": family, "final_context": "ctx"})
    path.write_text("".join(json.dumps(r) + "\n" for r in lines))


ARM_ACC = [("summary", 0.5, 900), ("clm", 0.7, 100), ("gate", 0.65, 800), ("skill", 0.6, 850)]


def test_a_unit_counts_its_streaming_cost_once(tmp_path):
    runs = tmp_path / "runs"
    for arm, acc, hit in ARM_ACC:
        fake_run(runs, arm, "CR/6", "u1", "factconsolidation", acc, hit)
        fake_run(runs, arm, "CR/2", "u1", "factconsolidation", acc, hit)  # same text: its stream replayed for free
    means, _, info, _, _ = load_units(runs, PRICES)
    one_stream = call_cost(call("edit", 900), PRICES["deepseek"], "cache_hit_tokens")
    one_query = call_cost(call("query", 990), PRICES["deepseek"], "cache_hit_tokens")
    assert means[("u1", "summary")]["billed"] == pytest.approx(one_stream + 2 * one_query)
    assert info["u1"] == ("factconsolidation", {"CR/2", "CR/6"})


def test_report_has_families_references_and_the_fact_error_split(tmp_path):
    runs = tmp_path / "runs"
    for u in ("e1", "e2", "e3"):
        for arm, acc, hit in ARM_ACC:
            fake_run(runs, arm, f"AR/{u}", u, "eventqa", acc, hit, edits=2)
    for arm, acc, hit in ARM_ACC:
        fake_run(runs, arm, "CR/6", "f1", "factconsolidation", acc, hit, wrong_present=(arm == "clm"))
    fake_run(runs, "none", "AR/e1", "e1", "eventqa", 0.3)
    write_report(runs, tmp_path / "out", PRICES)
    text = (tmp_path / "out" / "summary.md").read_text()
    assert "## All units" in text and "## eventqa" in text and "## factconsolidation" in text
    assert "CLM − summary:** +0.200" in text
    assert "| none |" in text  # the reference arm is listed
    assert "| clm | 0 | 1 |" in text  # factconsolidation errors: retention 0, resolution 1
    assert (tmp_path / "out" / "accuracy_vs_cost.png").exists()


def test_only_unit_repeats_finished_in_every_arm_count(tmp_path):
    runs = tmp_path / "runs"
    for arm, acc, hit in ARM_ACC:
        fake_run(runs, arm, "CR/6", "f1", "factconsolidation", acc, hit)
    fake_run(runs, "clm", "CR/6", "f1", "factconsolidation", 0.9, repeat=1)  # r1 has no other arm
    means, _, _, _, incomplete = load_units(runs, PRICES)
    assert means[("f1", "clm")]["accuracy"] == pytest.approx(0.7)
    assert incomplete == ["f1 clm r1"]


def test_a_unit_whose_streams_diverged_sums_its_streams(tmp_path):
    runs = tmp_path / "runs"
    for arm, acc, hit in ARM_ACC:
        fake_run(runs, arm, "CR/6", "u1", "factconsolidation", acc, hit)
        fake_run(runs, arm, "CR/2", "u1", "factconsolidation", acc, hit if arm != "clm" else 300)  # clm diverged
    means, _, _, _, _ = load_units(runs, PRICES)
    cost = lambda hit: call_cost(call("edit", hit), PRICES["deepseek"], "cache_hit_tokens")  # noqa: E731
    query = call_cost(call("query", 990), PRICES["deepseek"], "cache_hit_tokens")
    assert means[("u1", "clm")]["billed"] == pytest.approx(cost(100) + cost(300) + 2 * query)
    assert means[("u1", "clm")]["diverged"] == 1 and means[("u1", "summary")]["diverged"] == 0
    write_report(runs, tmp_path / "out", PRICES)
    assert "Diverged streams (costs summed): u1 clm" in (tmp_path / "out" / "summary.md").read_text()
