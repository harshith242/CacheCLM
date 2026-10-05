import json

import pytest

from cacheclm.report import call_cost, kept_share, paired, write_report

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


def fake_run(path, arm, sample, accuracy, hit, edits):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"arm": arm, "sample": sample, "source": "factconsolidation_sh_64k", "repeat": 0}
    lines = [{**meta, "event": "call", "phase": "edit", "model": "m", "date": "2026-10-05", "prompt_tokens": 1000,
              "cache_hit_tokens": hit, "ideal_hit_tokens": 900, "completion_tokens": 10, "latency_s": 0.1,
              "cached": False}]
    lines += [{**meta, "event": "edit", "command": "sed x", "allowed": i % 2 == 0, "reason": "r",
               "rebilled_tokens": 100, "deleted_tokens": 50} for i in range(edits)]
    lines.append({**meta, "event": "done", "accuracy": accuracy, "n_questions": 100, "ctx_tokens_final": 5000})
    path.write_text("".join(json.dumps(r) + "\n" for r in lines))


def test_write_report_end_to_end(tmp_path):
    for i, sample in enumerate(["S/1", "S/2", "S/3"]):
        for arm, acc, hit, edits in [("summary", 0.5, 900, 0), ("clm", 0.7, 100, 4), ("gate", 0.65, 800, 2)]:
            fake_run(tmp_path / "runs" / arm / f"S_{i}_r0.jsonl", arm, sample, acc + 0.01 * i, hit, edits)
    write_report(tmp_path / "runs", tmp_path / "out", PRICES)
    text = (tmp_path / "out" / "summary.md").read_text()
    assert "Primary endpoints" in text and "S/3" in text and "anthropic" in text
    assert (tmp_path / "out" / "summary.html").exists() and (tmp_path / "out" / "accuracy_vs_cost.png").exists()
