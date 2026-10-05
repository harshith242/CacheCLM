"""Report per unit (one unit per distinct text; its streaming cost counts once), overall and per family: accuracy,
billed cost, the primary endpoints, repricing under three providers, the fact error split and the chart."""
from collections import defaultdict
from pathlib import Path

import markdown
import matplotlib
import numpy as np

from cacheclm.files import read_jsonl, write_atomic

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARMS = ("summary", "clm", "gate", "skill")
REFERENCES = ("none", "full")
SPLIT = ("prompt", "hit", "ideal", "latency")  # summed per phase, like the costs
COUNTERS = ("edits", "rejected", "rebilled", "forced", "cut_off")  # happen while streaming: count once per unit
SUMMED = ("correct", "n", "retention", "resolution")  # per question set: add up across a unit's samples


def call_cost(call, price, hit_key):
    """USD of one logged call with hits taken from hit_key (billed or ideal cache hits)."""
    hit = min(call[hit_key], call["prompt_tokens"])
    miss = call["prompt_tokens"] - hit
    return (hit * price["cache_read"] + miss * price["cache_write"] + call["completion_tokens"] * price["output"]) / 1e6


def paired(diffs, iters=10000, seed=0):
    """(mean, 95% CI low, high, sign-flip p) of per-unit paired differences."""
    d = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    boot = rng.choice(d, (iters, len(d))).mean(axis=1)
    flips = (rng.choice([-1.0, 1.0], (iters, len(d))) * d).mean(axis=1)
    p = float((np.abs(flips) >= abs(d.mean()) - 1e-12).mean())
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), p


def kept_share(s, c, g, iters=10000, seed=0):
    """(share, CI low, high) of CLM's accuracy gain over summary that the gate keeps; None if CLM gains under 1 point."""
    s, c, g = (np.asarray(x, dtype=float) for x in (s, c, g))
    if c.mean() - s.mean() < 0.01:  # no CLM gain (or a loss): nothing for the gate to keep
        return None
    idx = np.random.default_rng(seed).integers(0, len(s), (iters, len(s)))
    gain = c[idx].mean(axis=1) - s[idx].mean(axis=1)
    ok = np.abs(gain) >= 0.01
    shares = (g[idx].mean(axis=1) - s[idx].mean(axis=1))[ok] / gain[ok]
    share = (g.mean() - s.mean()) / (c.mean() - s.mean())
    return float(share), float(np.percentile(shares, 2.5)), float(np.percentile(shares, 97.5))


def run_row(recs, prices):
    """(done record, metrics) of one finished run log, with streaming and query parts kept apart."""
    done = [r for r in recs if r.get("event") == "done"][-1]
    calls = [r for r in recs if r.get("event") == "call"]
    edits = [r for r in recs if r.get("event") == "edit" and r.get("changed", True)]
    wrong = [r for r in recs if r.get("event") == "answer" and not r["correct"]]
    row = {"correct": done["accuracy"] * done["n_questions"], "n": done["n_questions"],
           "retention": sum(r.get("gold_fact_present") is False for r in wrong),
           "resolution": sum(r.get("gold_fact_present") is True for r in wrong),
           "edits": len(edits), "rejected": sum(not e["allowed"] for e in edits),
           "rebilled": sum(e.get("rebilled_tokens", 0) for e in edits if e["allowed"]),
           "forced": sum(r.get("event") == "forced_truncation" for r in recs),
           "cut_off": sum(r.get("event") == "cut_off" or bool(r.get("event") == "summary" and r.get("cut_off"))
                          for r in recs)}
    keys = {"billed": (prices["deepseek"], "cache_hit_tokens"), **{n: (p, "ideal_hit_tokens") for n, p in prices.items()}}
    for part, chosen in (("stream", [c for c in calls if c["phase"] != "query"]),
                         ("query", [c for c in calls if c["phase"] == "query"])):
        for name, (price, hit_key) in keys.items():
            row[f"{name}_{part}"] = sum(call_cost(c, price, hit_key) for c in chosen)
        for m, field in zip(SPLIT, ("prompt_tokens", "cache_hit_tokens", "ideal_hit_tokens", "latency_s")):
            row[f"{m}_{part}"] = sum(c[field] for c in chosen)
    return done, row


def merge(rows):
    """One unit-arm-repeat row: streaming parts once when the unit's samples replayed the same stream, questions summed.
    If the streams diverged (their costs differ), every stream is a real run, so stream costs are summed instead."""
    diverged = len({round(r["billed_stream"], 9) for r in rows}) > 1
    out = {}
    for m in rows[0]:
        values = [r[m] for r in rows]
        out[m] = sum(values) if m in SUMMED or m.endswith("_query") or (diverged and m.endswith("_stream")) else max(values)
    out["diverged"] = int(diverged)
    for name in [m[:-len("_stream")] for m in rows[0] if m.endswith("_stream")]:
        out[name] = out.pop(f"{name}_stream") + out.pop(f"{name}_query")
    out["accuracy"] = out["correct"] / out["n"]
    return out


def load_units(runs_dir, prices):
    """(means per (unit, arm) over repeats, reference means, {unit: (family, samples)}, gate edits, incomplete runs).
    Only (unit, repeat) pairs finished in every primary arm count; reference arms are listed when present."""
    grouped, info, examples = defaultdict(list), {}, []
    for path in sorted(Path(runs_dir).glob("*/*.jsonl")):
        recs = read_jsonl(path)
        if not any(r.get("event") == "done" for r in recs):
            continue
        done, row = run_row(recs, prices)
        grouped[(done["unit"], done["arm"], done["repeat"])].append(row)
        info.setdefault(done["unit"], (done["family"], set()))[1].add(done["sample"])
        if done["arm"] == "gate":
            examples += [r for r in recs if r.get("event") == "edit" and r.get("changed", True)]
    merged = {key: merge(rows) for key, rows in grouped.items()}
    complete = {(u, r) for u, _, r in merged if all((u, a, r) in merged for a in ARMS)}
    table, refs = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for (u, a, r), row in merged.items():
        target = refs if a in REFERENCES else table if (u, r) in complete else None
        if target is not None:
            for m, v in row.items():
                target[(u, a)][m].append(v)
    incomplete = sorted(f"{u} {a} r{r}" for u, a, r in merged if a not in REFERENCES and (u, r) not in complete)
    mean = lambda t: {k: {m: float(np.mean(v)) for m, v in row.items()} for k, row in t.items()}  # noqa: E731
    return mean(table), mean(refs), info, examples, incomplete


def endpoint_lines(data, units):
    get = lambda arm, m: [data[(u, arm)][m] for u in units]  # noqa: E731
    s_acc, c_acc, g_acc = get("summary", "accuracy"), get("clm", "accuracy"), get("gate", "accuracy")
    m, lo, hi, p = paired(np.subtract(c_acc, s_acc))
    lines = [f"1. **Accuracy, CLM − summary:** {m:+.3f} (95% CI {lo:+.3f} to {hi:+.3f}; sign-flip p = {p:.3f})"]
    for n, arm in ((2, "clm"), (3, "gate"), (5, "skill")):
        m, lo, hi, p = paired(np.log(np.divide(get(arm, "billed"), get("summary", "billed"))))
        label = " (exploratory)" if arm == "skill" else ""
        lines.append(f"{n}. **Billed $ on DeepSeek, {arm} / summary{label}:** {np.exp(m):.2f}x "
                     f"(95% CI {np.exp(lo):.2f}x to {np.exp(hi):.2f}x; sign-flip p = {p:.3f})")
    kept = kept_share(s_acc, c_acc, g_acc)
    lines.insert(3, "4. **Share of CLM's accuracy gain the gate keeps:** "
                 + (f"{kept[0]:.0%} (95% CI {kept[1]:.0%} to {kept[2]:.0%})" if kept else
                    "undefined (CLM gained under 1 point over summary)"))
    m, lo, hi, p = paired(np.subtract(get("skill", "accuracy"), s_acc))
    lines.append(f"6. **Accuracy, skill − summary (exploratory):** {m:+.3f} (95% CI {lo:+.3f} to {hi:+.3f}; "
                 f"sign-flip p = {p:.3f})")
    return lines


def arm_table(data, refs, units):
    md = ["| Arm | Accuracy | Billed $ | Edits | Rejected | Re-billed tokens | Forced truncations | Cut off | "
          "Cache hit (billed / ideal) |", "|---|---|---|---|---|---|---|---|---|"]
    for arm, source in [(a, data) for a in ARMS] + [(a, refs) for a in REFERENCES]:
        rows = [source[(u, arm)] for u in units if (u, arm) in source]
        if not rows:
            continue
        avg = lambda m: np.mean([r[m] for r in rows])  # noqa: E731
        md.append(f"| {arm} | {avg('accuracy'):.3f} | {avg('billed'):.4f} | {avg('edits'):.1f} | "
                  f"{avg('rejected'):.1f} | {avg('rebilled'):,.0f} | {avg('forced'):.1f} | {avg('cut_off'):.1f} | "
                  f"{avg('hit') / max(avg('prompt'), 1):.0%} / {avg('ideal') / max(avg('prompt'), 1):.0%} |")
    return md


def chart(data, units, prices, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    panels = [("DeepSeek (billed)", "billed")] + [(f"{n} (repriced)", n) for n in prices if n != "deepseek"]
    for ax, (title, key) in zip(axes, panels):
        for arm in ARMS:
            xs = [data[(u, arm)][key] for u in units]
            ys = [data[(u, arm)]["accuracy"] for u in units]
            ax.errorbar(np.mean(xs), np.mean(ys), xerr=1.96 * np.std(xs) / np.sqrt(len(xs)),
                        yerr=1.96 * np.std(ys) / np.sqrt(len(ys)), fmt="o", capsize=3, label=arm)
        ax.set_title(title)
        ax.set_xlabel("USD per unit")
    axes[0].set_ylabel("accuracy")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(runs_dir, out_dir, prices):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data, refs, info, examples, incomplete = load_units(runs_dir, prices)
    units = sorted({u for u, _ in data})
    if not units:
        raise SystemExit(f"no unit has finished runs for all four arms in {runs_dir}")
    md = ["# CacheCLM results", "", f"{len(units)} units (one per distinct text), means over repeats. "
          f"Unpaired runs (left out): {', '.join(incomplete) or 'none'}.",
          "Diverged streams (costs summed): " + (", ".join(f"{u} {a}" for (u, a), row in sorted(data.items())
                                                           if row["diverged"]) or "none") + "."]
    families = sorted({info[u][0] for u in units})
    for title, chosen in [("All units", units)] + [(f, [u for u in units if info[u][0] == f]) for f in families]:
        md += ["", f"## {title}", "", f"Units: {len(chosen)}.", ""] + endpoint_lines(data, chosen)
        md += [""] + arm_table(data, refs, chosen)
    md += ["", "## Repriced under three providers (ideal cache, USD per unit)", "",
           "| Arm | " + " | ".join(prices) + " |", "|---|" + "---|" * len(prices)]
    for arm in ARMS:
        md.append(f"| {arm} | " + " | ".join(f"{np.mean([data[(u, arm)][n] for u in units]):.4f}" for n in prices)
                  + " |")
    fact_units = [u for u in units if info[u][0] == "factconsolidation"]
    if fact_units:
        md += ["", "## Fact errors: retention (gold fact lost) vs resolution (gold fact kept, wrong answer)", "",
               "| Arm | Retention | Resolution |", "|---|---|---|"]
        for arm in ARMS:
            md.append(f"| {arm} | {sum(data[(u, arm)]['retention'] for u in fact_units):.0f} | "
                      f"{sum(data[(u, arm)]['resolution'] for u in fact_units):.0f} |")
    md += ["", "## Per unit", "", "| Unit | Family | Samples | " + " | ".join(f"{a} acc | {a} $" for a in ARMS) + " |",
           "|---|---|---|" + "---|---|" * len(ARMS)]
    for u in units:
        md.append(f"| {u} | {info[u][0]} | {', '.join(sorted(info[u][1]))} | "
                  + " | ".join(f"{data[(u, a)]['accuracy']:.2f} | {data[(u, a)]['billed']:.4f}" for a in ARMS) + " |")
    md += ["", "## Gate decisions (examples)", ""]
    picks = [e for e in examples if e["allowed"]][:2] + [e for e in examples if not e["allowed"]][:2]
    md += [f"- `{e['command'][:160]}`: {e['reason']}" for e in picks] or ["- no gate edits logged"]
    md += ["", "![Accuracy vs cost](accuracy_vs_cost.png)"]
    chart(data, units, prices, out / "accuracy_vs_cost.png")
    text = "\n".join(md) + "\n"
    write_atomic(out / "summary.md", text)
    write_atomic(out / "summary.html", "<meta charset='utf-8'>" + markdown.markdown(text, extensions=["tables"]))
