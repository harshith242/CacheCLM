"""Report: accuracy and billed cost per arm, the three primary endpoints with CIs, the same runs repriced under three
providers, cache behaviour, edit examples, and the accuracy-vs-cost chart."""
from collections import defaultdict
from pathlib import Path

import markdown
import matplotlib
import numpy as np

from cacheclm.files import read_jsonl, write_atomic

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARMS = ("summary", "clm", "gate")


def call_cost(call, price, hit_key):
    """USD of one logged call with hits taken from hit_key (billed or ideal cache hits)."""
    hit = min(call[hit_key], call["prompt_tokens"])
    miss = call["prompt_tokens"] - hit
    return (hit * price["cache_read"] + miss * price["cache_write"] + call["completion_tokens"] * price["output"]) / 1e6


def paired(diffs, iters=10000, seed=0):
    """(mean, 95% CI low, high, sign-flip p) of per-sample paired differences."""
    d = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    boot = rng.choice(d, (iters, len(d))).mean(axis=1)
    flips = (rng.choice([-1.0, 1.0], (iters, len(d))) * d).mean(axis=1)
    p = float((np.abs(flips) >= abs(d.mean()) - 1e-12).mean())
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), p


def kept_share(s, c, g, iters=10000, seed=0):
    """(share, CI low, high) of CLM's accuracy gain over summary that the gate keeps; None if CLM gains under 1 point."""
    s, c, g = (np.asarray(x, dtype=float) for x in (s, c, g))
    if abs(c.mean() - s.mean()) < 0.01:
        return None
    idx = np.random.default_rng(seed).integers(0, len(s), (iters, len(s)))
    gain = c[idx].mean(axis=1) - s[idx].mean(axis=1)
    ok = np.abs(gain) >= 0.01
    shares = (g[idx].mean(axis=1) - s[idx].mean(axis=1))[ok] / gain[ok]
    share = (g.mean() - s.mean()) / (c.mean() - s.mean())
    return float(share), float(np.percentile(shares, 2.5)), float(np.percentile(shares, 97.5))


def load_runs(runs_dir, prices):
    """{(sample, arm): metric means over repeats} from finished runs, plus edit examples from the gate arm."""
    table, examples = defaultdict(lambda: defaultdict(list)), []
    for path in sorted(Path(runs_dir).glob("*/*.jsonl")):
        recs = read_jsonl(path)
        done = [r for r in recs if r.get("event") == "done"]
        if not done:
            continue
        calls = [r for r in recs if r.get("event") == "call"]
        edits = [r for r in recs if r.get("event") == "edit"]
        row = table[(done[-1]["sample"], done[-1]["arm"])]
        row["accuracy"].append(done[-1]["accuracy"])
        row["correct"].append(done[-1]["accuracy"] * done[-1]["n_questions"])
        row["billed"].append(sum(call_cost(c, prices["deepseek"], "cache_hit_tokens") for c in calls))
        for name, price in prices.items():
            row[name].append(sum(call_cost(c, price, "ideal_hit_tokens") for c in calls))
        row["edits"].append(len(edits))
        row["rejected"].append(sum(not e["allowed"] for e in edits))
        row["rebilled"].append(sum(e.get("rebilled_tokens", 0) for e in edits if e["allowed"]))
        row["forced"].append(sum(r.get("event") == "forced_truncation" for r in recs))
        row["prompt"].append(sum(c["prompt_tokens"] for c in calls))
        row["hit"].append(sum(c["cache_hit_tokens"] for c in calls))
        row["latency"].append(sum(c["latency_s"] for c in calls))
        row["ideal"].append(sum(c["ideal_hit_tokens"] for c in calls))
        if done[-1]["arm"] == "gate":
            examples += edits
    means = {k: {m: float(np.mean(v)) for m, v in row.items()} for k, row in table.items()}
    return means, examples


def endpoint_lines(data, samples):
    get = lambda arm, m: [data[(s, arm)][m] for s in samples]  # noqa: E731
    s_acc, c_acc, g_acc = get("summary", "accuracy"), get("clm", "accuracy"), get("gate", "accuracy")
    m, lo, hi, p = paired(np.subtract(c_acc, s_acc))
    lines = [f"1. **Accuracy, CLM − summary:** {m:+.3f} (95% CI {lo:+.3f} to {hi:+.3f}; sign-flip p = {p:.3f})"]
    for n, arm in ((2, "clm"), (3, "gate")):
        m, lo, hi, p = paired(np.log(np.divide(get(arm, "billed"), get("summary", "billed"))))
        lines.append(f"{n}. **Billed $ on DeepSeek, {arm} / summary:** {np.exp(m):.2f}x "
                     f"(95% CI {np.exp(lo):.2f}x to {np.exp(hi):.2f}x; sign-flip p = {p:.3f})")
    kept = kept_share(s_acc, c_acc, g_acc)
    lines.append("4. **Share of CLM's accuracy gain the gate keeps:** "
                 + (f"{kept[0]:.0%} (95% CI {kept[1]:.0%} to {kept[2]:.0%})" if kept else
                    "undefined (CLM gained under 1 point over summary)"))
    return lines


def chart(data, samples, prices, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    panels = [("DeepSeek (billed)", "billed")] + [(f"{n} (repriced)", n) for n in prices if n != "deepseek"]
    for ax, (title, key) in zip(axes, panels):
        for arm in ARMS:
            xs = [data[(s, arm)][key] for s in samples]
            ys = [data[(s, arm)]["accuracy"] for s in samples]
            ax.errorbar(np.mean(xs), np.mean(ys), xerr=1.96 * np.std(xs) / np.sqrt(len(xs)),
                        yerr=1.96 * np.std(ys) / np.sqrt(len(ys)), fmt="o", capsize=3, label=arm)
        ax.set_title(title)
        ax.set_xlabel("USD per sample")
    axes[0].set_ylabel("accuracy")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(runs_dir, out_dir, prices):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data, examples = load_runs(runs_dir, prices)
    samples = sorted({s for s, _ in data if all((s, a) in data for a in ARMS)})
    if not samples:
        raise SystemExit(f"no sample has finished runs for all three arms in {runs_dir}")
    incomplete = sorted({s for s, _ in data} - set(samples))
    md = ["# CacheCLM results", "", f"{len(samples)} samples, means over repeats. Incomplete samples (left out): "
          f"{', '.join(incomplete) or 'none'}.", "", "## Primary endpoints", ""]
    md += endpoint_lines(data, samples)
    md += ["", "## Per arm", "", "| Arm | Accuracy | Billed $ | $ per correct | Edits | Rejected | Re-billed tokens | "
           "Forced truncations | Cache hit (billed / ideal) | Call time s |", "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        rows = [data[(s, arm)] for s in samples]
        avg = lambda m: np.mean([r[m] for r in rows])  # noqa: E731
        md.append(f"| {arm} | {avg('accuracy'):.3f} | {avg('billed'):.4f} | "
                  f"{sum(r['billed'] for r in rows) / max(sum(r['correct'] for r in rows), 1):.6f} | "
                  f"{avg('edits'):.1f} | {avg('rejected'):.1f} | {avg('rebilled'):,.0f} | {avg('forced'):.1f} | "
                  f"{avg('hit') / avg('prompt'):.0%} / {avg('ideal') / avg('prompt'):.0%} | {avg('latency'):.0f} |")
    md += ["", "## Repriced under three providers (ideal cache, USD per sample)", "",
           "| Arm | " + " | ".join(prices) + " |", "|---|" + "---|" * len(prices)]
    for arm in ARMS:
        md.append(f"| {arm} | " + " | ".join(f"{np.mean([data[(s, arm)][n] for s in samples]):.4f}" for n in prices)
                  + " |")
    md += ["", "## Per sample", "", "| Sample | " + " | ".join(f"{a} acc | {a} $" for a in ARMS) + " |",
           "|---|" + "---|---|" * len(ARMS)]
    for s in samples:
        md.append(f"| {s} | " + " | ".join(f"{data[(s, a)]['accuracy']:.2f} | {data[(s, a)]['billed']:.4f}"
                                           for a in ARMS) + " |")
    md += ["", "## Gate decisions (examples)", ""]
    picks = [e for e in examples if e["allowed"]][:2] + [e for e in examples if not e["allowed"]][:2]
    md += [f"- `{e['command'][:160]}`: {e['reason']}" for e in picks] or ["- no gate edits logged"]
    md += ["", "![Accuracy vs cost](accuracy_vs_cost.png)"]
    chart(data, samples, prices, out / "accuracy_vs_cost.png")
    text = "\n".join(md) + "\n"
    write_atomic(out / "summary.md", text)
    write_atomic(out / "summary.html", "<meta charset='utf-8'>" + markdown.markdown(text, extensions=["tables"]))
