# CacheCLM

**Question:** when an agent edits its own context (a Context Language Model, or CLM), does it still save money on a hosted API that bills by prompt cache?

## Why it matters

[Context Language Models](https://arxiv.org/abs/2609.37725) (Shao et al., Sep 2026) give the model its context as a file it edits with bash. The paper reports higher accuracy than harness-scheduled summarization, with fewer FLOPs, on a self-hosted server. On that server, a mid-context edit can reuse cached work (the paper's Suffix Cache Reuse).

Hosted APIs are different. Their prompt cache reuses only the start of a request that matches the previous one. An edit in the middle therefore turns every later token into a cache miss, and DeepSeek bills a miss at 50x a hit. This project measures that trade-off in billed dollars.

## The three arms

All three use the same model (DeepSeek Flash, thinking off), the same text chunks, questions and pinned task block, and a 32K-token context budget.

1. **Summary.** When the context passes 75% of the budget, one call compacts everything except the last 2 chunks into a summary.
2. **CLM.** The model edits `ctx.txt` with one-line shell commands between chunks.
3. **CLM + gate.** The same, but each edit must pass the cache-aware gate below. The model only notices the gate when it rejects an edit.

## The gate

An edit is allowed only if what it saves on later calls outweighs the cached tokens it forces the provider to re-bill:

```text
deleted_tokens × turns_left × hit_price   >=   unchanged_tokens_after_the_edit × (miss_price − hit_price)
```

If the next chunk would not fit otherwise, any edit that shrinks the context is allowed.

Example: DeepSeek prices (hit $0.006/M, miss $0.30/M), a 30K context, 10 turns left:

| Edit | Deleted | Re-billed after it | Cost | Saving | Decision |
|---|---|---|---|---|---|
| A search result in the middle | 6K | 16K | $0.0047 | $0.00036 | reject |
| The newest tool output | 6K | 0.5K | $0.00015 | $0.00036 | allow |

## Data

**Source:** [MemoryAgentBench](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench) (MIT), pinned to commit `7ea0669`. Only two parquet files are used. `scripts/get_data.py` downloads them over plain HTTPS and checks their SHA-256.

**Samples:** 7, chosen by length before any run:
- the 5 EventQA books at about 140K tokens (`eventqa_131072`);
- FactConsolidation single-hop and multi-hop at about 68K tokens (`*_64k`).

Each streams 2-4.4x the context budget through the agent, in 4K-token chunks.

**Prompts and scoring** are adapted from the benchmark's `utils/templates.py` and `utils/eval_other_utils.py` (`substring_exact_match`).

## Run it

Needs macOS (edits run under `sandbox-exec`), Python 3.13, [uv](https://docs.astral.sh/uv/), and `DEEPSEEK_API_KEY` in a `.env` file.

```bash
uv sync
uv run python scripts/get_data.py
uv run cacheclm smoke                # 1 sample, 5 questions, 3 arms: about $0.10
uv run cacheclm report --smoke
uv run cacheclm run                  # 7 samples x 3 arms x 2 repeats
uv run cacheclm report               # results/summary.md, summary.html, accuracy_vs_cost.png
```

**Cost:**
- **Cap:** every real API call counts against a $5 cap stored in `cache/spend.json`. The full run is estimated at about $4.
- **Replays:** cached calls replay for free.
- **Resuming:** an interrupted run skips the (sample, arm, repeat) runs that already finished.

## Credits

- **Method and context-file format:** Context Language Models (arXiv 2609.37725). Its repository is CC BY-NC 4.0, so no code is copied.
- **Data and scoring:** MemoryAgentBench (arXiv 2507.05257), MIT.
- **Code:** MIT.
