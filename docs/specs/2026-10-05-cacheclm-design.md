# CacheCLM: design

Date: 2026-10-05. Project 5 of `llm_systems_techniques/projects_revised.md`, designed in brainstorming.

## Question

Context Language Models (CLMs; Shao et al., arXiv 2609.37725, Sep 2026) let the model edit its own context, which is mirrored to a file it changes with bash. The paper reports higher accuracy and fewer FLOPs than harness-scheduled summarization, but it measures cost on a self-hosted SGLang server. There, a mid-context edit can reuse cached work through the paper's Suffix Cache Reuse patch.

Hosted APIs work differently. The prompt cache reuses only the prefix that matches the previous request. An edit in the middle of the context therefore makes every later token a cache miss, and DeepSeek bills a miss at 50x a hit. No provider offers suffix reuse (as of Oct 2026).

**Headline question:** on a hosted API with prompt caching, does model-edited context still pay off once billed dollars are counted? Does a cache-aware gate keep the accuracy gain while cutting the cost?

Hypotheses, stated before any live run:
- **H1.** CLM is more accurate than the summary baseline but costs more in billed dollars on DeepSeek, because mid-context edits break the prompt cache.
- **H2.** CLM + gate keeps most of CLM's accuracy gain at a billed cost close to or below the summary baseline.
- **H3.** Repriced under OpenAI and Anthropic price sheets, where a miss costs about 10x a hit, unrestricted editing becomes relatively cheaper.

Out of scope: RL training, Suffix Cache Reuse itself, multi-agent swarms, tail-only editing, append-only notes, a separate context-manager ("hypervisor") model, and injection-persistence tests. These are possible follow-ups if the main result is clear.

## Setting

- **Benchmark:** MemoryAgentBench (Hu et al., ICLR 2026; Hugging Face `ai-hyz/MemoryAgentBench`, MIT, 76.6 MB). Each sample is one long text followed by 60-100 questions. The text is fed to the agent in chunks, one per message, so the agent must manage its context while reading.
- **Subsets:** Accurate Retrieval (exact facts in the streamed text) and Conflict Resolution (FactConsolidation, where later facts override earlier ones). Long-Range Understanding (LLM-judged summaries) and Test-Time Learning are skipped.
- **Samples:** about 10 of the shortest samples across both subsets (up to about 150K tokens each), chosen by length only, before any run. Every sample is at least 5x the context budget.
- **Chunks:** 4K tokens per message, in the benchmark's order. Order matters for Conflict Resolution, so it is never shuffled.
- **Context budget:** 32K tokens, as in the paper.
- **Model:** DeepSeek Flash, thinking off, temperature 0.3, for every call in every arm.
- **Repeats:** 2 per arm and sample. Chunk order is fixed by the task, so repeats replace question orders as the noise measure.
- **Scoring:** MemoryAgentBench's own matcher for each subset.
- **Budget:** hard cap of $5 for live runs. Estimate: about $3.5 (about $0.06 per sample, arm and repeat), plus about $0.10 for the smoke test.
- **Data download:** only after the user approves it.

## Arms

All three arms share the model, the system prompt, the chunks, the questions and the 32K budget.

1. **Summary (baseline).** When the context passes 75% of the budget, one LLM call compacts everything except the pinned blocks and the last 2 chunks into a single summary block. This is Codex-style harness-scheduled compaction, the paper's main baseline.
2. **CLM.** The model has one bash tool over `ctx.txt`.
   - Nudges are sent at 25%, 50% and 75% of the budget.
   - Before each chunk, the model may run up to 3 edit commands. Edit turns are free and do not count as steps.
   - If the next chunk would overflow the budget, the harness asks the model to condense first, as the paper's rollback-and-condense does.
3. **CLM + gate.** The same as CLM, plus the cache-aware gate (below) checking every edit.

## The cache-aware gate

The prompt cache reuses the request prefix up to the first changed token. An edit therefore has:
- a one-time cost: every token after the edit point loses its cache once;
- a recurring saving: the deleted tokens are no longer sent on each remaining call, but those tokens would have been cheap cache hits.

```text
allow if   deleted_tokens × turns_left × hit_price   >=   tokens_after_edit × (miss_price − hit_price)
```

- `tokens_after_edit` counts from the first changed token to the end of the old context.
- `deleted_tokens` is the net shrink, old length minus new length, floored at 0. An edit that does not shrink the context has no saving, so it passes only when it invalidates nothing (a pure append). Any rewrite that keeps the length is rejected.
- `turns_left` = chunks remaining + number of questions. The harness knows both; the model never sees the questions in advance.
- **Override:** if the context would otherwise overflow the budget, an edit is allowed regardless.
- **A rejected edit** is reverted. The model gets one line, for example: `edit rejected: breaks 16,000 cached tokens to save 6,000; prefer deleting near the end`.

**Worked example** (DeepSeek, hit $0.006/M, miss $0.30/M; 30K context; 10 turns left):

| Edit | Deleted | Tokens after edit | Cost | Saving | Decision |
|---|---|---|---|---|---|
| Middle search result | 6K | 16K | $0.0047 | $0.00036 | reject |
| Latest tool output | 6K | 0.5K | $0.00015 | $0.00036 | allow |

At Anthropic prices (read $0.30/M, write $3.75/M), the middle edit costs $0.055. It is still rejected at 10 turns left (saving $0.018) and becomes roughly break-even at 30 turns left ($0.054).

The gate uses a local tokenizer estimate, because it must decide before the next call. Billed usage fields are the source of truth for cost, and the gap between estimate and billing is logged.

## How a sample runs

1. **Streaming phase.** For each chunk:
   - The harness writes `ctx.txt`.
   - The model replies with a bash command, or `ready` for the next chunk. At most 3 commands per chunk.
   - The sandbox runs the command, and the harness parses `ctx.txt` back into messages.
   - Gate arm only: `decide(old, new)` allows the edit or reverts it.
   - If a pinned block is damaged, the edit is rolled back.
   - The harness then appends the next chunk.
   - The summary arm skips the edit turns and compacts at the 75% threshold instead.
2. **Query phase.** The final context is frozen. Each question is asked in its own call, with no edits allowed. Separate calls keep answers independent, and because the frozen context is cached, they cost about $0.02 per sample.

## Components

Repo `new-techniques/cacheclm/`: its own git repo, MIT, Python 3.13 with uv. Each module has one job:

- **`ctxfile.py`:** converts the message list to and from `ctx.txt` using the paper's `[[CTX_TURN i role=…]]` block format. The system prompt and the task block are pinned.
- **`sandbox.py`:** runs one bash command in a temp folder that contains only `ctx.txt`.
  - Allow-list: `sed`, `awk`, `grep`, `head`, `tail`, `cat`, `python3`, plus redirects.
  - No network, a 10 s timeout, and output capped at 2K characters.
- **`gate.py`:** pure functions.
  - `first_change(old, new)` returns the token position of the first difference.
  - `decide(old, new, turns_left, prices)` returns allow or reject with a one-line reason.
- **`arms.py`:** the three context policies behind one interface, `before_chunk(ctx) -> ctx`.
- **`run.py`:** the streaming and query loop. Every call is logged with the model id, date, prompt, cache-hit and cache-miss tokens, output tokens and wall time.
- **`llm.py`:** an OpenAI-compatible client.
  - Responses are cached by request hash, so re-running the analysis is free.
  - A budget guard stops at the cap, and runs resume after a stop.
- **`reprice.py`:** recomputes each call's cost under the three price sheets in `prices.yaml`.
- **`report.py`:** writes `results/summary.md` and `summary.html`.

Data is pinned by hash, like EvoSQL's data scripts.

## Error handling

- **Malformed `ctx.txt` after an edit:** roll back and tell the model "parse failed".
- **Banned command:** rejected without running, with the allow-list repeated.
- **No room made before an overflowing chunk:** the harness drops the oldest unpinned blocks and logs a forced truncation, which is counted per arm.
- **API error:** 3 retries. After that the sample is marked failed, skipped, and failures are reported per arm.

## Metrics and analysis

**Primary endpoints** (fixed before any live run):
1. Accuracy: CLM − Summary (H1).
2. Billed dollars on DeepSeek: CLM / Summary (H1).
3. Gate: the share of CLM's accuracy gain kept, (Gate − Summary) / (CLM − Summary), and billed dollars Gate / Summary (H2).

**Secondary:**
- accuracy by subset;
- cost per correct answer;
- edits proposed, accepted and gate-rejected;
- tokens invalidated per edit;
- forced truncations;
- observed vs ideal cache-hit rate, where ideal is the prefix shared with the previous request;
- wall time.

**Repricing (H3), with no extra spend.** For each logged call, the ideal cached prefix and the new tokens are priced under three sheets:
- **DeepSeek:** hit $0.006/M, miss $0.30/M.
- **OpenAI:** cached input at 10% of the input price.
- **Anthropic:** reads at 10% of the input price; new tokens billed as cache writes at 125%, assuming a breakpoint at the end of every call. That is the best case for Anthropic; its 5-minute TTL is ignored because runs are fast.

Prices are stored with their date.

**Statistics:**
- Results are paired by sample and repeat.
- Bootstrap 95% CIs over samples, plus a sign-flip test on per-sample differences.
- With about 10 samples, a per-sample table is shown next to every aggregate, and results are framed as direction plus effect size.

**Report:**
- The headline chart: accuracy vs dollars per sample, in three panels (DeepSeek, OpenAI and Anthropic pricing), one point per arm with CIs.
- 3-4 real edit examples, allowed vs rejected, with the gate's numbers.
- A caveats section.

## Testing

Unit tests only, no network:
- **`ctxfile`:** round-trip; damaged pinned blocks are detected.
- **`first_change`:** edits at the start, middle and end; a no-op; an append-only edit.
- **`gate.decide`:**
  - the worked example (middle edit rejected and tail edit allowed on DeepSeek);
  - the Anthropic middle edit becomes roughly break-even at 30 turns left;
  - the overflow override;
  - a non-shrinking mid-context edit is rejected.
- **`sandbox`:** a banned command is rejected, the timeout fires, nothing outside the temp folder is touched.
- **`reprice`:** a hand-computed call under each sheet.
- **Budget guard:** stops at the cap.
- **Replay:** a replay from cache returns identical logs.

## Runs

1. **Smoke:** 1 short sample, 5 questions, all 3 arms, about $0.10.
2. **Full:** about 10 samples × 3 arms × 2 repeats, hard cap $5.

The user runs both in their own terminal, from a command handed over. Live runs are not extended after seeing results; a follow-up would be a separate, newly stated run.

## Risks

- **The CLM arm may edit rarely ("inert").** Edit counts are reported, as EvoSQL reports injection rates.
- **DeepSeek caching is best-effort and works in 64-token units.** The observed vs ideal gap is measured and reported.
- **Small n (about 10 samples).** Effect sizes and per-sample tables carry the result.
- **The gate's `turns_left` is large in the query phase.** Edits during streaming are judged by both remaining chunks and questions, which favours shrinking before the questions. This is intended, because the frozen context is re-read once per question.
- **The tokenizer estimate differs from billed counts.** The gap is logged.

## Credits

- Context Language Models (arXiv 2609.37725) for the method and the context-file format. Its repository is CC BY-NC 4.0, so no code is copied.
- MemoryAgentBench (arXiv 2507.05257) for the data.
