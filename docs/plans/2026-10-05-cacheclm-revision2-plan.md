# CacheCLM Revision 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement spec Revision 2:
- the harness fixes;
- the paper-faithful CLM prompt (prompt A);
- a 4th `skill` arm;
- gate feedback only on rejection;
- the summary fix;
- the `none` and `full` reference arms;
- unit- and family-level reporting with the FactConsolidation retention/resolution split;
- the extra FactConsolidation repeat;
- the two smoke samples on data that is not evaluated.

**Architecture:**
- **Context.** The pinned task block becomes a separate string (`task`), and the arms edit only `body`. Every request sends `task + body`, so the cache prefix is unchanged.
- **Forced truncation** cuts whole lines from the oldest body text and re-adds a header.
- **The report** groups runs into units (one unit per distinct text). A unit's streaming cost is counted once, and results are reported overall and per family.

**Tech Stack:** Python 3.13, uv, `openai` client (DeepSeek), `pyarrow`, `numpy`, `matplotlib`, `markdown`, `pytest`.

**Spec:** `docs/specs/2026-10-05-cacheclm-design.md`. The section "Revision 2" is binding; the earlier sections give the context.

## Global Constraints
- **Model:** DeepSeek Flash with thinking off, temperature 0.3, the same for every call in every arm.
- **Budget:** hard cap of $5 (`budget_usd: 5.0`), of which $0.49 has already been spent.
- **Context:** a 32K-token budget and 4K-token chunks for the full run. The FactConsolidation smoke uses a 12K budget.
- **No live full run.** Only the two smoke samples may be run live, after the final review.
- **Code style:** simple code with at most one-line comments; a file docstring of at most 3 lines.
- **Commits:** no co-author or AI-attribution lines.
- **Data:** the pinned MemoryAgentBench parquet files only (`7ea0669`); no new downloads.

## Review Focus
1. **An edit that keeps only the last N lines of ctx.txt** must leave the newest facts in place, never push the context over the budget, and leave the body starting with a header. Owned by Task 1: `test_keeping_the_last_lines_keeps_the_newest_facts_within_budget`.
2. **A model-written header containing `pinned`** must be stripped, so its block stays droppable. Owned by Task 1: `test_model_written_pinned_header_is_stripped`.
3. **An edit that empties ctx.txt** must be rolled back, not applied. Owned by Task 1: `test_an_edit_that_empties_ctx_is_rolled_back`.
4. **An edit reply cut off at the token cap** must be logged as `cut_off`, and the edit phase must continue rather than treating the reply as READY. Owned by Task 1: `test_a_cut_off_reply_is_not_read_as_ready`.
5. **FactConsolidation rows 2 and 6 share their text.** Their streaming cost must be counted once per unit, not twice. Owned by Task 2: `test_a_unit_counts_its_streaming_cost_once`.

---

## File structure

```text
src/cacheclm/ctxfile.py   rewritten: body-only helpers (blocks, next_index, append, normalize, cut_oldest_lines,
                          split_for_summary); drop_oldest and split_pinned removed
src/cacheclm/arms.py      rewritten: task/body signatures, prompt A, SKILL, 4 arms + references, gate feedback,
                          emptied/cut_off handling, line-level fit, summary cut-off logging
src/cacheclm/llm.py       modified: replies carry finish_reason
src/cacheclm/mab.py       modified: gold_fact()
src/cacheclm/run.py       rewritten: task + body, references, unit/family/final_context in done, gold_fact_present
src/cacheclm/report.py    rewritten: units, families, references, error split, 4-arm chart
src/cacheclm/__main__.py  rewritten: jobs() (samples, extra repeat, references, smoke samples with overrides)
configs/base.yaml         modified: edit_max_tokens, summary_words 1500, extra_runs, smoke samples
tests/                    test_ctxfile rewritten; test_run, test_llm, test_mab, test_report updated
```

---

### Task 1: Harness fixes and arm changes (task/body split, prompt A, skill arm, gate feedback, summary fix)

**Files:**
- Rewrite: `src/cacheclm/ctxfile.py`, `src/cacheclm/arms.py`
- Modify: `src/cacheclm/llm.py`, `src/cacheclm/run.py`, `src/cacheclm/mab.py`, `configs/base.yaml`
- Test: `tests/test_ctxfile.py` (rewrite), `tests/test_run.py`, `tests/test_llm.py`, `tests/test_mab.py`

**Interfaces:**
- **Produces:**
  - `ctxfile.normalize(body, role="notes") -> str`
  - `ctxfile.cut_oldest_lines(body, chars) -> str`
  - `ctxfile.split_for_summary(body, keep) -> (older, tail) | None`
  - `ctxfile.append(body, role, text) -> str`
  - `ctxfile.next_index(body) -> int`
  - `arms.ARMS = ("summary", "clm", "gate", "skill")`, `arms.REFERENCES = ("none", "full")`, `arms.SKILL`
  - `arms.system(arm, budget, repeat=0)`
  - `arms.edit_phase(task, body, arm, chat, cfg, incoming, turns_left, price, log, repeat=0) -> body`
  - `arms.summary_step(task, body, chat, cfg, incoming, log, repeat=0) -> body`
  - `arms.fit(task, body, incoming, budget, log) -> body`
  - `mab.gold_fact(context, question, golds) -> str | None`
  - `run.run_sample(...)`, same signature as before. The done record gains `unit`, `family` and `final_context`; answer records gain `gold_fact_present`.
  - LLM replies gain `finish_reason`.
- **Consumes:** `sandbox.run(command, text) -> (text, output)`, `gate.decide(old, new, turns_left, price, overflow)`, and the existing `mab` constants.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_ctxfile.py` entirely:
```python
from cacheclm.ctxfile import append, blocks, cut_oldest_lines, next_index, normalize, split_for_summary


def build(n):
    body = ""
    for i in range(n):
        body = append(body, "chunk", f"part {i}")
    return body


def test_body_blocks_are_numbered_from_one():
    assert next_index("") == 1
    assert [(b[2], b[3]) for b in blocks(build(3))] == [(1, "chunk"), (2, "chunk"), (3, "chunk")]


def test_append_after_an_edit_without_trailing_newline():
    body = build(1) + "[[CTX_TURN 7 role=notes]]\nmy note"
    assert [b[2] for b in blocks(append(body, "chunk", "next"))] == [1, 7, 8]


def test_normalize_heads_loose_text_and_strips_model_pinned_marks():
    assert normalize("\n\n12. a fact\n").startswith("[[CTX_TURN 1 role=notes]]\n12. a fact")
    assert normalize("[[CTX_TURN 4 role=notes pinned]]\nx\n") == "[[CTX_TURN 4 role=notes]]\nx\n"
    assert normalize("") == "" and normalize(build(2)) == build(2)


def test_cut_oldest_lines_keeps_the_newest_lines_under_a_header():
    body = "".join(f"{i}. fact {i}\n" for i in range(10))
    cut = cut_oldest_lines(body, 30)
    assert cut.startswith("[[CTX_TURN 1 role=chunk]]\n") and cut.endswith("9. fact 9\n")
    assert "0. fact 0" not in cut and len(cut) < len(body)


def test_split_for_summary_keeps_the_last_chunks():
    older, tail = split_for_summary(build(4), keep=2)
    assert "part 0" in older and "part 1" in older and "part 2" not in older
    assert tail.startswith("[[CTX_TURN 3 role=chunk]]") and "part 3" in tail
    assert split_for_summary(build(2), keep=2) is None
```

In `tests/test_run.py`:
- In `CFG`, add `"edit_max_tokens": 200`.
- Change the `FakeLLM.chat` body so a script may return `(content, finish_reason)`:
```python
    def chat(self, messages, max_tokens, repeat):
        self.calls += 1
        text = messages[-1]["content"]
        out = self.script(text)
        content, finish = out if isinstance(out, tuple) else (out, "stop")
        return {"content": content, "model": "fake", "prompt_tokens": len(text) // 4, "cache_hit_tokens": 0,
                "completion_tokens": 10, "latency_s": 0.0, "cached": False, "finish_reason": finish}
```
- Add these helpers and tests at the end of the file:
```python
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


def test_a_cut_off_reply_is_not_read_as_ready(tmp_path):
    replies = iter([("THOUGHT: keep the newest.\n```bash\npython3 -c \"print(1)", "length")])
    llm = FakeLLM(lambda t: answers(t) or next(replies, "READY"))
    run_sample(sample(), "clm", 0, llm, CFG, PRICE, tmp_path)
    records = [r["event"] for r in read_jsonl(run_log(tmp_path, "clm", sample(), 0))]
    after = records[records.index("cut_off") + 1:]
    assert next(e for e in after if e in ("call", "step")) == "call"  # same phase asked again, not ended as READY


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
```
- Update the imports at the top of `tests/test_run.py`:
```python
from cacheclm.arms import ARMS, SKILL, control, parse_command, system
```

In `tests/test_llm.py`, change `FakeClient.create` to return `choices=[SimpleNamespace(message=SimpleNamespace(content="hi"), finish_reason="length")]`, and add:
```python
def test_replies_carry_the_finish_reason(tmp_path):
    llm = LLM("deepseek-flash", FakeClient(), tmp_path, 0.3, PRICE)
    assert llm.chat([{"role": "user", "content": "q"}], 64, repeat=0)["finish_reason"] == "length"
```

In `tests/test_mab.py`, add `gold_fact` to the `cacheclm.mab` import, and add:
```python
def test_gold_fact_is_the_latest_matching_fact_about_the_question():
    context = ("1. The author of Hard Times is Charles Dickens.\n"
               "2. Martin Luther King Jr. was born in Atlanta.\n"
               "9. The author of Hard Times is Martin Luther King Jr.\n")
    golds = ["Martin Luther King Jr."]
    assert gold_fact(context, "Who is the author of Hard Times?", golds) == "9. The author of Hard Times is Martin Luther King Jr."
    assert gold_fact(context, "Where is Paris?", ["Lyon"]) is None
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest -q 2>&1 | tail -15`
Expected: collection errors:
- `ImportError` for `cut_oldest_lines` / `normalize` (test_ctxfile), `SKILL` (test_run) and `gold_fact` (test_mab);
- `test_replies_carry_the_finish_reason` fails with `KeyError: 'finish_reason'`.

- [ ] **Step 3: Rewrite `src/cacheclm/ctxfile.py`**

```python
"""The working context: a pinned task block (kept apart, never in ctx.txt) and an editable body of blocks in the
paper's [[CTX_TURN i role=...]] format. The model edits only the body."""
import re

CHARS_PER_TOKEN = 4  # local estimate; the gate's inequality is scale-free and billing uses provider counts
HEADER = re.compile(r"^\[\[CTX_TURN (\d+) role=([\w-]+)( pinned)?\]\]$", re.M)
PINNED = re.compile(r"^(\[\[CTX_TURN \d+ role=[\w-]+) pinned\]\]$", re.M)


def tokens(text):
    return len(text) // CHARS_PER_TOKEN


def block(i, role, text, pinned=False):
    return f"[[CTX_TURN {i} role={role}{' pinned' if pinned else ''}]]\n{text.rstrip()}\n"


def new_context(task):
    """The pinned task block; it is sent before the body in every request but never written to ctx.txt."""
    return block(0, "task", task, pinned=True)


def blocks(text):
    """[(start, end, index, role)] for each block, in order."""
    heads = list(HEADER.finditer(text))
    ends = [h.start() for h in heads[1:]] + [len(text)]
    return [(h.start(), end, int(h.group(1)), h.group(2)) for h, end in zip(heads, ends)]


def next_index(body):
    """Body blocks are numbered from 1; the task block is 0."""
    return max([b[2] for b in blocks(body)] + [0]) + 1


def append(body, role, text):
    sep = "" if not body or body.endswith("\n") else "\n"
    return body + sep + block(next_index(body), role, text)


def normalize(body, role="notes"):
    """The body without model-written 'pinned' marks, starting with a header (loose leading text gets one)."""
    body = PINNED.sub(r"\1]]", body).lstrip("\n")
    if body and not HEADER.match(body):
        body = f"[[CTX_TURN {next_index(body)} role={role}]]\n" + body
    return body


def cut_oldest_lines(body, chars):
    """The body without its oldest whole lines (at least `chars` characters of them), re-headed as a chunk."""
    lines = body.splitlines(keepends=True)
    cut = 0
    while lines and cut < chars:
        cut += len(lines.pop(0))
    return normalize("".join(lines), "chunk")


def split_for_summary(body, keep):
    """(older, tail): everything before the last `keep` chunk blocks, and those blocks; None if nothing is older."""
    chunks = [b for b in blocks(body) if b[3] == "chunk"]
    if len(chunks) <= keep or not body[:chunks[-keep][0]].strip():
        return None
    start = chunks[-keep][0]
    return body[:start], body[start:]
```

- [ ] **Step 4: Rewrite `src/cacheclm/arms.py`**

````python
"""Context policies. summary: the harness compacts at 75%. clm: the model edits ctx.txt with bash (prompt A). gate: clm
plus the cache-aware gate. skill: clm plus a context-management skill. chat(messages, max_tokens, phase) -> reply."""
import re

from cacheclm import sandbox
from cacheclm.ctxfile import CHARS_PER_TOKEN, block, cut_oldest_lines, next_index, normalize, split_for_summary, tokens
from cacheclm.gate import decide

ARMS = ("summary", "clm", "gate", "skill")
REFERENCES = ("none", "full")  # floor (no context) and ceiling (whole text), outside the primary endpoints
UNGATED = ("clm", "skill")
BASE = "You are a helpful assistant that can read the context and memorize it for future retrieval."
EDITING = """
Your working context is shown in each message: first the task block ([[CTX_TURN 0 role=task pinned]]), which is fixed and is not in any file, then the file ctx.txt, which holds every block after it. Together they hold at most {budget} tokens. Parts of a long text arrive one at a time and are appended to the end of ctx.txt. Between parts you may edit ctx.txt to keep what will matter later: delete or shorten text, or keep notes in a block of your own such as [[CTX_TURN 99 role=notes]].
How to edit:
- Start your reply with a short THOUGHT about what to keep and why, then give exactly one shell command in a ```bash block.
- An edit makes everything after the edited point be re-read, so prefer one large edit over several small ones, and be generous in what you keep.
- The whole file is already shown to you: do not run commands that only look at it.
- Keep the [[CTX_TURN ...]] header line of every block you keep.
- Allowed programs: sed, awk, grep, head, tail, cat, wc, echo, printf, mv, cp, python3. No heredocs or $(...); for multi-step edits use python3 -c "..." (the script may span several lines; on this system, in-place sed is sed -i '').
- Reply READY when you are done editing."""
SKILL = """

# Skill: managing your context
## Book excerpts (questions ask which event comes next, anywhere in the story)
- Right after a new part arrives, replace it with an event log: one line per event, in story order.
- Each line keeps who did, said, felt or wore what, to or with whom, and where, with exact names and specific details (objects, colours, family relations).
- Keep every older event log.
## Numbered fact lists (newer facts override older ones)
- When the context is nearly full, in one python3 edit: delete every fact for which a later fact has the same subject and relation; then, only if still needed, delete the oldest facts."""
SUMMARY_NOTE = "\nParts of a long text are appended to your working context. When it gets full you will be asked to compact older parts into a summary."
SUMMARIZE = ("The working context is nearly full. Write one summary that replaces every block after the pinned task block "
             "and before the last {keep} parts, in at most {words:,} words. Keep the details most likely to be needed "
             "for later questions: names, events in story order, and facts with their serial numbers. Reply with only "
             "the summary text.")
CUT_OFF = "Your last reply was cut off before the command ended; send a shorter one."
FENCE = re.compile(r"```(?:bash|sh)?\n(.*?)```", re.S)


def system(arm, budget, repeat=0):
    """The arm's fixed system prompt; the first line keeps repeats from sharing the provider's prompt cache."""
    editing = EDITING.format(budget=budget)
    extra = {"summary": SUMMARY_NOTE, "clm": editing, "gate": editing, "skill": editing + SKILL}.get(arm, "")
    return f"Run r{repeat}.\n" + BASE + extra


def parse_command(reply):
    """The command in the reply's fenced block, or None (READY, or no complete block)."""
    m = FENCE.search(reply or "")
    return m.group(1).strip() if m else None


def control(ctx, budget, incoming, nudges, last):
    """The note after the context: budget use, a nudge or the overflow warning, the last command's result."""
    used = tokens(ctx)
    lines = [f"Context: {used:,} of {budget:,} tokens ({used / budget:.0%}). Next part: {incoming:,} tokens."]
    over = used + incoming - budget
    crossed = [n for n in nudges if used >= n * budget]
    if over > 0:
        lines.append(f"OVER LIMIT: free at least {over:,} tokens before the next part arrives.")
    elif crossed:
        lines.append(f"Your context is over {max(crossed):.0%} full.")
    if last:
        lines.append(f"Result of your last command:\n{last}")
    lines.append("Reply with one shell command in a ```bash block to edit ctx.txt, or READY.")
    return "\n".join(lines)


def edit_phase(task, body, arm, chat, cfg, incoming, turns_left, price, log, repeat=0):
    """The body after up to max_edits commands, plus up to max_condense more while the next part would not fit."""
    budget, last, edits = cfg["context_budget"], "", 0
    while True:
        ctx = task + body
        over = tokens(ctx) + incoming > budget
        if edits >= cfg["max_edits_per_chunk"] + (cfg["max_condense_tries"] if over else 0):
            return body
        note = control(ctx, budget, incoming, cfg["nudges"], last)
        messages = [{"role": "system", "content": system(arm, budget, repeat)},
                    {"role": "user", "content": ctx + "\n\n" + note}]
        reply = chat(messages, cfg["edit_max_tokens"], "edit")
        command = parse_command(reply["content"])
        if command is None and reply.get("finish_reason") == "length":
            edits += 1
            log({"event": "cut_off"})
            last = CUT_OFF
            continue
        if command is None:
            return body
        edits += 1
        new, output = sandbox.run(command, body)
        new = normalize(new)
        emptied = bool(body.strip()) and not new.strip()
        if emptied:
            allow, reason, numbers = False, "edit rolled back: it emptied ctx.txt", {}
        else:
            allow, reason, numbers = decide(ctx, task + new, turns_left, price, overflow=over)
            allow = allow or arm in UNGATED  # clm and skill log the gate's verdict but are never stopped by it
        log({"event": "edit", "command": command, "allowed": allow, "reason": reason, "over": over,
             "changed": new != body, "refused": output.startswith("REFUSED"), "emptied": emptied,
             "turns_left": turns_left, "chars_before": len(ctx), "chars_after": len(task + new), **numbers})
        if allow:
            body = new
        last = output if allow else f"{output}\n{reason}".strip()  # the gate explains itself only when it rejects


def summary_step(task, body, chat, cfg, incoming, log, repeat=0):
    """The body with older blocks compacted into one summary once the context plus the next part passes summary_at."""
    budget = cfg["context_budget"]
    if tokens(task + body) + incoming <= cfg["summary_at"] * budget:
        return body
    parts = split_for_summary(body, cfg["keep_recent_chunks"])
    if parts is None:
        return body
    older, tail = parts
    prompt = SUMMARIZE.format(keep=cfg["keep_recent_chunks"], words=cfg["summary_words"])
    messages = [{"role": "system", "content": system("summary", budget, repeat)},
                {"role": "user", "content": task + body + "\n\n" + prompt}]
    reply = chat(messages, cfg["summary_max_tokens"], "summary")
    log({"event": "summary", "chars_compacted": len(older), "summary_chars": len(reply["content"]),
         "cut_off": reply.get("finish_reason") == "length"})
    return block(next_index(body), "summary", reply["content"]) + tail


def fit(task, body, incoming, budget, log):
    """Cut the oldest whole lines of the body until the task, the body and the next part fit (forced truncation)."""
    over = (tokens(task + body) + incoming - budget) * CHARS_PER_TOKEN
    while over > 0 and body:
        smaller = cut_oldest_lines(body, over)
        log({"event": "forced_truncation", "chars_dropped": len(body) - len(smaller)})
        body = smaller
        over = (tokens(task + body) + incoming - budget) * CHARS_PER_TOKEN
    return body
````

- [ ] **Step 5: Update `src/cacheclm/llm.py`**

In `_call`, add `"finish_reason": resp.choices[0].finish_reason,` to the returned dict, directly after `"content": ...`.

- [ ] **Step 6: Add `gold_fact` to `src/cacheclm/mab.py`** (after `correct`)

```python
def gold_fact(context, question, golds):
    """The latest numbered fact containing a gold answer and sharing the most words with the question, or None."""
    words = set(normalize_answer(question).split())
    best = None
    for line in context.splitlines():
        if re.match(r"^\d+\. ", line) and any(normalize_answer(g) in normalize_answer(line) for g in golds):
            overlap = len(words & set(normalize_answer(line).split()))
            if overlap and (best is None or overlap >= best[0]):
                best = (overlap, line)
    return best[1] if best else None
```

- [ ] **Step 7: Rewrite `src/cacheclm/run.py`**

```python
"""Runs one sample through one arm: stream the chunks under the arm's context policy (or none / the full text for the
reference arms), then ask every question on the frozen context. Calls are logged with billed and ideal cache hits."""
import datetime
import hashlib
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cacheclm.arms import edit_phase, fit, summary_step, system
from cacheclm.ctxfile import CHARS_PER_TOKEN, append, new_context, tokens
from cacheclm.files import append_jsonl, read_jsonl
from cacheclm.mab import MEMORIZE, QUERY, TASK, chunk_text, correct, family, gold_fact

ANSWER_NOTE = "Editing is over. Answer the question below in plain text; do not reply READY or with a command."


def run_log(runs_dir, arm, sample, repeat):
    return Path(runs_dir) / arm / f"{re.sub(r'[^\w.-]', '_', sample.sid)}_r{repeat}.jsonl"


def unit_id(sample):
    """Samples with the same text form one unit (FactConsolidation sh and mh share theirs)."""
    return hashlib.sha1(sample.context.encode()).hexdigest()[:10]


class Recorder:
    def __init__(self, path, llm, repeat, meta):
        self.path, self.llm, self.repeat, self.meta = path, llm, repeat, meta
        self.ref, self.lock = "", threading.Lock()

    def log(self, record):
        with self.lock:
            append_jsonl(self.path, {**self.meta, **record})

    def chat(self, messages, max_tokens, phase, ref=None):
        """The reply; ref fixes the ideal-cache reference (query phase), otherwise it is the previous request."""
        text = "".join(m["content"] for m in messages)
        reference = self.ref if ref is None else ref
        estimate = len(text) // CHARS_PER_TOKEN  # logged next to billed prompt_tokens to measure the estimate's gap
        shared = len(os.path.commonprefix([reference, text]))
        reply = self.llm.chat(messages, max_tokens, self.repeat)
        ideal = round(reply["prompt_tokens"] * shared / len(text)) if text else 0  # shared share, in billed tokens
        if ref is None:
            self.ref = text
        self.log({"event": "call", "phase": phase, "model": reply["model"], "date": datetime.date.today().isoformat(),
                  "prompt_tokens": reply["prompt_tokens"], "cache_hit_tokens": reply["cache_hit_tokens"],
                  "completion_tokens": reply["completion_tokens"], "latency_s": reply["latency_s"],
                  "cached": reply["cached"], "ideal_hit_tokens": min(ideal, reply["prompt_tokens"]),
                  "estimated_tokens": estimate, "finish_reason": reply.get("finish_reason")})
        return reply


def answer_all(ctx, arm, fam, questions, rec, cfg, serial_max=3):
    """Each question in its own call on the frozen context. The first is compared with the real previous request;
    questions go one at a time until the provider reports a cache hit (at most serial_max), then in parallel."""
    head = system(arm, cfg["context_budget"], rec.repeat)

    def ask(q, ref):
        messages = [{"role": "system", "content": head},
                    {"role": "user", "content": ctx + "\n\n" + ANSWER_NOTE + "\n\n" + QUERY[fam].format(question=q)}]
        return rec.chat(messages, 256, "query", ref=ref)

    replies = [ask(questions[0], None)]
    while len(replies) < min(serial_max, len(questions)) and not (replies[-1]["cache_hit_tokens"] or
                                                                  replies[-1]["cached"]):
        replies.append(ask(questions[len(replies)], head + ctx))
    with ThreadPoolExecutor(cfg["query_workers"]) as pool:
        replies += list(pool.map(lambda q: ask(q, head + ctx), questions[len(replies):]))
    return [r["content"] for r in replies]


def stream(task, chunks, arm, fam, n_questions, rec, cfg, price, repeat):
    """The body after every chunk has streamed through the arm's context policy."""
    body, budget = "", cfg["context_budget"]
    for i, chunk in enumerate(chunks + [None]):  # None: one last edit phase after the final part
        text = MEMORIZE[fam].format(chunk=chunk) if chunk is not None else ""
        incoming = tokens(text) + 8 if chunk is not None else 0  # +8 for the block header
        turns_left = len(chunks) - i + n_questions
        if arm == "summary":
            if chunk is not None:
                body = summary_step(task, body, rec.chat, cfg, incoming, rec.log, repeat)
        else:
            body = edit_phase(task, body, arm, rec.chat, cfg, incoming, turns_left, price, rec.log, repeat)
        body = fit(task, body, incoming, budget, rec.log)
        if chunk is not None:
            body = append(body, "chunk", text)
        rec.log({"event": "step", "chunk": i, "ctx_tokens": tokens(task + body)})
    return body


def run_sample(sample, arm, repeat, llm, cfg, price, runs_dir, question_limit=None):
    """Accuracy of one (sample, arm, repeat); a log that already ends with a done record is not rerun."""
    path = run_log(runs_dir, arm, sample, repeat)
    done = [r for r in read_jsonl(path) if r.get("event") == "done"]
    if done:
        return done[-1]["accuracy"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)  # a partial run restarts; its calls replay from the cache for free
    rec = Recorder(path, llm, repeat, {"arm": arm, "sample": sample.sid, "source": sample.source, "repeat": repeat})
    fam = family(sample.source)
    questions, golds = sample.questions[:question_limit], sample.answers[:question_limit]
    chunks = chunk_text(sample.context, cfg["chunk_tokens"] * CHARS_PER_TOKEN)
    task = new_context(TASK[fam].format(n=len(questions)))
    body = ""
    if arm == "full":
        for chunk in chunks:
            body = append(body, "chunk", MEMORIZE[fam].format(chunk=chunk))
    elif arm != "none":
        body = stream(task, chunks, arm, fam, len(questions), rec, cfg, price, repeat)
    ctx = task + body
    predictions = answer_all(ctx, arm, fam, questions, rec, cfg)
    results = [correct(p, g) for p, g in zip(predictions, golds)]
    for qi, (p, ok) in enumerate(zip(predictions, results)):
        fact = gold_fact(sample.context, questions[qi], golds[qi]) if fam == "factconsolidation" else None
        rec.log({"event": "answer", "qi": qi, "prediction": p, "correct": ok,
                 "gold_fact_present": (fact in ctx) if fact else None})
    accuracy = sum(results) / len(results)
    rec.log({"event": "done", "accuracy": accuracy, "n_questions": len(results), "ctx_tokens_final": tokens(ctx),
             "unit": unit_id(sample), "family": fam, "final_context": ctx})
    return accuracy
```

- [ ] **Step 8: Update `configs/base.yaml`**
- Replace the `summary_words` line with `summary_words: 1500         # ~2K tokens; the 3,000-word target still hit the cap in smoke 2`.
- Add `edit_max_tokens: 2048` after `max_condense_tries: 3`.

- [ ] **Step 9: Run all tests**

Run: `uv run pytest -q 2>&1 | tail -5`
Expected: all pass. If an existing test in `test_run.py` that calls `run_sample` fails because of a changed internal assumption, fix the test helper (not the spec behaviour) and record a `Ruling:` in the ledger.

- [ ] **Step 10: Commit**

```bash
git add src/cacheclm configs/base.yaml tests
git commit -m "feat: task kept apart from the editable body, line-level truncation, prompt A, skill arm, gate feedback on rejection only"
```

---

### Task 2: Reference arms, unit and family reporting, extra repeat, smoke samples

**Files:**
- Rewrite: `src/cacheclm/report.py`, `src/cacheclm/__main__.py`
- Modify: `configs/base.yaml`
- Test: `tests/test_report.py` (update), `tests/test_main.py` (new)

**Interfaces:**
- **Consumes, from Task 1:**
  - the done record has `unit`, `family`, `final_context`, `n_questions`, `accuracy`, `repeat`, `arm`, `sample`;
  - call records have `phase` (`edit` / `summary` / `query`), with `cut_off` and `summary` events;
  - answer records have `gold_fact_present`;
  - `arms.ARMS`, `arms.REFERENCES`; `mab.Sample`, `mab.load_sample`.
- **Produces:**
  - `report.call_cost(call, price, hit_key)`, `report.paired`, `report.kept_share` (unchanged);
  - `report.load_units(runs_dir, prices) -> (means, refs, info, examples, incomplete)`;
  - `report.write_report(runs_dir, out_dir, prices)`;
  - `__main__.jobs(cfg, smoke) -> [(spec, repeat, arms)]`, `__main__.load(cfg, spec) -> Sample`.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_report.py` entirely:
```python
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
```

Create `tests/test_main.py`:
```python
from cacheclm.__main__ import jobs
from cacheclm.arms import ARMS, REFERENCES

CFG = {"repeats": [0], "samples": [{"split": "A", "row": 1}, {"split": "B", "row": 2}],
       "extra_runs": [{"split": "B", "row": 2, "repeat": 1}],
       "smoke": {"samples": [{"split": "C", "row": 1, "questions": 100, "context_budget": 12000}]}}


def test_full_run_jobs_add_references_once_and_the_extra_repeat():
    planned = jobs(CFG, smoke=False)
    assert [(s["split"], r, arms) for s, r, arms in planned] == [
        ("A", 0, ARMS + REFERENCES), ("B", 0, ARMS + REFERENCES), ("B", 1, ARMS)]


def test_smoke_jobs_run_the_primary_arms_only():
    assert [(s["split"], r, arms) for s, r, arms in jobs(CFG, smoke=True)] == [("C", 0, ARMS)]
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_report.py tests/test_main.py -q 2>&1 | tail -5`
Expected: `ImportError: cannot import name 'load_units'` and `ImportError: cannot import name 'jobs'`.

- [ ] **Step 3: Rewrite `src/cacheclm/report.py`**

```python
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
    """One unit-arm-repeat row: streaming parts once (the unit's samples replay the same stream), questions summed."""
    out = {}
    for m in rows[0]:
        values = [r[m] for r in rows]
        out[m] = sum(values) if m in SUMMED or m.endswith("_query") else max(values)
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
          f"Unpaired runs (left out): {', '.join(incomplete) or 'none'}."]
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
```

- [ ] **Step 4: Rewrite `src/cacheclm/__main__.py`**

```python
"""CLI: `cacheclm smoke` (the smoke samples, 4 arms), `cacheclm run` (the samples, 4 arms plus the reference arms, and
the extra FactConsolidation repeat), `cacheclm report [--smoke]`. Settings in configs/base.yaml; DEEPSEEK_API_KEY from .env."""
import argparse
import os

import openai
import yaml
from dotenv import find_dotenv, load_dotenv

from cacheclm.arms import ARMS, REFERENCES
from cacheclm.budget import Budget, BudgetExceeded
from cacheclm.llm import LLM
from cacheclm.mab import Sample, load_sample
from cacheclm.report import write_report
from cacheclm.run import run_sample


def make_llm(cfg, price, budget):
    agent = cfg["agent"]
    client = openai.OpenAI(base_url=agent["base_url"], api_key=os.environ[agent["api_key_env"]], max_retries=0,
                           timeout=300)
    return LLM(agent["model"], client, cfg["cache_dir"], cfg["temperature"], price, agent.get("options"), budget.spend)


def jobs(cfg, smoke):
    """[(sample spec, repeat, arms)]: reference arms run once per full-run sample, never in the smoke or extra repeats."""
    if smoke:
        return [(s, 0, ARMS) for s in cfg["smoke"]["samples"]]
    planned = [(s, r, ARMS + (REFERENCES if r == 0 else ())) for r in cfg["repeats"] for s in cfg["samples"]]
    return planned + [(s, s["repeat"], ARMS) for s in cfg.get("extra_runs", [])]


def load(cfg, spec):
    """The sample a spec names, cut to [start:end] when given (the EventQA smoke uses text no evaluated row covers)."""
    sample = load_sample(cfg["data_dir"], spec["split"], spec["row"])
    if "start" in spec:
        sample = Sample(f"{sample.sid}[{spec['start']}:{spec['end']}]", sample.source,
                        sample.context[spec["start"]:spec["end"]], sample.questions, sample.answers)
    return sample


def main():
    parser = argparse.ArgumentParser(prog="cacheclm")
    parser.add_argument("command", choices=["smoke", "run", "report"])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--smoke", action="store_true", help="report on the smoke run")
    args = parser.parse_args()
    load_dotenv(find_dotenv(usecwd=True))
    cfg = yaml.safe_load(open(args.config))
    prices = yaml.safe_load(open(cfg["prices"]))
    smoke = args.command == "smoke" or args.smoke
    runs_dir = cfg["smoke"]["runs_dir"] if smoke else cfg["runs_dir"]
    if args.command == "report":
        write_report(runs_dir, cfg["smoke"]["results_dir"] if smoke else cfg["results_dir"], prices)
        return
    budget = Budget(cfg["budget_file"], cfg["budget_usd"])
    llm = make_llm(cfg, prices["deepseek"], budget)
    try:
        for spec, repeat, arms in jobs(cfg, smoke):
            sample = load(cfg, spec)
            run_cfg = {**cfg, "context_budget": spec.get("context_budget", cfg["context_budget"])}
            for arm in arms:
                try:
                    acc = run_sample(sample, arm, repeat, llm, run_cfg, prices["deepseek"], runs_dir, spec.get("questions"))
                except RuntimeError as e:  # API kept failing after retries: skip, the report lists it as incomplete
                    print(f"FAILED   {arm:8s} {sample.sid:28s} r{repeat}: {e}")
                    continue
                print(f"{arm:8s} {sample.sid:28s} r{repeat}  accuracy {acc:.2f}  total spend ${budget.total:.3f}")
    except BudgetExceeded as e:
        print(f"Stopped: {e}. Finished runs are kept; raise budget_usd in the config to continue.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Update `configs/base.yaml`**
- Replace the whole `smoke:` section with:
```yaml
# Smoke runs use data that is not evaluated. FactConsolidation mh_32k (row 1) gets a 12K budget so editing is needed,
# and all 100 questions so the gate sees a realistic turns_left. EventQA uses eventqa_full text beyond what rows 7-11 cover.
smoke:
  samples:
    - {split: Conflict_Resolution, row: 1, questions: 100, context_budget: 12000}
    - {split: Accurate_Retrieval, row: 2, start: 285000, end: 570000, questions: 5}
  runs_dir: runs_smoke
  results_dir: results_smoke
```
- Add after `samples:`:
```yaml
# A second repeat of the FactConsolidation unit (rows 6 and 2 share one text) to measure run-to-run noise.
extra_runs:
  - {split: Conflict_Resolution, row: 6, repeat: 1}
  - {split: Conflict_Resolution, row: 2, repeat: 1}
```

- [ ] **Step 6: Run all tests**

Run: `uv run pytest -q 2>&1 | tail -5`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/cacheclm/report.py src/cacheclm/__main__.py configs/base.yaml tests/test_report.py tests/test_main.py
git commit -m "feat: reference arms, unit and family reporting with the fact error split, extra repeat and new smoke samples"
```

---

### Task 3: README, final review and the smoke run

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update `README.md`**
- **The arms section** lists 4 arms (summary, clm, gate, skill) and the 2 reference arms (none, full), with one line each. It says the `skill` arm is exploratory and mostly tail-only editing.
- **Add a "Units" paragraph:** FactConsolidation rows 2 and 6 share one text, so they count as one unit with 200 questions and a second repeat.
- **Replace the smoke description** with the two smoke samples.
- **Cost:** full run about $3.4-3.9, plus about $0.4 for the smoke runs.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README for revision 2 (four arms, reference arms, units, smoke samples)"
```

- [ ] **Step 3: Final whole-branch review.** Use executing-plans' final review on the range from `d8b9dcc` to HEAD, against the spec's Revision 2 and this plan's Review Focus. Fix Critical and Important findings test-first.

- [ ] **Step 4: Archive the old smoke runs and run the smoke**

```bash
cd /Users/hvinnakota/Downloads/new-techniques/cacheclm && mv runs_smoke runs_smoke_v2 && mv results_smoke results_smoke_v2
set -a && . ../fastlane/.env && set +a && caffeinate -i bash -c 'uv run cacheclm smoke && uv run cacheclm report --smoke' > cache/smoke_v3.log 2>&1
```
- **How to run it:** in the background. It should take about 15-25 minutes and cost about $0.4.
- **What to check afterwards:** in `results_smoke/summary.md`:
  - CLM edits applied without forced truncation of the newest data;
  - the skill arm writes event logs (EventQA) and dedupes facts (FactConsolidation);
  - the gate rejects mid-context edits;
  - few summaries and edit replies are cut off;
  - the projected full-run cost.
- **Do not start the full run.**
