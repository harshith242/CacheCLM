# CacheCLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether model-edited context (CLM) pays off in billed dollars on a hosted API with prompt caching, against a summary baseline and a cache-aware edit gate, on 7 MemoryAgentBench samples.

**Architecture:**
- **Context:** one text in the paper's `[[CTX_TURN i role=…]]` format, sent as a single user message after a fixed per-arm system prompt, so the context is the cacheable prefix.
- **Edits:** CLM arms edit it with one-line bash commands inside a macOS `sandbox-exec` jail. The gate arm accepts an edit only if `deleted × turns_left × hit ≥ re-billed survivors × (miss − hit)`.
- **Logging:** every call is logged with billed usage and an "ideal" cache hit, so the report can reprice the same runs under DeepSeek, OpenAI and Anthropic price sheets for free.

**Tech Stack:** Python 3.13, uv, `openai` client (DeepSeek), `pyarrow`, `numpy`, `matplotlib`, `markdown`, `pyyaml`, `python-dotenv`, `pytest`. No `litellm`: its PyPI release was compromised in 2026.

**Spec:** `docs/specs/2026-10-05-cacheclm-design.md`.

**Working rules for this repo:**
- Simple code; at most one-line comments, plus a file docstring of up to 3 lines.
- Tests target real logic, with no network.
- Commits have no co-author or AI-attribution lines.
- Never run live LLM commands yourself. Hand the user the command.

**Reviews:** milestone code reviews after Task 3 and after Task 4 only.

---

## File structure

```text
cacheclm/
  pyproject.toml, .gitignore, LICENSE, README.md
  configs/base.yaml            run settings, the 7 samples, smoke settings
  configs/prices.yaml          USD/1M token prices of 3 providers (dated)
  scripts/get_data.py          pinned download + SHA-256 check
  src/cacheclm/
    __init__.py
    __main__.py                CLI: smoke | run | report
    files.py                   atomic writes, JSONL
    mab.py                     MemoryAgentBench: load, chunk, prompts, scoring
    ctxfile.py                 the context text: blocks, append, drop, split, pinned check
    gate.py                    first_change, survivors, decide
    sandbox.py                 command allow-list + sandbox-exec runner
    llm.py                     cached OpenAI-compatible client, cost
    budget.py                  persisted spend cap
    arms.py                    prompts, edit_phase, summary_step, fit
    run.py                     Recorder, run_sample, answer_all
    report.py                  load runs, stats, repricing, chart, summary.md/html
  tests/ test_mab.py test_ctxfile.py test_gate.py test_sandbox.py test_llm.py test_run.py test_report.py
  data/memoryagentbench/       (git-ignored) the two parquet files, already downloaded
```

---

### Task 1: Project scaffold, data access and scoring

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `LICENSE`, `configs/base.yaml`, `configs/prices.yaml`, `scripts/get_data.py`, `src/cacheclm/__init__.py`, `src/cacheclm/files.py`, `src/cacheclm/mab.py`
- Test: `tests/test_mab.py`

- [ ] **Step 1: Scaffold the project files**

`pyproject.toml`:
```toml
[project]
name = "cacheclm"
version = "0.1.0"
description = "Does model-edited context pay off under hosted-API prompt caching? CLM vs summary vs a cache-aware gate"
requires-python = ">=3.13"
dependencies = [
    "markdown>=3.11",
    "matplotlib>=3.10",
    "numpy>=2.5.3",
    "openai>=3.19.2",
    "pyarrow>=20",
    "python-dotenv>=1.2.3",
    "pyyaml>=6.0.3",
    "tqdm>=4.70.1",
]

[project.scripts]
cacheclm = "cacheclm.__main__:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[dependency-groups]
dev = ["pytest>=9.1.1"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.uv]
exclude-newer = "2026-09-01T00:00:00Z"  # supply-chain guard: no packages published after this date
```

`.gitignore`:
```text
.env
data/
cache/
runs*/
results_smoke/
__pycache__/
.venv/
.pytest_cache/
```

`LICENSE`: the standard MIT license text, `Copyright (c) 2026 harshith242`.

`src/cacheclm/__init__.py`:
```python
"""CacheCLM: model-edited context under hosted-API prompt caching."""
```

`configs/base.yaml`:
```yaml
# CacheCLM settings (spec docs/specs/2026-10-05-cacheclm-design.md).
data_dir: data/memoryagentbench
cache_dir: cache/llm
budget_file: cache/spend.json
prices: configs/prices.yaml
runs_dir: runs
results_dir: results
budget_usd: 5.0
context_budget: 32000       # tokens, as in the paper
chunk_tokens: 4000
temperature: 0.3
repeats: [0, 1]
max_edits_per_chunk: 3
max_condense_tries: 3
summary_at: 0.75
keep_recent_chunks: 2
nudges: [0.25, 0.5, 0.75]
query_workers: 8
agent:
  model: deepseek-flash
  base_url: https://api.deepseek.com
  api_key_env: DEEPSEEK_API_KEY
  options: {extra_body: {thinking: {type: disabled}}}
# Chosen by length before any run: the 5 EventQA books at ~140K tokens and both FactConsolidation 64K sets.
samples:
  - {split: Accurate_Retrieval, row: 12}
  - {split: Accurate_Retrieval, row: 13}
  - {split: Accurate_Retrieval, row: 14}
  - {split: Accurate_Retrieval, row: 15}
  - {split: Accurate_Retrieval, row: 16}
  - {split: Conflict_Resolution, row: 2}
  - {split: Conflict_Resolution, row: 6}
smoke:
  sample: {split: Conflict_Resolution, row: 6}
  questions: 5
  runs_dir: runs_smoke
  results_dir: results_smoke
```

`configs/prices.yaml`:
```yaml
# USD per 1M tokens, list prices noted 2026-10-05; re-check the provider pages before publishing results.
# cache_read = cached prompt tokens, cache_write = uncached prompt tokens (Anthropic bills these as cache writes).
deepseek:  {cache_read: 0.006, cache_write: 0.30, output: 1.20}
openai:    {cache_read: 0.175, cache_write: 1.75, output: 14.00}
anthropic: {cache_read: 0.30,  cache_write: 3.75, output: 15.00}
```

- [ ] **Step 2: Write `files.py`** (copied from EvoSQL)

```python
"""Small file helpers: atomic writes and JSONL logs."""
import json
import os
from pathlib import Path


def write_atomic(path, text):
    """Write via a temp file and rename, so a kill never leaves a truncated file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def read_jsonl(path):
    """Records of a JSONL log; a torn last line (killed mid-write) is ignored."""
    records = []
    for line in Path(path).read_text().splitlines() if Path(path).exists() else []:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
```

- [ ] **Step 3: Write the failing tests** in `tests/test_mab.py`

```python
from pathlib import Path

import pytest

from cacheclm.mab import check_file, chunk_text, correct, family, load_sample, normalize_answer

DATA = Path("data/memoryagentbench")


def test_chunks_rejoin_and_respect_the_limit():
    text = "".join(f"{i}. fact number {i}.\n" for i in range(200))
    chunks = chunk_text(text, 100)
    assert "".join(chunks) == text
    assert all(len(c) <= 100 for c in chunks)
    assert all(c.endswith("\n") for c in chunks)  # short lines are never split


def test_an_overlong_line_is_cut_hard():
    chunks = chunk_text("a" * 250 + "\nb\n", 100)
    assert "".join(chunks) == "a" * 250 + "\nb\n"
    assert [len(c) for c in chunks][:3] == [100, 100, 50]


def test_scoring_matches_the_benchmark():
    assert normalize_answer("The  Ancient-Greek!") == "ancientgreek"
    assert correct("It was written in Ancient Greek.", ["Ancient Greek"])
    assert correct("answer: France", ["Germany", "France"])
    assert not correct("Paris", ["France"])
    assert not correct(None, ["France"])


def test_family():
    assert family("eventqa_131072") == "eventqa"
    assert family("factconsolidation_sh_64k") == "factconsolidation"


def test_a_tampered_file_is_refused(tmp_path):
    fake = tmp_path / "Conflict_Resolution.parquet"
    fake.write_bytes(b"not the pinned file")
    with pytest.raises(ValueError, match="SHA-256"):
        check_file(fake, "Conflict_Resolution")


@pytest.mark.skipif(not DATA.exists(), reason="run scripts/get_data.py first")
def test_the_configured_samples_load():
    event = load_sample(DATA, "Accurate_Retrieval", 12)
    facts = load_sample(DATA, "Conflict_Resolution", 6)
    assert event.source == "eventqa_131072" and facts.source == "factconsolidation_sh_64k"
    assert len(event.questions) == len(event.answers) == 100
    assert facts.answers[0] == ["Ancient Greek"]
```

- [ ] **Step 4: Run the tests to see them fail**

Run: `uv run pytest tests/test_mab.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cacheclm.mab'`.

- [ ] **Step 5: Write `mab.py`**

```python
"""MemoryAgentBench samples: pinned parquet rows, chunking, prompts adapted from the benchmark's templates.py, and its scoring."""
import hashlib
import re
import string
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

URL = "https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/7ea0669/data/{split}-00000-of-00001.parquet"
SHA256 = {
    "Accurate_Retrieval": "56c3cd80fb6731a3e53cd1a6be3148f54df60ff2d290ee50e28f8acebf9655c1",
    "Conflict_Resolution": "24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45",
}

# Pinned task block (ours): what will be asked, never the questions themselves.
TASK = {
    "eventqa": "You are reading a book excerpt in parts. After the last part you will answer {n} questions, each asking "
               "which event happens next at some point in the story. You cannot see the questions now.",
    "factconsolidation": "You are reading a numbered list of facts in parts. After the last part you will answer {n} "
                         "questions about these facts. When facts conflict, the fact with the larger serial number is "
                         "the newest and the correct one. You cannot see the questions now.",
}
MEMORIZE = {
    "eventqa": "The following context is the book excerpt:\n{chunk}",
    "factconsolidation": "The following context is the facts I have learned:\n{chunk}",
}
QUERY = {
    "eventqa": "Based on the context you memorized, complete the task below:\n\n{question}\n\n The event that happens next is:",
    "factconsolidation": "Based on the facts you memorized, answer the question below. Solve the conflicts of facts in "
                         "the knowledge pool by finding the newest fact with larger serial number. Answer without saying "
                         "other words, only from the knowledge pool you have memorized.\n\nQuestion: {question}\nAnswer:",
}


@dataclass
class Sample:
    sid: str  # "<split>/<row>"
    source: str
    context: str
    questions: list
    answers: list  # one list of gold answers per question


def family(source):
    return "eventqa" if source.startswith("eventqa") else "factconsolidation"


def check_file(path, split):
    """Raise unless the file has the pinned SHA-256."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != SHA256[split]:
        raise ValueError(f"{path}: SHA-256 {digest} is not the pinned {SHA256[split]}")


def load_sample(data_dir, split, row):
    path = Path(data_dir) / f"{split}.parquet"
    check_file(path, split)
    r = pq.read_table(path).slice(row, 1).to_pylist()[0]
    return Sample(f"{split}/{row}", r["metadata"]["source"], r["context"], list(r["questions"]),
                  [list(a) for a in r["answers"]])


def chunk_text(text, max_chars):
    """Split on line breaks into chunks of at most max_chars."""
    chunks, cur = [], ""
    for line in text.splitlines(keepends=True):
        while len(line) > max_chars:  # an over-long line is cut hard
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(line[:max_chars])
            line = line[max_chars:]
        if cur and len(cur) + len(line) > max_chars:
            chunks.append(cur)
            cur = ""
        cur += line
    if cur:
        chunks.append(cur)
    return chunks


def normalize_answer(text):
    """The benchmark's normalize_answer (utils/eval_other_utils.py)."""
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def correct(prediction, golds):
    """The benchmark's substring_exact_match, against any gold answer."""
    pred = normalize_answer(prediction or "")
    return any(normalize_answer(g) in pred for g in golds)
```

- [ ] **Step 6: Write `scripts/get_data.py`**

```python
"""Downloads the two MemoryAgentBench splits pinned to commit 7ea0669 (plain HTTPS, no dataset code) and checks SHA-256."""
import urllib.request
from pathlib import Path

from cacheclm.mab import SHA256, URL, check_file


def main(out="data/memoryagentbench"):
    Path(out).mkdir(parents=True, exist_ok=True)
    for split in SHA256:
        path = Path(out) / f"{split}.parquet"
        if not path.exists():
            urllib.request.urlretrieve(URL.format(split=split), path)
        check_file(path, split)
        print("ok", path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Sync, verify the data, and run the tests**

Run: `uv sync && uv run python scripts/get_data.py && uv run pytest tests/test_mab.py -v`
Expected: two `ok data/memoryagentbench/...parquet` lines (the files are already present, so nothing downloads), then 6 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock .gitignore LICENSE configs scripts src tests
git commit -m "feat: project scaffold, pinned MemoryAgentBench loader, chunking and benchmark scoring"
```

---

### Task 2: Context text, cache-aware gate and command sandbox

**Files:**
- Create: `src/cacheclm/ctxfile.py`, `src/cacheclm/gate.py`, `src/cacheclm/sandbox.py`
- Test: `tests/test_ctxfile.py`, `tests/test_gate.py`, `tests/test_sandbox.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_ctxfile.py`:
```python
from cacheclm.ctxfile import append, blocks, drop_oldest, new_context, next_index, pinned_intact, split_for_summary, tokens


def build(n):
    ctx = new_context("TASK")
    for i in range(n):
        ctx = append(ctx, "chunk", f"part {i}")
    return ctx


def test_blocks_are_numbered_in_order():
    ctx = build(3)
    assert [(b[2], b[3], b[4]) for b in blocks(ctx)] == [(0, "task", True), (1, "chunk", False), (2, "chunk", False),
                                                       (3, "chunk", False)]
    assert next_index(ctx) == 4


def test_append_after_an_edit_without_trailing_newline():
    ctx = build(1).rstrip("\n") + "\n[[CTX_TURN 7 role=notes]]\nmy note"
    ctx = append(ctx, "chunk", "next")
    assert [b[2] for b in blocks(ctx)] == [0, 1, 7, 8]


def test_drop_oldest_keeps_the_pinned_block():
    ctx = drop_oldest(build(2))
    assert "part 0" not in ctx and "part 1" in ctx and ctx.startswith(new_context("TASK"))
    assert drop_oldest(new_context("TASK")) == new_context("TASK")


def test_split_for_summary_keeps_the_last_chunks():
    head, middle, tail = split_for_summary(build(4), keep=2)
    assert head == new_context("TASK")
    assert "part 0" in middle and "part 1" in middle and "part 2" not in middle
    assert "part 2" in tail and "part 3" in tail
    assert split_for_summary(build(2), keep=2) is None


def test_pinned_block_must_stay_first_and_unchanged():
    old = build(2)
    assert pinned_intact(old, old.replace("part 0", "p0"))
    assert not pinned_intact(old, old.replace("TASK", "TASK!"))
    assert not pinned_intact(old, old[old.index("[[CTX_TURN 1"):])


def test_tokens_are_chars_over_four():
    assert tokens("x" * 4001) == 1000
```

`tests/test_gate.py`:
```python
from cacheclm.gate import decide, first_change, survivors

DEEPSEEK = {"cache_read": 0.006, "cache_write": 0.30}
ANTHROPIC = {"cache_read": 0.30, "cache_write": 3.75}
K = 1000 * 4  # characters in 1K tokens


def test_first_change_and_survivors():
    old, new = "aaaXbbb", "aaabbb"
    assert first_change(old, new) == 3
    assert survivors(old, new, 3) == 3  # "bbb" survives after the deleted X
    assert survivors("abc", "abc", 3) == 0


def test_worked_example_middle_edit_is_rejected_on_deepseek():
    old = "A" * (8 * K) + "B" * (6 * K) + "C" * (16 * K)
    new = "A" * (8 * K) + "C" * (16 * K)
    allow, reason, numbers = decide(old, new, turns_left=10, price=DEEPSEEK)
    assert not allow and "rejected" in reason
    assert numbers["rebilled_tokens"] == 16000 and numbers["deleted_tokens"] == 6000


def test_worked_example_tail_edit_is_allowed():
    old = "A" * (23500 * 4) + "B" * (6 * K) + "C" * (500 * 4)
    new = "A" * (23500 * 4) + "C" * (500 * 4)
    allow, _, numbers = decide(old, new, turns_left=10, price=DEEPSEEK)
    assert allow and numbers["rebilled_tokens"] == 500


def test_anthropic_middle_edit_flips_near_thirty_turns():
    old = "A" * (8 * K) + "B" * (6 * K) + "C" * (16 * K)
    new = "A" * (8 * K) + "C" * (16 * K)
    assert not decide(old, new, 10, ANTHROPIC)[0]
    assert not decide(old, new, 30, ANTHROPIC)[0]  # 54,000 < 55,200
    assert decide(old, new, 31, ANTHROPIC)[0]  # 55,800 >= 55,200


def test_overflow_override_allows_a_shrinking_edit():
    old = "A" * (8 * K) + "B" * (6 * K) + "C" * (16 * K)
    new = "A" * (8 * K) + "C" * (16 * K)
    assert decide(old, new, 1, DEEPSEEK, overflow=True)[0]
    assert not decide(old, old.replace("B", "D"), 1, DEEPSEEK, overflow=True)[0]  # does not shrink


def test_same_length_rewrite_in_the_middle_is_rejected_and_append_passes():
    old = "A" * K + "B" * K + "C" * K
    assert not decide(old, "A" * K + "D" * K + "C" * K, 100, DEEPSEEK)[0]
    assert decide(old, old + "note", 1, DEEPSEEK)[0]
```

`tests/test_sandbox.py`:
```python
from pathlib import Path

from cacheclm.sandbox import check, run


def test_allow_list_and_shape_checks():
    assert check("sed -i '' 's/a/b/' ctx.txt") is None
    assert check("grep -v x ctx.txt > t && mv t ctx.txt") is None
    assert check("python3 -c \"t=open('ctx.txt').read(); open('ctx.txt','w').write(t[:10])\"") is None
    assert "not allowed" in check("curl example.com")
    assert "not allowed" in check("cat ctx.txt | nc host 1")
    assert "heredoc" in check("python3 - <<EOF\nprint(1)\nEOF")
    assert "one line" in check("cat ctx.txt\nrm ctx.txt")


def test_edit_runs_on_ctx():
    new, out = run("sed -i '' 's/apple/pear/' ctx.txt", "an apple\n")
    assert new == "an pear\n"


def test_refused_command_does_not_run():
    new, out = run("curl example.com", "keep\n")
    assert new == "keep\n" and out.startswith("REFUSED")


def test_no_network():
    new, out = run("python3 -c \"import socket; socket.create_connection(('1.1.1.1', 53), 2)\"", "x\n")
    assert "Error" in out and new == "x\n"


def test_no_writes_outside_and_no_home_reads(tmp_path):
    escape = tmp_path / "escape.txt"
    run(f"echo hi > {escape}", "x\n")
    assert not escape.exists()
    _, out = run(f"ls {Path.home()}", "x\n")
    assert "not permitted" in out.lower()


def test_timeout():
    _, out = run("python3 -c 'while True: pass'", "x\n", timeout=1)
    assert "timed out" in out
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_ctxfile.py tests/test_gate.py tests/test_sandbox.py -v`
Expected: FAIL with `ModuleNotFoundError` for `cacheclm.ctxfile`, `cacheclm.gate` and `cacheclm.sandbox`.

- [ ] **Step 3: Write `ctxfile.py`**

```python
"""The working context as one text: a pinned task block, then blocks in the paper's [[CTX_TURN i role=...]] format."""
import re

CHARS_PER_TOKEN = 4  # local estimate; the gate's inequality is scale-free and billing uses provider counts
HEADER = re.compile(r"^\[\[CTX_TURN (\d+) role=([\w-]+)( pinned)?\]\]$", re.M)


def tokens(text):
    return len(text) // CHARS_PER_TOKEN


def block(i, role, text, pinned=False):
    return f"[[CTX_TURN {i} role={role}{' pinned' if pinned else ''}]]\n{text.rstrip()}\n"


def new_context(task):
    return block(0, "task", task, pinned=True)


def blocks(ctx):
    """[(start, end, index, role, pinned)] for each block, in file order."""
    heads = list(HEADER.finditer(ctx))
    ends = [h.start() for h in heads[1:]] + [len(ctx)]
    return [(h.start(), end, int(h.group(1)), h.group(2), bool(h.group(3))) for h, end in zip(heads, ends)]


def next_index(ctx):
    return max((b[2] for b in blocks(ctx)), default=-1) + 1


def append(ctx, role, text):
    """ctx with a new block at the end, numbered after the highest index."""
    sep = "" if ctx.endswith("\n") else "\n"
    return ctx + sep + block(next_index(ctx), role, text)


def drop_oldest(ctx):
    """Remove the first unpinned block (forced truncation); unchanged if there is none."""
    for start, end, _, _, pinned in blocks(ctx):
        if not pinned:
            return ctx[:start] + ctx[end:]
    return ctx


def split_for_summary(ctx, keep):
    """(head, middle, tail): the pinned block, the blocks to compact, the last `keep` chunk blocks; None if nothing to compact."""
    bs = blocks(ctx)
    head_end = bs[0][1]
    chunks = [b for b in bs if b[3] == "chunk"]
    if len(chunks) <= keep or chunks[-keep][0] <= head_end:
        return None
    tail_start = chunks[-keep][0]
    return ctx[:head_end], ctx[head_end:tail_start], ctx[tail_start:]


def pinned_intact(old, new):
    """True when new still starts with old's pinned first block, unchanged."""
    start, end = blocks(old)[0][:2]
    return new.startswith(old[start:end])
```

- [ ] **Step 4: Write `gate.py`**

```python
"""Cache-aware edit gate: an edit passes only if its per-turn saving outweighs the cached tokens it forces to be re-billed."""
import os

from cacheclm.ctxfile import CHARS_PER_TOKEN


def first_change(old, new):
    """Character position where old and new first differ."""
    return len(os.path.commonprefix([old, new]))


def survivors(old, new, at):
    """Characters after the first change that both versions end with: cached text the provider must re-bill."""
    return len(os.path.commonprefix([old[at:][::-1], new[at:][::-1]]))


def decide(old, new, turns_left, price, overflow=False):
    """(allow, reason, numbers); price has cache_read and cache_write in USD per 1M tokens."""
    at = first_change(old, new)
    rebilled = survivors(old, new, at) // CHARS_PER_TOKEN
    deleted = max(0, len(old) - len(new)) // CHARS_PER_TOKEN
    cost = rebilled * (price["cache_write"] - price["cache_read"])
    saving = deleted * turns_left * price["cache_read"]
    numbers = {"first_change": at, "rebilled_tokens": rebilled, "deleted_tokens": deleted}
    if overflow and len(new) < len(old):
        return True, f"allowed (over the limit): breaks {rebilled:,} cached tokens to free {deleted:,}", numbers
    if saving >= cost:
        return True, f"allowed: frees {deleted:,} tokens for {turns_left} turns, breaks {rebilled:,} cached tokens", numbers
    return False, (f"edit rejected: breaks {rebilled:,} cached tokens to save {deleted:,}; "
                   "prefer deleting near the end"), numbers
```

- [ ] **Step 5: Write `sandbox.py`**

```python
"""Runs one model-written shell command on ctx.txt in a temp folder: allow-listed programs, macOS sandbox-exec jail
(no network, no writes outside the folder, no home-folder reads except Python), 10 s timeout, 2K characters of output."""
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ALLOWED = {"sed", "awk", "grep", "head", "tail", "cat", "wc", "echo", "printf", "mv", "cp", "python3"}
OPERATORS = {"|", "||", "&&", ";", "&"}
MAX_OUTPUT = 2000
PROFILE = """(version 1)
(allow default)
(deny network*)
(deny file-write* (require-not (subpath "{tmp}")))
(allow file-write* (literal "/dev/null"))
(deny file-read* (subpath "{home}"))
(allow file-read* (subpath "{tmp}") (subpath "{py}"))
"""


def check(command):
    """Why the command may not run, or None."""
    if "\n" in command:
        return "one line only; use python3 -c '...' for multi-step edits"
    if "<<" in command or "$(" in command or "`" in command:
        return "no heredocs or subshells; use python3 -c '...' instead"
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        words = list(lexer)
    except ValueError as e:
        return f"cannot parse the command: {e}"
    start = True
    for word in words:
        if word in OPERATORS:
            start = True
        elif start:
            if word not in ALLOWED:
                return f"{word!r} is not allowed; use one of: {', '.join(sorted(ALLOWED))}"
            start = False
    return None


def run(command, ctx, timeout=10):
    """(new ctx, output) after running the command on ctx.txt; ctx is unchanged when the command is refused."""
    why = check(command)
    if why:
        return ctx, f"REFUSED: {why}"
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d).resolve()
        (tmp / "ctx.txt").write_text(ctx)
        py = Path(sys.base_prefix).resolve()
        profile = PROFILE.format(tmp=tmp, home=Path.home().resolve(), py=py)
        env = {"PATH": f"{py / 'bin'}:/usr/bin:/bin", "HOME": str(tmp), "LC_ALL": "en_US.UTF-8"}
        try:
            p = subprocess.run(["sandbox-exec", "-p", profile, "/bin/bash", "-c", command], cwd=tmp, env=env,
                               capture_output=True, text=True, timeout=timeout)
            output = p.stdout + p.stderr
        except subprocess.TimeoutExpired:
            output = f"ERROR: timed out after {timeout} s"
        path = tmp / "ctx.txt"
        new = path.read_text(errors="replace") if path.exists() else ""
    return new, output[:MAX_OUTPUT]
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_ctxfile.py tests/test_gate.py tests/test_sandbox.py -v`
Expected: all PASS.

If `test_no_writes_outside_and_no_home_reads` or `test_no_network` fails because `sandbox-exec` is missing or the profile is rejected: stop and report the exact error to the user. Do not weaken the sandbox.

If `python3` cannot start under the profile (for example `Operation not permitted` on a path under the home folder): print that path from the error and add only that subpath to the `allow file-read*` line.

- [ ] **Step 7: Commit**

```bash
git add src/cacheclm/ctxfile.py src/cacheclm/gate.py src/cacheclm/sandbox.py tests/test_ctxfile.py tests/test_gate.py tests/test_sandbox.py
git commit -m "feat: context blocks, cache-aware edit gate and sandboxed edit commands"
```

---

### Task 3: Cached client, budget, the three arms and the run loop (milestone review after this task)

**Files:**
- Create: `src/cacheclm/llm.py`, `src/cacheclm/budget.py`, `src/cacheclm/arms.py`, `src/cacheclm/run.py`, `src/cacheclm/__main__.py`
- Test: `tests/test_llm.py`, `tests/test_run.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_llm.py`:
```python
from types import SimpleNamespace

import pytest

from cacheclm.budget import Budget, BudgetExceeded
from cacheclm.llm import LLM, cost

PRICE = {"cache_read": 0.006, "cache_write": 0.30, "output": 1.20}


class FakeClient:
    def __init__(self):
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls += 1
        usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=100, prompt_cache_hit_tokens=600)
        return SimpleNamespace(model="deepseek-flash", usage=usage,
                               choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))])


def test_cost_bills_hits_misses_and_output():
    reply = {"prompt_tokens": 1000, "cache_hit_tokens": 600, "completion_tokens": 100}
    assert cost(reply, PRICE) == pytest.approx((600 * 0.006 + 400 * 0.30 + 100 * 1.20) / 1e6)


def test_replay_is_free_and_repeats_are_separate(tmp_path):
    client, spent = FakeClient(), []
    llm = LLM("deepseek-flash", client, tmp_path, 0.3, PRICE, on_spend=spent.append)
    msgs = [{"role": "user", "content": "q"}]
    first = llm.chat(msgs, 64, repeat=0)
    again = llm.chat(msgs, 64, repeat=0)
    other = llm.chat(msgs, 64, repeat=1)
    assert first["cache_hit_tokens"] == 600 and not first["cached"] and again["cached"] and not other["cached"]
    assert client.calls == 2 and len(spent) == 2


def test_budget_persists_and_stops_past_the_cap(tmp_path):
    budget = Budget(tmp_path / "spend.json", 1.0)
    budget.spend(0.6)
    assert Budget(tmp_path / "spend.json", 1.0).total == pytest.approx(0.6)
    with pytest.raises(BudgetExceeded):
        budget.spend(0.5)
```

`tests/test_run.py`:
```python
from cacheclm.arms import control, parse_command
from cacheclm.files import read_jsonl
from cacheclm.mab import Sample
from cacheclm.run import run_log, run_sample

PRICE = {"cache_read": 0.006, "cache_write": 0.30, "output": 1.20}
CFG = {"context_budget": 400, "chunk_tokens": 40, "max_edits_per_chunk": 3, "max_condense_tries": 3,
       "summary_at": 0.75, "keep_recent_chunks": 2, "nudges": [0.25, 0.5, 0.75], "query_workers": 2}
DROP_FIRST = ("```bash\npython3 -c \"t=open('ctx.txt').read(); b=t.split('[[CTX_TURN '); "
              "open('ctx.txt','w').write('[[CTX_TURN '.join(b[:2]+b[3:]))\"\n```")


def sample():
    facts = "".join(f"{i}. fact number {i}.\n" for i in range(60))
    return Sample("T/0", "factconsolidation_sh_test", facts, ["q1", "q2"], [["fact number 59"], ["nope"]])


class FakeLLM:
    def __init__(self, script):
        self.script, self.calls = script, 0

    def chat(self, messages, max_tokens, repeat):
        self.calls += 1
        text = messages[-1]["content"]
        return {"content": self.script(text), "model": "fake", "prompt_tokens": len(text) // 4,
                "cache_hit_tokens": 0, "completion_tokens": 10, "latency_s": 0.0, "cached": False}


def answers(text):
    return "fact number 59" if "Question:" in text else None


def events(tmp_path, arm, name):
    return [r for r in read_jsonl(run_log(tmp_path, arm, sample(), 0)) if r.get("event") == name]


def test_parse_command_and_control_note():
    assert parse_command("READY") is None
    assert parse_command("```bash\ncat ctx.txt\n```") == "cat ctx.txt"
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


def test_damaging_the_pinned_block_is_rolled_back(tmp_path):
    bad = "```bash\npython3 -c \"t=open('ctx.txt').read(); open('ctx.txt','w').write(t[10:])\"\n```"
    llm = FakeLLM(lambda t: answers(t) or bad)
    run_sample(sample(), "clm", 0, llm, CFG, PRICE, tmp_path)
    assert all(not e["allowed"] and "pinned" in e["reason"] for e in events(tmp_path, "clm", "edit"))


def test_finished_run_is_skipped(tmp_path):
    llm = FakeLLM(lambda t: answers(t) or "READY")
    first = run_sample(sample(), "clm", 0, llm, CFG, PRICE, tmp_path)
    calls = llm.calls
    assert run_sample(sample(), "clm", 0, llm, CFG, PRICE, tmp_path) == first and llm.calls == calls
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_llm.py tests/test_run.py -v`
Expected: FAIL with `ModuleNotFoundError` for `cacheclm.budget`, `cacheclm.llm`, `cacheclm.arms` and `cacheclm.run`.

- [ ] **Step 3: Write `budget.py`**

```python
"""Persisted spend guard: adds up real (non-cached) API spend across runs and stops past the cap."""
import json
import threading
from pathlib import Path

from cacheclm.files import write_atomic


class BudgetExceeded(Exception):
    pass


class Budget:
    def __init__(self, path, cap_usd):
        self.path, self.cap = Path(path), cap_usd
        self.total = json.loads(self.path.read_text())["usd"] if self.path.exists() else 0.0
        self.lock = threading.Lock()

    def spend(self, usd):
        """Pass as an LLM's on_spend: records each real call, raises once the total passes the cap."""
        with self.lock:
            self.total += usd
            write_atomic(self.path, json.dumps({"usd": self.total}))
            if self.total > self.cap:
                raise BudgetExceeded(f"spent ${self.total:.3f} of ${self.cap:.2f}")
```

- [ ] **Step 4: Write `llm.py`**

```python
"""OpenAI-compatible chat client with a disk cache (replays are free), retries, and billed usage incl. cache-hit tokens."""
import hashlib
import json
import time
from pathlib import Path

import openai

from cacheclm.files import write_atomic

RETRYABLE = (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError, openai.RateLimitError)


def cost(reply, price):
    """USD of one call: cache hits at cache_read, other prompt tokens at cache_write, output at output."""
    hit = reply["cache_hit_tokens"]
    miss = reply["prompt_tokens"] - hit
    return (hit * price["cache_read"] + miss * price["cache_write"] + reply["completion_tokens"] * price["output"]) / 1e6


def _hits(usage):
    """Prompt tokens served from the provider's prefix cache (DeepSeek or OpenAI-style field)."""
    hits = getattr(usage, "prompt_cache_hit_tokens", None)
    if hits is None:
        details = getattr(usage, "prompt_tokens_details", None)
        hits = getattr(details, "cached_tokens", 0) if details else 0
    return hits or 0


class LLM:
    def __init__(self, model, client, cache_dir, temperature, price, options=None, on_spend=None, max_tries=4):
        self.model, self.client, self.cache_dir = model, client, Path(cache_dir)
        self.temperature, self.price, self.options = temperature, price, options or {}
        self.on_spend, self.max_tries = on_spend, max_tries

    def chat(self, messages, max_tokens, repeat):
        """{content, model, prompt_tokens, cache_hit_tokens, completion_tokens, latency_s, cached}."""
        payload = [self.model, self.options, self.temperature, repeat, max_tokens, messages]
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / key[:2] / f"{key}.json"
        if path.exists():
            return {**json.loads(path.read_text()), "cached": True}
        reply = self._call(messages, max_tokens)
        write_atomic(path, json.dumps(reply))
        if self.on_spend:
            self.on_spend(cost(reply, self.price))
        return {**reply, "cached": False}

    def _call(self, messages, max_tokens):
        last = None
        for attempt in range(self.max_tries):
            start = time.monotonic()
            try:
                resp = self.client.chat.completions.create(model=self.model, messages=messages,
                                                           temperature=self.temperature, max_tokens=max_tokens,
                                                           **self.options)
            except RETRYABLE as e:
                last = e
                time.sleep(min(2 ** (attempt + 1), 60))
                continue
            return {"content": resp.choices[0].message.content or "", "model": resp.model,
                    "prompt_tokens": resp.usage.prompt_tokens, "cache_hit_tokens": _hits(resp.usage),
                    "completion_tokens": resp.usage.completion_tokens,
                    "latency_s": round(time.monotonic() - start, 3)}
        raise RuntimeError(f"{self.model}: gave up after {self.max_tries} tries: {last}")
```

- [ ] **Step 5: Write `arms.py`**

````python
"""The three context policies: summary (harness compacts at 75%), clm (model edits ctx.txt with bash),
gate (clm plus the cache-aware gate). chat(messages, max_tokens, phase) -> reply; log(record) stores an event."""
import re

from cacheclm import sandbox
from cacheclm.ctxfile import block, drop_oldest, next_index, pinned_intact, split_for_summary, tokens
from cacheclm.gate import decide

ARMS = ("summary", "clm", "gate")
BASE = "You are a helpful assistant that can read the context and memorize it for future retrieval."
EDITING = """
Your working context is the file ctx.txt, shown in each message. It holds at most {budget} tokens. Parts of a long text arrive one at a time and are appended to the end of ctx.txt. Between parts you may edit ctx.txt to keep what will matter later: delete irrelevant passages, shorten text, or keep notes in a block of your own such as [[CTX_TURN 99 role=notes]]. Never change or move the first block ([[CTX_TURN 0 role=task pinned]]).
To edit, reply with exactly one shell command in a ```bash block. Allowed programs: sed, awk, grep, head, tail, cat, wc, echo, printf, mv, cp, python3. One line only, no heredocs; use python3 -c '...' for multi-step edits (on this system, in-place sed is sed -i ''). You will see the command's output. Reply READY when you are done editing."""
SUMMARY_NOTE = "\nParts of a long text are appended to your working context. When it gets full you will be asked to compact older parts into a summary."
SUMMARIZE = ("The working context is nearly full. Write one summary that replaces every block after the pinned task block "
             "and before the last {keep} parts. Keep every detail that may be needed to answer later questions: names, "
             "events in story order, and facts with their serial numbers. Reply with only the summary text.")
FENCE = re.compile(r"```(?:bash|sh)?\n(.*?)```", re.S)


def system(arm, budget):
    return BASE + (SUMMARY_NOTE if arm == "summary" else EDITING.format(budget=budget))


def parse_command(reply):
    """The command in the reply's fenced block, or None (READY, or no block)."""
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


def edit_phase(ctx, arm, chat, cfg, incoming, turns_left, price, log):
    """ctx after up to max_edits commands, plus up to max_condense more while the next part would not fit."""
    budget, last, edits = cfg["context_budget"], "", 0
    while True:
        over = tokens(ctx) + incoming > budget
        if edits >= cfg["max_edits_per_chunk"] + (cfg["max_condense_tries"] if over else 0):
            return ctx
        note = control(ctx, budget, incoming, cfg["nudges"], last)
        messages = [{"role": "system", "content": system(arm, budget)}, {"role": "user", "content": ctx + "\n\n" + note}]
        command = parse_command(chat(messages, 1024, "edit")["content"])
        if command is None:
            return ctx
        edits += 1
        new, output = sandbox.run(command, ctx)
        allow, reason, numbers = decide(ctx, new, turns_left, price, overflow=over)
        if not pinned_intact(ctx, new):
            allow, reason = False, "edit rolled back: the pinned first block must stay unchanged at the top"
        elif arm == "clm":
            allow, reason = True, "applied"
        log({"event": "edit", "command": command, "allowed": allow, "reason": reason, "over": over,
             "turns_left": turns_left, "chars_before": len(ctx), "chars_after": len(new), **numbers})
        if allow:
            ctx = new
        last = f"{output}\n{reason}".strip()


def summary_step(ctx, chat, cfg, incoming, log):
    """ctx with older blocks compacted into one summary once ctx plus the next part passes summary_at of the budget."""
    budget = cfg["context_budget"]
    if tokens(ctx) + incoming <= cfg["summary_at"] * budget:
        return ctx
    parts = split_for_summary(ctx, cfg["keep_recent_chunks"])
    if parts is None:
        return ctx
    head, middle, tail = parts
    prompt = SUMMARIZE.format(keep=cfg["keep_recent_chunks"])
    messages = [{"role": "system", "content": system("summary", budget)},
                {"role": "user", "content": ctx + "\n\n" + prompt}]
    summary = chat(messages, budget // 4, "summary")["content"]
    log({"event": "summary", "chars_before": len(ctx), "chars_compacted": len(middle), "summary_chars": len(summary)})
    return head + block(next_index(ctx), "summary", summary) + tail


def fit(ctx, incoming, budget, log):
    """Drop the oldest unpinned blocks until ctx plus the next part fits (forced truncation)."""
    while tokens(ctx) + incoming > budget:
        smaller = drop_oldest(ctx)
        if smaller == ctx:
            break
        log({"event": "forced_truncation", "chars_dropped": len(ctx) - len(smaller)})
        ctx = smaller
    return ctx
````

- [ ] **Step 6: Write `run.py`**

```python
"""Runs one sample through one arm: stream the chunks under the arm's context policy, then ask every question on the
frozen context. Every call is logged with billed usage and an ideal cache hit (prefix shared with the reference request)."""
import datetime
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cacheclm.arms import edit_phase, fit, summary_step, system
from cacheclm.ctxfile import CHARS_PER_TOKEN, append, new_context, tokens
from cacheclm.files import append_jsonl, read_jsonl
from cacheclm.mab import MEMORIZE, QUERY, TASK, chunk_text, correct, family


def run_log(runs_dir, arm, sample, repeat):
    return Path(runs_dir) / arm / f"{sample.sid.replace('/', '_')}_r{repeat}.jsonl"


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
        ideal = len(os.path.commonprefix([reference, text])) // CHARS_PER_TOKEN
        reply = self.llm.chat(messages, max_tokens, self.repeat)
        if ref is None:
            self.ref = text
        self.log({"event": "call", "phase": phase, "model": reply["model"], "date": datetime.date.today().isoformat(),
                  "prompt_tokens": reply["prompt_tokens"], "cache_hit_tokens": reply["cache_hit_tokens"],
                  "completion_tokens": reply["completion_tokens"], "latency_s": reply["latency_s"],
                  "cached": reply["cached"], "ideal_hit_tokens": min(ideal, reply["prompt_tokens"]),
                  "estimated_tokens": estimate})
        return reply


def answer_all(ctx, arm, fam, questions, rec, cfg):
    """Each question in its own call on the frozen context; the first call warms the provider cache."""
    head = system(arm, cfg["context_budget"])

    def ask(q):
        messages = [{"role": "system", "content": head},
                    {"role": "user", "content": ctx + "\n\n" + QUERY[fam].format(question=q)}]
        return rec.chat(messages, 256, "query", ref=head + ctx)["content"]

    first = [ask(questions[0])]
    with ThreadPoolExecutor(cfg["query_workers"]) as pool:
        return first + list(pool.map(ask, questions[1:]))


def run_sample(sample, arm, repeat, llm, cfg, price, runs_dir, question_limit=None):
    """Accuracy of one (sample, arm, repeat); a log that already ends with a done record is not rerun."""
    path = run_log(runs_dir, arm, sample, repeat)
    done = [r for r in read_jsonl(path) if r.get("event") == "done"]
    if done:
        return done[-1]["accuracy"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)  # a partial run restarts; its calls replay from the cache for free
    rec = Recorder(path, llm, repeat, {"arm": arm, "sample": sample.sid, "source": sample.source, "repeat": repeat})
    fam, budget = family(sample.source), cfg["context_budget"]
    questions, golds = sample.questions[:question_limit], sample.answers[:question_limit]
    chunks = chunk_text(sample.context, cfg["chunk_tokens"] * CHARS_PER_TOKEN)
    ctx = new_context(TASK[fam].format(n=len(questions)))
    for i, chunk in enumerate(chunks + [None]):  # None: one last edit phase after the final part
        text = MEMORIZE[fam].format(chunk=chunk) if chunk is not None else ""
        incoming = tokens(text) + 8 if chunk is not None else 0  # +8 for the block header
        turns_left = len(chunks) - i + len(questions)
        if arm == "summary":
            if chunk is not None:
                ctx = summary_step(ctx, rec.chat, cfg, incoming, rec.log)
        else:
            ctx = edit_phase(ctx, arm, rec.chat, cfg, incoming, turns_left, price, rec.log)
        ctx = fit(ctx, incoming, budget, rec.log)
        if chunk is not None:
            ctx = append(ctx, "chunk", text)
        rec.log({"event": "step", "chunk": i, "ctx_tokens": tokens(ctx)})
    predictions = answer_all(ctx, arm, fam, questions, rec, cfg)
    results = [correct(p, g) for p, g in zip(predictions, golds)]
    for qi, (p, ok) in enumerate(zip(predictions, results)):
        rec.log({"event": "answer", "qi": qi, "prediction": p, "correct": ok})
    accuracy = sum(results) / len(results)
    rec.log({"event": "done", "accuracy": accuracy, "n_questions": len(results), "ctx_tokens_final": tokens(ctx)})
    return accuracy
```

- [ ] **Step 7: Write `__main__.py`**

```python
"""CLI: `cacheclm smoke` (1 sample, 5 questions, 3 arms), `cacheclm run` (the 7 samples x 3 arms x 2 repeats),
`cacheclm report [--smoke]`. Settings in configs/base.yaml; DEEPSEEK_API_KEY from a .env in this or a parent folder."""
import argparse
import os

import openai
import yaml
from dotenv import find_dotenv, load_dotenv

from cacheclm.arms import ARMS
from cacheclm.budget import Budget, BudgetExceeded
from cacheclm.llm import LLM
from cacheclm.mab import load_sample
from cacheclm.report import write_report
from cacheclm.run import run_sample


def make_llm(cfg, price, budget):
    agent = cfg["agent"]
    client = openai.OpenAI(base_url=agent["base_url"], api_key=os.environ[agent["api_key_env"]], max_retries=0,
                           timeout=300)
    return LLM(agent["model"], client, cfg["cache_dir"], cfg["temperature"], price, agent.get("options"), budget.spend)


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
    samples = [cfg["smoke"]["sample"]] if smoke else cfg["samples"]
    repeats = [0] if smoke else cfg["repeats"]
    limit = cfg["smoke"]["questions"] if smoke else None
    try:
        for repeat in repeats:
            for spec in samples:
                sample = load_sample(cfg["data_dir"], spec["split"], spec["row"])
                for arm in ARMS:
                    try:
                        acc = run_sample(sample, arm, repeat, llm, cfg, prices["deepseek"], runs_dir, limit)
                    except RuntimeError as e:  # API kept failing after retries: skip, the report lists it as incomplete
                        print(f"FAILED   {arm:8s} {sample.sid:24s} r{repeat}: {e}")
                        continue
                    print(f"{arm:8s} {sample.sid:24s} r{repeat}  accuracy {acc:.2f}  total spend ${budget.total:.3f}")
    except BudgetExceeded as e:
        print(f"Stopped: {e}. Finished runs are kept; raise budget_usd in the config to continue.")
```

- [ ] **Step 8: Run all tests**

`__main__.py` imports `write_report`, which arrives in Task 4. Add a temporary `src/cacheclm/report.py` containing only:

```python
"""Report (written in Task 4)."""


def write_report(runs_dir, out_dir, prices):
    raise NotImplementedError("Task 4")
```

Run: `uv run pytest -v`
Expected: all tests PASS (Tasks 1-3).

- [ ] **Step 9: Commit**

```bash
git add src/cacheclm tests/test_llm.py tests/test_run.py
git commit -m "feat: cached client, budget guard, the three context arms and the run loop"
```

- [ ] **Step 10: Milestone review**

Dispatch the `superpowers:code-reviewer` agent on the diff since the spec commit, against `docs/specs/2026-10-05-cacheclm-design.md`. Fix every finding marked important, then rerun `uv run pytest -v` and commit (`fix: review findings`).

---

### Task 4: Report, README and the smoke handover (milestone review after this task)

**Files:**
- Modify: `src/cacheclm/report.py` (replace the stub)
- Create: `README.md`
- Test: `tests/test_report.py`

- [ ] **Step 1: Write the failing tests** in `tests/test_report.py`

```python
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
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL with `ImportError: cannot import name 'call_cost'`.

- [ ] **Step 3: Write `report.py`** (replacing the stub)

```python
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
```

- [ ] **Step 4: Run all tests**

Run: `uv run pytest -v`
Expected: all PASS.

- [ ] **Step 5: Write `README.md`**

The README should have these sections, in plain words, with no results yet:
1. **One-sentence question.**
2. **Why it matters:** CLM's FLOP savings vs hosted-API cache billing; cite arXiv 2609.37725.
3. **The three arms.**
4. **The gate:** the inequality, plus the worked-example table from the spec.
5. **Data:** MemoryAgentBench pin and hashes, the 7 samples, and the prompts adapted from `templates.py`.
6. **How to run:** `uv sync`; `uv run python scripts/get_data.py`; `uv run cacheclm smoke`; `uv run cacheclm report --smoke`; `uv run cacheclm run`; `uv run cacheclm report`.
7. **Cost and the cap.**
8. **Credits:** CLM paper (CC BY-NC repository; no code copied), MemoryAgentBench (MIT).

- [ ] **Step 6: Commit**

```bash
git add src/cacheclm/report.py tests/test_report.py README.md
git commit -m "feat: report with primary endpoints, three-provider repricing and the accuracy-vs-cost chart"
```

- [ ] **Step 7: Milestone review**

Dispatch the `superpowers:code-reviewer` agent on the whole repo against the spec. Fix the important findings, rerun `uv run pytest -v`, and commit (`fix: review findings`).

- [ ] **Step 8: Hand the smoke run to the user** (do not run it yourself)

Give the user:

```bash
cd /Users/hvinnakota/Downloads/new-techniques/cacheclm && uv run cacheclm smoke && uv run cacheclm report --smoke
```

Expected: 3 lines like `summary  Conflict_Resolution/6  r0  accuracy 0.80  total spend $0.03`, then `results_smoke/summary.md`.

When the user reports back, read `results_smoke/summary.md` and the run logs, and check:
- each CLM arm made edits;
- the gate both allowed and rejected some edits;
- the billed vs ideal cache-hit gap;
- the projected full cost.

**Projection:** smoke spend × (5 EventQA samples at about 2x the chunks of the smoke sample + 2 FactConsolidation samples) × 2 repeats × (100 / 5 questions on the query share). Present the numbers. If the projection is above $4.5, switch the EventQA rows in `configs/base.yaml` to 7-11 (`eventqa_65536`), note the switch in the spec, and commit before the full run.

- [ ] **Step 9: Hand the full run to the user**

```bash
cd /Users/hvinnakota/Downloads/new-techniques/cacheclm && uv run cacheclm run && uv run cacheclm report
```

The run resumes after a stop: finished (sample, arm, repeat) logs are skipped, and cached calls replay for free. Once it finishes, add a Results section to the README with the chart and the endpoints, and the caveats from the spec's Risks section.
