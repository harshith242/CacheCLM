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
