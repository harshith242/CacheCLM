"""Runs one sample through one arm: stream the chunks under the arm's context policy (or none / the full text for the
reference arms), then ask every question on the frozen context. Calls are logged with billed and ideal cache hits."""
import datetime
import hashlib
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cacheclm.arms import BASE, edit_phase, fit, summary_step
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
        self.stream = hashlib.sha1()  # fingerprint of every streaming request and reply: identifies a replayed stream

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
        if phase != "query":
            self.stream.update(f"{text}\0{reply['content']}\0".encode())
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
    """Each question in its own call on the frozen context, under one query prompt for every arm (no editing protocol
    or skill recipe), so answers and their cost do not depend on the arm's instructions. The first question is compared
    with the real previous request; questions go one at a time until the provider reports a cache hit (at most
    serial_max), then in parallel."""
    head = f"Run r{rec.repeat}.\n" + BASE

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
    """The body after every chunk has streamed through the arm's context policy. An edit arm that cannot make room for
    the next part stops reading there; the summary arm falls back to the harness cut (fit)."""
    body, budget, prev_used = "", cfg["context_budget"], 0
    for i, chunk in enumerate(chunks + [None]):  # None: one last edit phase after the final part
        text = MEMORIZE[fam].format(chunk=chunk) if chunk is not None else ""
        incoming = tokens(text) + 8 if chunk is not None else 0  # +8 for the block header
        turns_left = len(chunks) - i + n_questions
        if arm == "summary":
            if chunk is not None:
                body = summary_step(task, body, rec.chat, cfg, incoming, rec.log, repeat)
            body = fit(task, body, incoming, budget, rec.log)  # the harness's own fallback, as for Codex compaction
        else:
            body = edit_phase(task, body, arm, rec.chat, cfg, incoming, turns_left, price, rec.log, repeat, prev_used,
                              fam)
            prev_used = tokens(task + body)
            if prev_used + incoming > budget:  # CacheCLM policy (not the paper's): no harness cut, the reading ends
                rec.log({"event": "overflow_stop", "policy": "CacheCLM stop-on-overflow", "part": i,
                         "unread_parts": len(chunks) - i})
                break
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
        single_hop = sample.source.startswith("factconsolidation_sh")  # multi-hop golds do not name the subject
        fact = gold_fact(sample.context, questions[qi], golds[qi]) if single_hop else None
        rec.log({"event": "answer", "qi": qi, "prediction": p, "correct": ok,
                 "gold_fact_present": (fact in ctx) if fact else None})
    accuracy = sum(results) / len(results)
    rec.stream.update(ctx.encode())
    rec.log({"event": "done", "accuracy": accuracy, "n_questions": len(results), "ctx_tokens_final": tokens(ctx),
             "unit": unit_id(sample), "family": fam, "final_context": ctx, "stream_hash": rec.stream.hexdigest()})
    return accuracy
