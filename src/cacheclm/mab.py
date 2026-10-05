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
