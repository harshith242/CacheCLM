from pathlib import Path

import pytest

from cacheclm.mab import check_file, chunk_text, correct, gold_fact, load_sample, normalize_answer

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
    assert [len(c) for c in chunks][:3] == [100, 100, 53]  # the 51-char tail ("a"*50 + newline) joins "b\n"


def test_scoring_matches_the_benchmark():
    assert normalize_answer("The  Ancient-Greek!") == "ancientgreek"
    assert correct("It was written in Ancient Greek.", ["Ancient Greek"])
    assert correct("answer: France", ["Germany", "France"])
    assert not correct("Paris", ["France"])
    assert not correct(None, ["France"])


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


def test_gold_fact_is_the_latest_matching_fact_about_the_question():
    context = ("1. The author of Hard Times is Charles Dickens.\n"
               "2. Martin Luther King Jr. was born in Atlanta.\n"
               "9. The author of Hard Times is Martin Luther King Jr.\n")
    golds = ["Martin Luther King Jr."]
    assert gold_fact(context, "Who is the author of Hard Times?", golds) == "9. The author of Hard Times is Martin Luther King Jr."
    assert gold_fact(context, "Where is Paris?", ["Lyon"]) is None


def test_gold_fact_ignores_question_stopwords():
    context = ("3. iPad Pro was developed by Lakshmi Mittal.\n"
               "9. The chief executive officer of Google is Lakshmi Mittal.\n")
    assert gold_fact(context, "Who is the developer of iPad Pro?", ["Lakshmi Mittal"]).startswith("3.")
