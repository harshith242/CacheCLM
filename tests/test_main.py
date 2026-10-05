from cacheclm.__main__ import jobs
from cacheclm.arms import ARMS, REFERENCES

CFG = {"repeats": [0], "samples": [{"split": "A", "row": 1}, {"split": "B", "row": 2}],
       "extra_runs": [{"split": "B", "row": 2, "repeat": 1}],
       "smoke": {"samples": [{"split": "C", "row": 1, "questions": 100, "context_budget": 12000}]}}


def test_full_run_jobs_add_references_once_and_the_extra_repeat():
    planned = jobs(CFG, smoke=False)
    assert [(s["split"], r, arms) for s, r, arms in planned] == [  # references last: a budget stop hits them first
        ("A", 0, ARMS), ("B", 0, ARMS), ("B", 1, ARMS), ("A", 0, REFERENCES), ("B", 0, REFERENCES)]


def test_smoke_jobs_run_every_arm_so_the_references_are_checked_live_once():
    assert [(s["split"], r, arms) for s, r, arms in jobs(CFG, smoke=True)] == [("C", 0, ARMS + REFERENCES)]
