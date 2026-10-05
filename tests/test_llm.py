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
