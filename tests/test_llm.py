from types import SimpleNamespace

import httpx2 as httpx  # the HTTP client this openai version uses
import openai
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
                               choices=[SimpleNamespace(message=SimpleNamespace(content="hi"), finish_reason="length")])


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


class ErrorClient(FakeClient):
    def __init__(self, error):
        super().__init__()
        self.error = error

    def create(self, **kw):
        self.calls += 1
        raise self.error


def status_error(cls, code):
    return cls("boom", response=httpx.Response(code, request=httpx.Request("POST", "https://x")), body=None)


def test_a_rejected_request_fails_this_sample_only(tmp_path):
    client = ErrorClient(status_error(openai.BadRequestError, 400))
    llm = LLM("m", client, tmp_path, 0.3, PRICE)
    with pytest.raises(RuntimeError, match="400"):
        llm.chat([{"role": "user", "content": "q"}], 8, 0)
    assert client.calls == 1  # not retried


def test_auth_errors_still_stop_the_run(tmp_path):
    llm = LLM("m", ErrorClient(status_error(openai.AuthenticationError, 401)), tmp_path, 0.3, PRICE)
    with pytest.raises(openai.AuthenticationError):
        llm.chat([{"role": "user", "content": "q"}], 8, 0)


def test_connection_errors_are_retried_then_the_sample_fails(tmp_path, monkeypatch):
    monkeypatch.setattr("cacheclm.llm.time.sleep", lambda s: None)
    client = ErrorClient(openai.APIConnectionError(request=httpx.Request("POST", "https://x")))
    llm = LLM("m", client, tmp_path, 0.3, PRICE, max_tries=4)
    with pytest.raises(RuntimeError, match="gave up after 4 tries"):
        llm.chat([{"role": "user", "content": "q"}], 8, 0)
    assert client.calls == 4 and not list(tmp_path.rglob("*.json"))  # nothing cached, so a rerun tries again


def test_replies_carry_the_finish_reason(tmp_path):
    llm = LLM("deepseek-flash", FakeClient(), tmp_path, 0.3, PRICE)
    assert llm.chat([{"role": "user", "content": "q"}], 64, repeat=0)["finish_reason"] == "length"
