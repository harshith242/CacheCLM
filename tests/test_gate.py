from cacheclm.gate import decide

DEEPSEEK = {"cache_read": 0.006, "cache_write": 0.30}
ANTHROPIC = {"cache_read": 0.30, "cache_write": 3.75}
K = 1000 * 4  # characters in 1K tokens


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
