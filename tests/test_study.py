from src.study import wilson_interval


def test_wilson_interval_contains_observed_rate() -> None:
    low, high = wilson_interval(5, 5)
    assert 0.0 < low < 1.0
    assert high == 1.0
    assert low <= 1.0 <= high


def test_wilson_interval_rejects_invalid_counts() -> None:
    for successes, total in ((-1, 5), (6, 5), (0, 0)):
        try:
            wilson_interval(successes, total)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid binomial counts must fail")
