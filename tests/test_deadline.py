from datetime import date

import pytest

from foi_agent.deadline import foi_deadline


@pytest.mark.parametrize(
    ("received", "expected"),
    [
        (date(2026, 10, 26), date(2026, 11, 23)),  # Monday
        (date(2026, 10, 30), date(2026, 11, 27)),  # Friday
        (date(2026, 10, 31), date(2026, 11, 27)),  # Saturday: day one is Monday
        (date(2026, 11, 1), date(2026, 11, 27)),  # Sunday: day one is Monday
    ],
)
def test_foi_deadline_is_20_working_days_after_receipt(received, expected):
    assert foi_deadline(received) == expected
