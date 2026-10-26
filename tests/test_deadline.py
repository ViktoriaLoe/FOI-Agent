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


@pytest.mark.parametrize(
    ("received", "expected"),
    [
        # St Andrew's Day (Mon 30 Nov 2026) is a Scottish bank holiday, and FOIA s10(6) counts
        # bank holidays in any part of the UK, so the deadline moves from 30 Nov to 1 Dec.
        (date(2026, 11, 2), date(2026, 12, 1)),
        # Christmas Day, the Boxing Day substitute (28 Dec), New Year's Day and Scotland's
        # 2 January substitute (4 Jan) all fall in this window.
        (date(2026, 12, 24), date(2027, 1, 27)),
        # Received on a bank holiday: day one is the next working day.
        (date(2026, 12, 28), date(2027, 1, 27)),
        # St Patrick's Day (Northern Ireland only), Good Friday and Easter Monday.
        (date(2027, 3, 1), date(2027, 4, 1)),
    ],
)
def test_bank_holidays_anywhere_in_the_uk_are_not_working_days(received, expected):
    assert foi_deadline(received) == expected


def test_a_deadline_beyond_the_bank_holiday_list_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="bank_holidays.json"):
        foi_deadline(date(2028, 12, 20))
