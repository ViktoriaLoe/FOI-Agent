"""Statutory response deadlines: FOI and EIR in working days, subject access in calendar months."""

import calendar
import json
from datetime import date, timedelta
from functools import cache
from importlib import resources

WORKING_DAYS = 20


@cache
def uk_bank_holidays() -> frozenset[date]:
    """Bank holidays in any part of the UK, from the gov.uk snapshot in data/bank_holidays.json.

    FOIA s10(6) excludes a bank holiday in any part of the UK, so St Andrew's Day (Scotland) and
    St Patrick's Day (Northern Ireland) count for every authority.
    """
    path = resources.files("foi_agent") / "data" / "bank_holidays.json"
    divisions = json.loads(path.read_text(encoding="utf-8"))["divisions"]
    return frozenset(date.fromisoformat(e["date"]) for events in divisions.values() for e in events)


def is_working_day(day: date) -> bool:
    return day.weekday() < 5 and day not in uk_bank_holidays()


def _require_known_holidays(day: date) -> None:
    """Refuse a deadline beyond the years the bank holiday snapshot covers, rather than silently
    ignoring unknown holidays."""
    last_covered = max(uk_bank_holidays()).year
    if day.year > last_covered:
        raise ValueError(
            f"Bank holidays are only known up to {last_covered}: "
            "refresh data/bank_holidays.json from https://www.gov.uk/bank-holidays.json"
        )


def foi_deadline(received: date) -> date:
    """The 20th working day after receipt. EIR requests have the same limit.

    Day one is the first working day after the request is received, including when it arrives on
    a non-working day (ICO guidance). Raises ValueError if the deadline would fall beyond the
    years the bank holiday snapshot covers.
    """
    day = received
    counted = 0
    while counted < WORKING_DAYS:
        day += timedelta(days=1)
        _require_known_holidays(day)
        if is_working_day(day):
            counted += 1
    return day


def sar_deadline(received: date) -> date:
    """One calendar month after receipt, following ICO guidance on subject access.

    The deadline is the same date in the next month, or that month's last day if it has no such
    date (31 January gives 28 February). If that is a weekend or a bank holiday, the deadline is
    the next working day. Raises ValueError beyond the bank holiday snapshot, as foi_deadline does.
    """
    if received.month == 12:
        year, month = received.year + 1, 1
    else:
        year, month = received.year, received.month + 1
    day = date(year, month, min(received.day, calendar.monthrange(year, month)[1]))
    _require_known_holidays(day)
    while not is_working_day(day):
        day += timedelta(days=1)
        _require_known_holidays(day)
    return day
