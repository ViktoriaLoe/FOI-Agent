"""The statutory FOI response deadline."""

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


def foi_deadline(received: date) -> date:
    """The 20th working day after receipt.

    Day one is the first working day after the request is received, including when it arrives on
    a non-working day (ICO guidance). Raises ValueError if the deadline would fall beyond the
    years the bank holiday snapshot covers, rather than silently ignoring unknown holidays.
    """
    last_covered = max(uk_bank_holidays()).year
    day = received
    counted = 0
    while counted < WORKING_DAYS:
        day += timedelta(days=1)
        if day.year > last_covered:
            raise ValueError(
                f"Bank holidays are only known up to {last_covered}: "
                "refresh data/bank_holidays.json from https://www.gov.uk/bank-holidays.json"
            )
        if is_working_day(day):
            counted += 1
    return day
