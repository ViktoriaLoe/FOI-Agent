"""The statutory FOI response deadline."""

from datetime import date, timedelta

WORKING_DAYS = 20


def foi_deadline(received: date) -> date:
    """The 20th working day after receipt.

    Day one is the first working day after the request is received, including when it arrives on
    a non-working day (ICO guidance).
    """
    # Bank holidays are not yet excluded: only weekends are skipped.
    day = received
    counted = 0
    while counted < WORKING_DAYS:
        day += timedelta(days=1)
        if day.weekday() < 5:
            counted += 1
    return day
