"""Fixed elapsed-time units, measured from an origin at day zero.

Unit bins include their lower boundary and exclude their upper boundary:
month zero covers days 0–29, and year zero covers days 0–364.
Calendar dates and birthday-based ages use calendar arithmetic instead.
Twelve fixed 30-day months are 360 days, not one fixed 365-day year.
"""

from typing import Final

DAYS_PER_WEEK: Final[int] = 7
DAYS_PER_MONTH: Final[int] = 30
DAYS_PER_YEAR: Final[int] = 365
