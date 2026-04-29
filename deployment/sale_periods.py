"""Sale periods for each market/variety/grade combination.

Extracted from web/forecast_bp.py. Defines the seasonal window when each
variety is typically sold. Cross-year periods (e.g., Sep-Jan) are handled
by the inference engine.
"""

SALE_PERIODS = {
    # Azadpur
    ("Azadpur", "American", "A"): {"start": "09-01", "end": "01-30", "years": list(range(2012, 2026))},
    ("Azadpur", "American", "B"): {"start": "09-01", "end": "01-30", "years": list(range(2012, 2026))},
    ("Azadpur", "Delicious", "A"): {"start": "09-01", "end": "03-30", "years": list(range(2012, 2026))},
    ("Azadpur", "Delicious", "B"): {"start": "09-01", "end": "03-30", "years": list(range(2012, 2026))},

    # Shopian
    ("Shopian", "American", "A"): {"start": "10-01", "end": "11-30", "years": list(range(2017, 2026))},
    ("Shopian", "American", "B"): {"start": "10-01", "end": "11-30", "years": list(range(2017, 2026))},
    ("Shopian", "Delicious", "A"): {"start": "09-15", "end": "12-31", "years": list(range(2017, 2026))},
    ("Shopian", "Delicious", "B"): {"start": "09-15", "end": "12-31", "years": list(range(2017, 2026))},

    # Sopore
    ("Sopore", "American", "A"): {"start": "08-01", "end": "02-28", "years": list(range(2015, 2026))},
    ("Sopore", "American", "B"): {"start": "08-01", "end": "02-28", "years": list(range(2015, 2026))},
    ("Sopore", "Delicious", "A"): {"start": "08-01", "end": "02-28", "years": list(range(2015, 2026))},
    ("Sopore", "Delicious", "B"): {"start": "08-01", "end": "02-28", "years": list(range(2015, 2026))},
}


def get_sale_period(market: str, variety: str, grade: str):
    """Get sale period info for a market/variety/grade combination."""
    key = (market, variety, grade)
    return SALE_PERIODS.get(key)


def is_date_in_sale_period(date, start_md: str, end_md: str) -> bool:
    """Check if a date falls within a sale period (month-day range).

    Handles cross-year periods (e.g., Sep 01 to Jan 30).
    """
    import pandas as pd

    dt = pd.Timestamp(date)
    month_day = dt.strftime("%m-%d")

    if end_md >= start_md:
        # Same-year period (e.g., May 01 to Jun 30)
        return start_md <= month_day <= end_md
    else:
        # Cross-year period (e.g., Sep 01 to Jan 30)
        return month_day >= start_md or month_day <= end_md


def generate_sale_dates_2026(start_md: str, end_md: str) -> list:
    """Generate all dates in 2026/2027 that fall within the sale period.

    For cross-year periods (e.g., Sep-Jan), generates dates from
    the start month in 2026 through the end month in 2027.
    For same-year periods, generates dates within 2026.
    """
    import pandas as pd

    sm, sd = map(int, start_md.split("-"))
    em, ed = map(int, end_md.split("-"))
    cross_year = (em, ed) < (sm, sd)

    if not cross_year:
        # Same-year period (e.g., Oct 01 to Nov 30)
        start_date = pd.Timestamp(2026, sm, sd)
        end_date = pd.Timestamp(2026, em, ed)
    else:
        # Cross-year period (e.g., Sep 01 to Jan 30)
        start_date = pd.Timestamp(2026, sm, sd)
        end_date = pd.Timestamp(2027, em, ed)

    # Generate daily dates
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Filter to only dates within the sale period (handles edge cases)
    sale_dates = [d for d in dates if is_date_in_sale_period(d, start_md, end_md)]

    return sale_dates

