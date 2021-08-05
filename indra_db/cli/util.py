from datetime import datetime


def format_date(dt):
    if not isinstance(dt, datetime):
        return dt
    return dt.strftime("%Y %b %d %I:%M%p")
