from collections import defaultdict

def get_daily_counts(dates):
    counts = defaultdict(lambda: 0)
    for date in dates:
        counts[date.date()] += 1
    return zip(*sorted(counts.items()))


