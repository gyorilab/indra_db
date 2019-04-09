import sys
import json
from collections import defaultdict

from indra_db.util import get_db


def get_daily_counts(dates):
    counts = defaultdict(lambda: 0)
    for date in dates:
        counts[date.date()] += 1
    return zip(*sorted(counts.items()))


def get_all_daily_counts(db):
    data_json = {}

    # Daily update counts
    for table in [db.TextRef, db.TextContent, db.Reading, db.RawStatements,
                  db.PAStatements]:
        if table == db.TextContent:
            date_col = table.insert_date
        else:
            date_col = table.create_date
        print('Processing daily dates for %s.' % table.__tablename__)
        dates = [date for date, in db.select_all(date_col) if date is not None]
        t, n = get_daily_counts(dates)
        data_json[table.__tablename__ + '_daily_counts'] = [t, n]

    return

def main(db_name):
    db = get_db(db_name)
    data_json = {}

    data_json.update(get_all_daily_counts(db))

    print('Dumping json...')
    with open(db_name + '_stats.json', 'w') as f:
        json.dumps(data_json, f, indent=2)

    return


if __name__ == '__main__':
    db_name = sys.argv[1]
    main(db_name)

