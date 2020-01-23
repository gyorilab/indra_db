import re
import json
import boto3

from collections import defaultdict as dd


class StageData(object):
    def __init__(self):
        self._dict = dd(lambda: dd(lambda: dd(lambda: dd(lambda: 0))))

    def _update_dict(self, stage, flavor, key, day, value):
        if flavor:
            self._dict[stage][flavor][key][day] += value
        self._dict[stage]['total'][key][day] += value

    def add(self, stage, flavor, day, counts):
        for key, value in counts.items():
            self._update_dict(stage, flavor, key, day, value)
        self._update_dict(stage, flavor, 'jobs', day, 1)

    def get_json(self):
        return json.loads(json.dumps(self._dict))


def digest_s3_files(bucket, prefix):
    s3 = boto3.client('s3')
    res = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
    day_prefixes = [p for d in res['CommonPrefixes'] for p in d.values()]
    runtime_data = []
    stage_data = StageData()
    for day_prefix in day_prefixes:
        print(day_prefix)
        day_res = s3.list_objects_v2(Bucket=bucket, Prefix=day_prefix)
        day_keys = [d['Key'] for d in day_res['Contents']]

        day = day_prefix.split('/')[-2]
        day_runtimes = {'day': day, 'times': []}

        for key in day_keys:
            file_res = s3.get_object(Bucket=bucket, Key=key)
            data = file_res['Body'].read()

            # Get metadata
            m = re.match('%s/([0-9]+)/(\w*?)/?(\w+)_([0-9]+).json' % prefix,
                         key)
            if not m:
                continue
            data = json.loads(data)
            day, stage, flavor, time = m.groups()
            if not stage:
                stage = flavor
                flavor = None

            # Update runtime
            day_runtimes['times'].append({'stage': stage, 'type': flavor,
                                          'times': [data['timing']['start'],
                                                    data['timing']['end']]})

            # Handle stages
            stage_data.add(stage, flavor, day, data['counts'])

        runtime_data.append(day_runtimes)
    return runtime_data, stage_data.get_json()
