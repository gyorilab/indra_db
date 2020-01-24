import re
import json
import boto3
import logging
import functools
import traceback
from datetime import datetime, timedelta
from collections import defaultdict as dd


logger = logging.getLogger(__name__)

# TODO: Make this a config var
S3_DATA_LOC = {'bucket': 'bigmech', 'prefix': 'indra-db/managers/'}
DAY_FMT = '%Y%m%d'
TIME_FMT = '%H%M%S'


class DGContext(object):
    def __init__(self, gatherer):
        self.gatherer = gatherer

    def __enter__(self):
        self.gatherer.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.gatherer.dump(exc_type, exc_val, exc_tb)

    @classmethod
    def wrap(cls, gatherer, sub_label=None):
        def sub_wrap(func):
            @functools.wraps(func)
            def decorated(*args, **kwargs):
                if sub_label:
                    gatherer.set_sub_label(sub_label)
                with cls(gatherer):
                    return func(*args, **kwargs)
            return decorated
        return sub_wrap


class DataGatherer(object):
    def __init__(self, label, counts_fields):
        self._label = label
        self._sub_label = None
        self._counts_fields = counts_fields
        self._timing = self._counts = self._error = None
        self._in_context = False
        return

    def set_sub_label(self, sub_label):
        self._sub_label = sub_label
        return

    def start(self):
        self._timing = {
            'start': datetime.utcnow(),
            'end': None,
            'dur': None
        }
        self._counts = dict.fromkeys(self._counts_fields, 0)
        self._in_context = True
        return

    def add(self, field, num=1):
        if field not in self._counts:
            raise ValueError('Unexpected field: %s. Should be one of: %s.'
                             % (field, self._counts_fields))

        if not self._in_context:
            raise RuntimeError('Attempted to update value %s out of context.'
                               % field)
        self._counts[field] += num
        return

    def dump(self, err_type, err, tb):
        logger.info("Leaving manager env with error type: %s" % err_type)
        s3 = boto3.client('s3')

        if err_type:
            err_msg_lines = traceback.format_exception(err_type, err, tb)
            self._error = {
                'type': err_type.__name__,
                'value': str(err),
                'traceback': ''.join(err_msg_lines)
            }

        self._timing['end'] = datetime.utcnow()
        self._timing['dur'] = self._timing['end'] - self._timing['start']

        # Get the s3 key from the manager
        key = S3_DATA_LOC['prefix'] + self._timing['start'].strftime(DAY_FMT)
        key += '/' + self._label
        if self._sub_label:
            key += '/' + self._sub_label
        key += '_' + self._timing['start'].strftime(TIME_FMT)
        key += '.json'

        # Get the stats from the manager
        stats = {'timing': self._make_timing_json(),
                 'counts': self._counts,
                 'error': self._error}
        s3.put_object(Bucket=S3_DATA_LOC['bucket'], Key=key,
                      Body=json.dumps(stats))
        self._in_context = False
        return

    def _make_timing_json(self):
        timing_res = {}
        for name, value in self._timing.items():
            if isinstance(value, datetime):
                timing_res[name] = value.timestamp()*1000
            elif isinstance(value, timedelta):
                timing_res[name] = value.total_seconds()*1000
            else:
                logger.warning("Expected datetime related object, but got %s."
                               % type(value))
                timing_res[name] = value
        return timing_res


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


def digest_s3_files():
    s3 = boto3.client('s3')
    bucket = S3_DATA_LOC['bucket']
    prefix = S3_DATA_LOC['prefix']

    patt = re.compile(prefix + '([0-9]+)/(\w*?)/?(\w+)_([0-9]+).json')

    # Get a list of the prefixes for each day.
    res = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
    day_prefixes = [p for d in res['CommonPrefixes'] for p in d.values()]

    # Build up our data files.
    runtime_data = []
    stage_data = StageData()
    for day_prefix in day_prefixes:
        logger.info("Processing: %s" % day_prefix)
        day_res = s3.list_objects_v2(Bucket=bucket, Prefix=day_prefix)
        day_keys = [d['Key'] for d in day_res['Contents']]

        day = day_prefix.split('/')[-2]
        day_obj = datetime.strptime(day, '%Y%m%d')
        day_ts = day_obj.timestamp()*1000
        day_runtimes = {'day': day_ts,
                        'times': dd(lambda: dd(list))}

        for key in day_keys:
            file_res = s3.get_object(Bucket=bucket, Key=key)
            data = file_res['Body'].read()

            # Get metadata
            m = patt.match(key)
            if not m:
                logger.warning("No match for %s." % key)
                continue
            data = json.loads(data)
            day, stage, flavor, time = m.groups()
            if not stage:
                stage = flavor
                flavor = None

            # Update runtime
            div_factor = 3600*1000
            time_pair = [(data['timing']['start'] - day_ts)/div_factor + 5,
                         (data['timing']['end'] - day_ts)/div_factor + 5]
            if flavor:
                day_runtimes['times'][stage][flavor].append(time_pair)
            day_runtimes['times'][stage]['all'].append(time_pair)

            # Handle stages
            stage_data.add(stage, flavor, day, data['counts'])

        runtime_data.append(day_runtimes)
    return runtime_data, stage_data.get_json()

