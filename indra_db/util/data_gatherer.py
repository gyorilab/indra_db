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
            self._dict[stage][key][flavor][day] += value
        self._dict[stage][key]['total'][day] += value

    def add(self, stage, flavor, day, counts):
        for key, value in counts.items():
            self._update_dict(stage, flavor, key, day, value)
        self._update_dict(stage, flavor, 'jobs', day, 1)

    def get_json(self):
        ret = {}
        for stage, stage_data in self._dict.items():
            ret[stage] = {}
            for key, key_data in stage_data.items():
                ret[stage][key] = {}
                for flavor, flavor_data in key_data.items():
                    ret[stage][key][flavor] = []
                    for day, value in flavor_data.items():
                        ret[stage][key][flavor].append([day, value])
        return ret


class DayStack(object):
    def __init__(self):
        self._data = {}

    def __str__(self):
        s = ''
        for k, v in self._data.items():
            s += '%s: %s\n' % (k, v)
        return s

    def __repr__(self):
        return 'DayStack(' + repr(self._data) + ')'

    def add(self, num, datum):
        if not isinstance(num, int):
            raise ValueError("`num` must be of type `int`.")
        if num not in self._data:
            self._data[num] = []
        self._data[num].append(datum)

    def pop(self):
        if 0 not in self._data:
            ret = []
        else:
            ret = self._data[0]
        self._data = {k-1: v for k, v in self._data.items() if k > 0}
        return ret


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
    day_stack = DayStack()
    stage_data = StageData()
    for day_prefix in day_prefixes:
        logger.info("Processing: %s" % day_prefix)
        day_res = s3.list_objects_v2(Bucket=bucket, Prefix=day_prefix)
        day_keys = [d['Key'] for d in day_res['Contents']]

        day = day_prefix.split('/')[-2]
        day_obj = datetime.strptime(day, '%Y%m%d')
        day_ts = day_obj.timestamp()*1000
        day_runtimes = {'day_str': day_obj.strftime('%b %d %Y'),
                        'day_ts': day_ts,
                        'times': dd(lambda: dd(list))}

        # Pick up any wrap-arounds.
        for stage, flavor, time_pair in day_stack.pop():
            if flavor:
                day_runtimes['times'][stage][flavor].append(time_pair)
            day_runtimes['times'][stage]['all'].append(time_pair)

        # Get the datetime data for "today".
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
            start = (data['timing']['start'] - day_ts)/div_factor + 5
            end = (data['timing']['end'] - day_ts)/div_factor + 5

            # Check to see if this time wraps around, update day_stack if so.
            n_days_future = int(end // 24)
            for days_future in range(n_days_future - 1):
                day_stack.add(days_future, (stage, flavor, [0., 24.]))
            if n_days_future and end % 24:
                day_stack.add(n_days_future - 1, (stage, flavor, [0., end % 24]))
            if n_days_future:
                end = 24.

            # Add this runtime pair.
            time_pair = [start, end]
            if flavor:
                day_runtimes['times'][stage][flavor].append(time_pair)
            day_runtimes['times'][stage]['all'].append(time_pair)

            # Handle stages
            stage_data.add(stage, flavor, day_ts, data['counts'])

        runtime_data.append(day_runtimes)

    # Dump the digests on s3.
    s3.put_object(Bucket=bucket, Key=prefix + 'runtimes.json',
                  Body=json.dumps(runtime_data))
    for stage, data in stage_data.get_json().items():
        s3.put_object(Bucket=bucket, Key=prefix + stage + '.json',
                      Body=json.dumps(data))
    return

