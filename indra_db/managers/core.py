import functools
import json
import logging
import traceback
from datetime import datetime, timedelta

import boto3

logger = logging.getLogger(__name__)

# TODO: Make this a config var
S3_DATA_LOC = {'bucket': 'bigmech', 'prefix': 'indra-db/managers/'}
DATE_FMT = '%Y%m%d_%H%M%S'


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
        key = S3_DATA_LOC['prefix'] + self._label
        if self._sub_label:
            key += '/' + self._sub_label
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
