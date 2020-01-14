import functools
import json
import logging
import traceback
from datetime import datetime

import boto3

logger = logging.getLogger(__name__)

# TODO: Make this a config var
S3_DATA_LOC = {'bucket': 'bigmech', 'prefix': 'indra-db/manager-data/'}
RESERVED_KEYS = ['error_info', 'start_time', 'end_time']
DATE_FMT = '%Y%m%d_%H%M%S'


class Manager(object):
    data_inits = NotImplemented
    reserved_data_inits = {
        'error_info': dict,
        'end_time': str,
        'start_time': str,
        'duration': int
    }
    stage_name = NotImplemented

    def __init__(self):
        self.stats = {data_name: data_type()
                      for data_name, data_type in self.data_inits.items()}

    def get_stat_file_key(self):
        s3_suffix = '%s/%s.json' % (self.date_id, self.stage_name)
        s3_key = S3_DATA_LOC['prefix'] + s3_suffix
        return s3_key

    def get_stats(self):
        return self.stats.copy()


class ManagerEnv(object):
    @classmethod
    def wrap(cls, func):
        @functools.wraps(func)
        def decorated(obj, *args, **kwargs):
            with cls(obj):
                return func(obj, *args, **kwargs)
        return decorated

    def __init__(self, manager):
        self.manager = manager

    def __enter__(self):
        self.start_time = datetime.utcnow()
        logger.info("Entering manager env.")

    def __exit__(self, err_type, err, tb):
        logger.info("Leaving manager env with error type: %s" % err_type)
        s3 = boto3.client('s3')

        # Get the s3 key from the manager
        key = self.manager.get_stat_file_key()

        # Get the stats from the manager
        stats = self.manager.get_stats()
        s3.put_object(Bucket=S3_DATA_LOC['bucket'], Key=key,
                      Body=json.dumps(stats))

        # Add some generic stats.
        gen_stats = {}
        err_info = {
            'error_type': str(err_type),
            'error_value': str(err),
            'traceback': ''.join(traceback.format_exception(err_type, err, tb))
        }
        gen_stats['error_info'] = err_info
        self.end_time = datetime.utcnow()
        gen_stats['end_time'] = self.end_time.strftime(DATE_FMT)
        gen_stats['start_time'] = self.manager.date_id

        # Dump the gen_stats on s3.
        s3.put_object(Bucket=S3_DATA_LOC['bucket'], Key=key,
                      Body=json.dumps(gen_stats))
