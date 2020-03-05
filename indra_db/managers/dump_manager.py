import json
import boto3
from datetime import datetime

from indra_db.belief import get_belief
from indra_db.config import get_s3_dump
from indra_db.util.dump_sif import dump_sif

dump_names = ['sif', 'belief']


class Dumper(object):
    name = NotImplemented
    fmt = NotImplemented

    @classmethod
    def _gen_s3_name(cls):
        s3_config = get_s3_dump()
        dt_ts = datetime.now().strftime('%Y-%m-%d')
        key = s3_config['prefix'] + '%s-%s.%s' % (cls.name, dt_ts, cls.fmt)
        return s3_config['bucket'], key

    def dump(self, db):
        raise NotImplementedError()


class Sif(Dumper):
    name = 'sif'

    def dump(self, db):
        bucket, key = self._gen_s3_name()
        dump_sif('s3:' + bucket + '/' + key, ro=db)


class Belief(Dumper):
    name = 'belief'

    def dump(self, db):
        bucket, key = self._gen_s3_name()
        belief_dict = get_belief(db)
        s3 = boto3.client('s3')
        s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(belief_dict))
