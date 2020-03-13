import re
from os import path


class S3Path(object):
    """A simple object to make it easier to manage s3 locations."""
    def __init__(self, bucket, key=None):
        if not isinstance(bucket, str):
            raise ValueError("Bucket must be a string, not %s." % type(bucket))
        self.bucket = bucket
        if key is not None:
            if not isinstance(key, str):
                raise ValueError("Key must be a string, not %s." % type(key))
            elif key.startswith('/'):
                key = key[1:]
        self.key = key

    def kw(self):
        ret = {'Bucket': self.bucket}
        if self.key:
            ret['Key'] = self.key
        return ret

    def get_element_path(self, subkey):
        return self.from_key_parts(self.bucket, self.key, subkey)

    @classmethod
    def from_key_parts(cls, bucket, *key_elements):
        key = path.join(*key_elements)
        return cls(bucket, key)

    @classmethod
    def from_string(cls, s3_key_str):
        patt = re.compile('s3://([a-z0-9]+)/(.*)')
        m = patt.match(s3_key_str)
        if m is None:
            raise ValueError("Invalid format for s3 path: %s" % s3_key_str)
        bucket, key = m.groups()
        if not key:
            key = None
        return cls(bucket, key)

    def to_string(self):
        return 's3://{bucket}/{key}'.format(bucket=self.bucket, key=self.key)

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return 'S3Path({bucket}, {key})'.format(bucket=self.bucket,
                                                key=self.key)
