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

    def __lt__(self, other):
        if not isinstance(other, S3Path):
            raise ValueError(f"Cannot compare with type \"{type(other)}\".")
        return self.to_string() < other.to_string()

    def __eq__(self, other):
        if not isinstance(other, S3Path):
            raise ValueError(f"Cannot compare with type \"{type(other)}\".")
        return self.to_string() == other.to_string()

    def __le__(self, other):
        if not isinstance(other, S3Path):
            raise ValueError(f"Cannot compare with type \"{type(other)}\".")
        return self.to_string() <= other.to_string()

    def kw(self, prefix=False):
        ret = {'Bucket': self.bucket}
        if self.key:
            if prefix:
                ret['Prefix'] = self.key
            else:
                ret['Key'] = self.key
        return ret

    def get(self, s3):
        if not self.key:
            raise ValueError("Cannot get key-less s3 path.")
        return s3.get_object(**self.kw())

    def put(self, s3, body):
        if not self.key:
            raise ValueError("Cannot 'put' to a key-less s3 path.")
        return s3.put_bject(Body=body, **self.kw())

    def list_objects(self, s3):
        raw_res = s3.list_objects_v2(**self.kw(prefix=True))
        return [self.__class__(self.bucket, e['Key'])
                for e in raw_res['Contents']]

    def list_prefixes(self, s3):
        raw_res = s3.list_objects_v2(Delimiter='/', **self.kw(prefix=True))
        return [self.__class__(self.bucket, e['Prefix'])
                for e in raw_res['CommonPrefixes']]

    def exists(self, s3):
        return 'Contents' in s3.list_objects_v2(**self.kw(prefix=True))

    def delete(self, s3):
        return s3.delete_object(**self.kw())

    def get_element_path(self, *subkeys):
        args = []
        if self.key is not None:
            args.append(self.key)
        args += subkeys
        return self.from_key_parts(self.bucket, *args)

    @classmethod
    def from_key_parts(cls, bucket, *key_elements):
        key = path.join(*key_elements)
        return cls(bucket, key)

    @classmethod
    def from_string(cls, s3_key_str):
        patt = re.compile('s3://([a-z0-9\-.]+)/(.*)')
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
