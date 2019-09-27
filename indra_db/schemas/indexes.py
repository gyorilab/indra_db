__all__ = ['BtreeIndex', 'StringIndex']


class BtreeIndex(object):
    def __init__(self, name, colname, opts=None):
        self.name = name
        self.colname = colname
        contents = colname
        if opts is not None:
            contents += ' ' + opts
        self.definition = ('btree (%s)' % contents)


class StringIndex(BtreeIndex):
    def __init__(self, name, colname):
        opts = 'COLLATE pg_catalog."en_US.utf8" varchar_ops ASC NULLS LAST'
        super().__init__(name, colname, opts)

