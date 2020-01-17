__all__ = ['CopyManager', 'LazyCopyManager', 'PushCopyManager']

import tempfile

from pgcopy import CopyManager


class LazyCopyManager(CopyManager):
    """A copy manager that ignores entries which violate constraints."""
    base_cmd_fmt = ('CREATE TEMP TABLE "tmp_{table}" '
                    'ON COMMIT DROP '
                    'AS SELECT "{cols}" FROM "{schema}"."{table}" '
                    'WITH NO DATA; '
                    '\n'
                    'COPY "tmp_{table}" ("{cols}") '
                    'FROM STDIN WITH BINARY; '
                    '\n'
                    'INSERT INTO "{schema}"."{table}" ("{cols}") '
                    'SELECT "{cols}" '
                    'FROM "tmp_{table}" ON CONFLICT')

    def __init__(self, conn, table, cols, constraint=None):
        super(LazyCopyManager, self).__init__(conn, table, cols)
        self.constraint = constraint
        return

    def report_copy(self, data, order_by=None, return_cols=None,
                    fobject_factory=tempfile.TemporaryFile):
        datastream = fobject_factory()
        self.writestream(data, datastream)
        datastream.seek(0)
        self.copystream(datastream)
        datastream.close()

        res = self._get_skipped(len(data), order_by, return_cols)
        return res

    def _stringify_cols(self, cols):
        return '", "'.join(cols)

    def _get_sql_fmt(self):
        cmd_fmt = self.base_cmd_fmt
        if self.constraint:
            cmd_fmt += 'ON CONSTRAINT "%s" ' % self.constraint
        cmd_fmt += 'DO NOTHING;\n'
        return cmd_fmt

    def _get_skipped(self, num, order_by, return_cols=None):
        cursor = self.conn.cursor()
        inp_cols = self._stringify_cols(self.cols)
        if self.return_cols:
            ret_cols = self._stringify_cols(return_cols)
        else:
            ret_cols = inp_cols
        diff_sql = (
            'SELECT "{ret_cols}" FROM\n'
            '"tmp_{table}"\n'
            'EXCEPT\n'
            '(SELECT "{cols}"\n'
            ' FROM "{table}"\n'
            ' ORDER BY "{order_by}" DESC\n'
            ' LIMIT {num});'
        ).format(cols=inp_cols, table=self.table, ret_cols=ret_cols,
                 order_by=order_by, num=num)
        print(diff_sql)
        cursor.execute(diff_sql)
        res = cursor.fetchall()
        return res

    def copystream(self, datastream):
        cmd_fmt = self._get_sql_fmt()

        # Fill in the format.
        columns = self._stringify_cols(self.cols)
        sql = cmd_fmt.format(schema=self.schema,
                             table=self.table,
                             cols=columns)
        print(sql)
        cursor = self.conn.cursor()
        res = None
        try:
            cursor.copy_expert(sql, datastream)
        except Exception as e:
            templ = "error doing lazy binary copy into {0}.{1}:\n{2}"
            e.message = templ.format(self.schema, self.table, e)
            raise e
        return res


class PushCopyManager(LazyCopyManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.constraint:
            raise ValueError("A constraint is required if you are updating "
                             "on-conflict.")

    def _get_sql_fmt(self):
        cmd_fmt = self.base_cmd_fmt
        update = ', '.join('{0} = EXCLUDED.{0}'.format(c)
                           for c in self.cols)
        cmd_fmt += 'ON CONSTRAINT "%s" DO UPDATE SET %s;\n' \
                   % (self.constraint, update)
        return cmd_fmt
