__all__ = ['CopyManager', 'LazyCopyManager', 'PushCopyManager']

import logging
import tempfile

from pgcopy import CopyManager


logger = logging.getLogger(__name__)


class LazyCopyManager(CopyManager):
    """A copy manager that ignores entries which violate constraints."""
    _fill_tmp_fmt = ('CREATE TEMP TABLE "tmp_{table}"\n'
                     'ON COMMIT DROP\n'
                     'AS SELECT "{cols}" FROM "{schema}"."{table}"\n'
                     'WITH NO DATA;\n'
                     'COPY "tmp_{table}" ("{cols}")\n'
                     'FROM STDIN WITH BINARY;')

    _merge_fmt = ('INSERT INTO "{schema}"."{table}" ("{cols}")\n'
                  'SELECT "{cols}"\n'
                  'FROM "tmp_{table}" ON CONFLICT ')

    def __init__(self, conn, table, cols, constraint=None):
        super().__init__(conn, table, cols)
        self.constraint = constraint
        return

    def report_copy(self, data, order_by=None, return_cols=None,
                    fobject_factory=tempfile.TemporaryFile):
        self.copy(data, fobject_factory)
        return self._get_skipped(len(data), order_by, return_cols)

    def _stringify_cols(self, cols):
        return '", "'.join(cols)

    def _get_sql(self):
        cmd_fmt = '\n'.join([self._fill_tmp_fmt, self._merge_fmt])
        if self.constraint:
            cmd_fmt += 'ON CONSTRAINT "%s" ' % self.constraint
        cmd_fmt += 'DO NOTHING;\n'

        # Fill in the format.
        columns = self._stringify_cols(self.cols)
        sql = cmd_fmt.format(schema=self.schema, table=self.table,
                             cols=columns)
        return sql

    def _get_skipped(self, num, order_by, return_cols=None):
        cursor = self.conn.cursor()
        inp_cols = self._stringify_cols(self.cols)
        if return_cols:
            ret_cols = self._stringify_cols(return_cols)
        else:
            ret_cols = inp_cols
        diff_sql = (
            'SELECT "{ret_cols}" FROM\n'
            '(SELECT "{cols}" FROM "tmp_{table}"\n'
            ' EXCEPT\n'
            ' (SELECT "{cols}"\n'
            '  FROM "{schema}"."{table}"\n'
            '  ORDER BY "{order_by}" DESC\n'
            '  LIMIT {num})) as t;'
        ).format(cols=inp_cols, table=self.table, schema=self.schema,
                 ret_cols=ret_cols, order_by=order_by, num=num)
        logger.debug(diff_sql)
        cursor.execute(diff_sql)
        res = cursor.fetchall()
        return res

    def copystream(self, datastream):
        sql = self._get_sql()

        logger.debug(sql)
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
    _end_table_fmt = ('CREATE TEMP TABLE "end_{table}" '
                      'ON COMMIT DROP '
                      'AS SELECT "{order_by}" FROM "{schema}"."{table}" '
                      'ORDER BY "{order_by}" DESC LIMIT 1; ')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.constraint:
            raise ValueError("A constraint is required if you are updating "
                             "on-conflict.")
        self.reporting = False
        self.order_by = None
        return

    def _get_sql(self):
        if self.reporting:
            cmd_fmt = '\n'.join([self._fill_tmp_fmt, self._end_table_fmt,
                                 self._merge_fmt])
        else:
            cmd_fmt = '\n'.join([self._fill_tmp_fmt, self._merge_fmt])
        update = ', '.join('{0} = EXCLUDED.{0}'.format(c)
                           for c in self.cols)
        cmd_fmt += 'ON CONSTRAINT "%s" DO UPDATE SET %s;\n' \
                   % (self.constraint, update)
        columns = self._stringify_cols(self.cols)
        sql = cmd_fmt.format(schema=self.schema, table=self.table,
                             cols=columns, order_by=self.order_by)
        return sql

    def _get_report(self, return_cols=None):
        cursor = self.conn.cursor()
        inp_cols = self._stringify_cols(self.cols)
        if return_cols:
            ret_cols = self._stringify_cols(return_cols)
        else:
            ret_cols = inp_cols
        diff_sql = (
            'SELECT "{ret_cols}"\n'
            'FROM\n'
            '(SELECT "{cols}" FROM "tmp_{table}"\n'
            ' EXCEPT\n'
            ' (SELECT "{cols}"\n'
            '  FROM "{schema}"."{table}"\n'
            '  WHERE "{order_by}" > (SELECT "{order_by}" '
            '                        FROM "end_{table}"))) AS t;'
        ).format(ret_cols=ret_cols, order_by=self.order_by, table=self.table,
                 schema=self.schema, cols=inp_cols)
        logger.debug(diff_sql)
        cursor.execute(diff_sql)
        res = cursor.fetchall()
        return res

    def report_copy(self, data, order_by=None, return_cols=None,
                    fobject_factory=tempfile.TemporaryFile):
        self.reporting = True
        self.order_by = order_by
        self.copy(data, fobject_factory)
        updated = self._get_report(return_cols)
        return updated

