__all__ = ['CopyManager', 'LazyCopyManager', 'PushCopyManager',
           'ReturningCopyManager']

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

    @staticmethod
    def _stringify_cols(cols):
        if not isinstance(cols, list) and not isinstance(cols, tuple):
            raise ValueError(f"Argument `cols` must be a list or tuple, got: "
                             f"{type(cols)}")
        return '", "'.join(cols)

    def _fmt_sql(self, sql_fmt):
        columns = self._stringify_cols(self.cols)
        sql = sql_fmt.format(schema=self.schema, table=self.table,
                             cols=columns)
        return sql

    def _get_insert_sql(self):
        cmd_fmt = self._merge_fmt
        if self.constraint:
            cmd_fmt += 'ON CONSTRAINT "%s" ' % self.constraint
        cmd_fmt += 'DO NOTHING;\n'
        return self._fmt_sql(cmd_fmt)

    def _get_copy_sql(self):
        return self._fmt_sql(self._fill_tmp_fmt)

    def _get_sql(self):
        return '\n'.join([self._get_copy_sql(), self._get_insert_sql()])

    def _get_skipped(self, num, order_by, return_cols=None):
        cursor = self.conn.cursor()
        inp_cols = self._stringify_cols(self.cols)
        if return_cols:
            ret_cols = self._stringify_cols(return_cols)
        else:
            ret_cols = inp_cols
        diff_sql = (
            f'SELECT "{ret_cols}" FROM\n'
            f'(SELECT "{inp_cols}" FROM "tmp_{self.table}"\n'
            f' EXCEPT\n'
            f' (SELECT "{inp_cols}"\n'
            f'  FROM "{self.schema}"."{self.table}"\n'
            f'  ORDER BY "{order_by}" DESC\n'
            f'  LIMIT {num})) as t;'
        )
        logger.debug(diff_sql)
        cursor.execute(diff_sql)
        res = cursor.fetchall()
        return res

    def copystream(self, datastream):
        sql = self._get_sql()

        logger.debug(sql)
        cursor = self.conn.cursor()
        try:
            cursor.copy_expert(sql, datastream)
        except Exception as e:
            templ = "error doing lazy binary copy into {0}.{1}:\n{2}"
            e.message = templ.format(self.schema, self.table, e)
            raise e
        return


class ReturningCopyManager(LazyCopyManager):
    """Perform a lazy copy, and retrieve the new primary IDs generated."""

    def __init__(self, conn, table, input_cols, return_cols, constraint=None):
        super(ReturningCopyManager, self).__init__(conn, table, input_cols,
                                                   constraint)
        self.return_cols = return_cols
        return

    def report_copy(self, data, order_by=None, return_cols=None,
                    fobject_factory=tempfile.TemporaryFile):
        self.copy(data, fobject_factory)
        ret_cols = self._insert_with_return()
        skipped_rows = self._get_skipped(len(data), order_by, return_cols)
        return ret_cols, skipped_rows

    def _get_sql(self):
        return self._get_copy_sql()

    def _get_insert_sql(self):
        sql = super(ReturningCopyManager, self)._get_insert_sql()
        sql = sql.strip()
        if sql[-1] == ';':
            sql = sql[:-1]
        sql += f"\nRETURNING \"{self._stringify_cols(self.return_cols)}\";\n"
        return sql

    def _get_existing(self, compare_cols):
        cursor = self.conn.cursor()

        ret_col_str = '", "'.join(f't"."{col}' for col in self.return_cols)
        comparisons = []
        for col_pair in compare_cols:
            tmp_inp_cols = '", "'.join(f'tmp_t"."{col}' for col in col_pair)
            t_inp_cols = '", "'.join(f't"."{col}' for col in col_pair)
            comparisons.append(f'("{tmp_inp_cols}") = ("{t_inp_cols}")')
        where_clause = ' OR '.join(comparisons)
        sql = (
            f'SELECT "{ret_col_str}" '
            f'FROM "tmp_{self.table}" AS tmp_t,\n'
            f'     "{self.table}" AS t\n'
            f'WHERE {where_clause}\n'
        )
        logger.debug(sql)
        cursor.execute(sql)
        res = cursor.fetchall()
        return res

    def _insert_with_return(self):
        cursor = self.conn.cursor()
        sql = self._get_insert_sql()
        logger.debug(sql)
        cursor.execute(sql)
        res = cursor.fetchall()
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

    def _get_report_sql(self):
        return self._fmt_sql(self._end_table_fmt)

    def _get_insert_sql(self):
        cmd_fmt = self._merge_fmt
        update = ', '.join('{0} = EXCLUDED.{0}'.format(c)
                           for c in self.cols)
        cmd_fmt += 'ON CONSTRAINT "%s" DO UPDATE SET %s;\n' \
                   % (self.constraint, update)
        return self._fmt_sql(cmd_fmt)

    def _get_sql(self):
        if self.reporting:
            return '\n'.join([self._get_copy_sql(), self._get_report_sql(),
                                 self._get_insert_sql()])
        else:
            return '\n'.join([self._get_copy_sql(), self._get_insert_sql()])

    def _fmt_sql(self, sql_fmt):
        columns = self._stringify_cols(self.cols)
        sql = sql_fmt.format(schema=self.schema, table=self.table,
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

