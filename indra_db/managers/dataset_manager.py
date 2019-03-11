import logging

from indra_db.util import insert_db_stmts

logger = logging.getLogger(__name__)


class DatasetManager(object):
    """This is a class to lay out the methods for updating a dataset."""
    name = NotImplemented

    def upload(self, db):
        """Upload the content for this dataset into the database."""
        dbid = db.select_one(db.DBInfo.id, db.DBInfo.db_name == self.name)
        stmts = self._get_statements(db)
        insert_db_stmts(db, stmts, dbid)
        return

    def _get_statements(self, db):
        raise NotImplementedError("Statement retrieval must be defined in "
                                  "each child.")


class TasManager(DatasetManager):
    """This manager handles retrieval and processing of the TAS dataset."""
    name = 'tas'

    def _get_statements(self, db):
        from indra.sources.tas import process_csv
        proc = process_csv()
        return proc.statements
