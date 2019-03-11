import logging

from indra_db.util import insert_db_stmts

logger = logging.getLogger(__name__)


class DatasetManager(object):
    """This is a class to lay out the methods for updating a dataset."""
    name = NotImplemented

    def upload(self, db):
        """Upload the content for this dataset into the database."""
        dbid = self.check_reference(db)
        stmts = self._get_statements(db)
        insert_db_stmts(db, set(stmts), dbid)
        return

    def check_reference(self, db):
        """Ensure that this database has an entry in the database."""
        dbid = db.select_one(db.DBInfo.id, db.DBInfo.db_name == self.name)
        if dbid is None:
            dbid = db.insert(db.DBInfo, db_name=self.name)
        return dbid

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
