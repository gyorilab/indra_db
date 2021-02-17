__all__ = ['DbBuilder']

import json

from indra.statements import Evidence
from indra_db.databases import reader_versions
from indra_db.reading.read_db import DatabaseStatementData
from indra_db.util import insert_raw_agents, insert_pa_agents


class DbBuilder:
    def __init__(self, db):
        self.db = db
        self.text_refs = None
        self.text_content = None
        self.readings = None
        self.databases = None
        self.raw_statements = None
        self.pa_statements = None

    def add_text_refs(self, ref_defs):
        """Add text refs to the database."""
        assert self.text_refs is None
        self.text_refs = []
        for ref_tuple in ref_defs:
            self.text_refs.append(self.db.TextRef.new(*ref_tuple))
        self.db.session.add_all(self.text_refs)
        self.db.session.commit()

    def add_text_content(self, tr_content):
        """Add text content to the database."""
        assert self.text_refs is not None
        assert self.text_content is None
        self.text_content = []
        for i, src_list in enumerate(tr_content):
            basic_kwargs = {'text_ref_id': self.text_refs[i].id}
            for src in src_list:
                if src == 'pubmed-abs':
                    kwargs = {'source': 'pubmed', 'format': 'txt',
                              'text_type': 'abstract'}
                elif src == 'pubmed-ttl':
                    kwargs = {'source': 'pubmed', 'format': 'txt',
                              'text_type': 'title'}
                else:
                    kwargs = {'source': src, 'format': 'xml',
                              'text_type': 'fulltext'}
                kwargs.update(basic_kwargs)
                self.text_content.append(self.db.TextContent(**kwargs))
        self.db.session.add_all(self.text_content)
        self.db.session.commit()

    def add_readings(self, rdr_specs):
        """Add readings to the database."""
        assert self.text_content is not None
        assert self.readings is None
        self.readings = []
        for i, rdr_list in enumerate(rdr_specs):
            basic_kwargs = {'text_content_id': self.text_content[i].id,
                            'batch_id': 0}
            for rdr in rdr_list:
                kwargs = {'reader': rdr,
                          'reader_version': reader_versions[rdr.lower()][-1]}
                if rdr == 'SPARSER':
                    kwargs['format'] = 'json'
                else:
                    kwargs['format'] = 'xml'
                kwargs.update(basic_kwargs)
                self.readings.append(self.db.Reading(**kwargs))
        self.db.session.add_all(self.readings)
        self.db.session.commit()

    def add_raw_reading_statements(self, stmt_lists):
        """Add raw statements."""
        assert self.readings is not None
        if self.raw_statements is None:
            self.raw_statements = []
        new_raw_statements = []
        for ridx, stmt_list in enumerate(stmt_lists):
            rid = self.readings[ridx].id
            reader = self.readings[ridx].reader
            if reader == 'SPARSER':
                pmid = None
            else:
                pmid = self.readings[ridx].text_content.text_ref.pmid

            def ev(stmt, detail=None):
                reading = self.readings[ridx]
                text = f"{reading.text_content.source} from trid " \
                       f"{reading.text_content.text_ref_id} indicates " \
                       f"{type(stmt).__name__}: {stmt.agent_list()}."
                if detail is not None:
                    text = f"{detail} {text}"
                return Evidence(self.readings[ridx].reader.lower(), pmid=pmid,
                                text=text)

            for stmt in stmt_list:
                if isinstance(stmt, tuple):
                    stmt, detail = stmt
                else:
                    detail = None
                stmt.evidence.append(ev(stmt, detail))
                raw_json = stmt.to_json()
                db_rs = self.db.RawStatements(
                    reading_id=rid,
                    json=json.dumps(raw_json).encode('utf-8'),
                    type=raw_json['type'],
                    uuid=stmt.uuid,
                    batch_id=0,
                    text_hash=DatabaseStatementData(stmt)._get_text_hash(),
                    source_hash=stmt.evidence[0].get_source_hash(),
                    mk_hash=stmt.get_hash(),
                    indra_version="test"
                )
                self.raw_statements.append(db_rs)
                new_raw_statements.append(db_rs)
        self.db.session.add_all(new_raw_statements)
        self.db.session.commit()

        insert_raw_agents(self.db, 0,
                          [s[0] if isinstance(s, tuple) else s
                           for slist in stmt_lists for s in slist])

    def add_databases(self, db_names):
        """Add database refs into the database."""
        assert self.databases is None
        self.databases = []
        for db_name in db_names:
            self.databases.append(
                self.db.DBInfo(db_name=db_name, db_full_name=db_name,
                               source_api=db_name)
            )
        self.db.session.add_all(self.databases)
        self.db.session.commit()

    def add_raw_database_statements(self, stmt_lists):
        """Add raw statementes that came from knowledge bases/databases."""
        assert self.databases is not None
        if self.raw_statements is None:
            self.raw_statements = []
        new_raw_statements = []
        for dbidx, stmt_list in enumerate(stmt_lists):
            db_info = self.databases[dbidx]

            for stmt in stmt_list:
                ev = Evidence(db_info.source_api)
                stmt.evidence.append(ev)
                src_hash = ev.get_source_hash()
                raw_json = stmt.to_json()
                db_rs = self.db.RawStatements(
                    db_info_id=db_info.id,
                    json=json.dumps(raw_json).encode('utf-8'),
                    type=raw_json['type'],
                    uuid=stmt.uuid,
                    batch_id=1,
                    source_hash=src_hash,
                    mk_hash=stmt.get_hash(),
                    indra_version="test"
                )
                self.raw_statements.append(db_rs)
                new_raw_statements.append(db_rs)

        self.db.session.add_all(new_raw_statements)
        self.db.session.commit()

        insert_raw_agents(self.db, 1,
                          [s for slist in stmt_lists for s in slist])

    def add_pa_statements(self, stmt_list):
        """Add preassembled statements to the test database."""
        pa_stmts = []
        pa_supp_links = []
        raw_unique_links = []
        for tpl in stmt_list:
            if len(tpl) == 3:
                pa_stmt, raw_stmt_idx_list, supports = tpl
                for supp_idx in supports:
                    pa_supp_links.append(self.db.PASupportLinks(
                        supporting_mk_hash=pa_stmt.get_hash(),
                        supported_mk_hash=stmt_list[supp_idx][0].get_hash()
                    ))
            else:
                pa_stmt, raw_stmt_idx_list = tpl

            pa_json = pa_stmt.to_json()
            h = pa_stmt.get_hash()
            for raw_stmt_idx in raw_stmt_idx_list:
                raw_stmt = self.raw_statements[raw_stmt_idx]
                raw_unique_links.append(
                    self.db.RawUniqueLinks(raw_stmt_id=raw_stmt.id,
                                           pa_stmt_mk_hash=h)
                )
            pa_stmts.append(
                self.db.PAStatements(
                    mk_hash=h,
                    json=json.dumps(pa_json).encode('utf-8'),
                    type=pa_json['type'],
                    uuid=pa_stmt.uuid,
                    matches_key=pa_stmt.matches_key(),
                    indra_version='test'
                )
            )
        self.db.session.add_all(pa_stmts)
        self.db.session.commit()
        self.db.session.add_all(raw_unique_links)
        self.db.session.commit()
        self.db.session.add_all(pa_supp_links)
        self.db.session.commit()

        insert_pa_agents(self.db, [tpl[0] for tpl in stmt_list])


