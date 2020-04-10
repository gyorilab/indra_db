"""This file contains tools to calculate belief scores for the database.

Scores are calculated using INDRA's belief engine, with MockStatements and
MockEvidence derived from shallow metadata on the database, allowing the entire
corpus to be processed locally in RAM, in very little time.
"""

import pickle
import logging

from indra_db import get_primary_db

logger = logging.getLogger('db_belief')

from indra.belief import BeliefEngine, SimpleScorer


class LoadError(Exception):
    pass


class MockStatement(object):
    """A class to imitate real INDRA Statements for calculating belief."""
    def __init__(self, mk_hash, evidence=None, supports=None, supported_by=None):
        if isinstance(evidence, list):
            self.evidence = evidence
        elif evidence is None:
            self.evidence = []
        else:
            self.evidence = [evidence]
        self.__mk_hash = mk_hash
        if supports:
            self.supports = supports
        else:
            self.supports = []
        if supported_by:
            self.supported_by = supported_by
        else:
            self.supported_by = []
        self.belief = None

    def matches_key(self):
        return self.__mk_hash


class MockEvidence(object):
    """A class to imitate real INDRA Evidence for calculating belief."""
    def __init__(self, source_api, **annotations):
        self.source_api = source_api

        # The belief engine uses `Evidence.epistemics.get('negated')`
        self.epistemics = {}

        # Some annotations are used in indra.belief.tag_evidence_subtype.
        # TODO: optionally implement necessary annotations.
        self.annotations = annotations.copy()


def populate_support(stmts, links):
    """Populate the supports supported_by lists of statements given links.

    Parameters
    ----------
    stmts : list[MockStatement/Statement]
        A list of objects with supports and supported_by attributes which are
        lists or equivalent.
    links : list[tuple]
        A list of pairs of hashes or matches_keys, where the first supports the
        second.
    """
    if isinstance(stmts, dict):
        stmt_dict = stmts
    else:
        stmt_dict = {s.matches_key(): s for s in stmts}
    for link in links:
        invalid_idx = [idx for idx in link if idx not in stmt_dict.keys()]
        if invalid_idx:
            logger.warning("Found at least one invalid index %s, from support "
                           "link pair %s." % (invalid_idx, link))
            continue
        supped_idx, supping_idx = link
        stmt_dict[supping_idx].supports.append(stmt_dict[supped_idx])
        stmt_dict[supped_idx].supported_by.append(stmt_dict[supping_idx])
    return


def load_mock_statements(db):
    """Generate a list of mock statements from the pa statement table."""
    # Initialize a dictionary of evidence keyed by hash.
    stmts_dict = {}

    def add_evidence(src_api, mk_hash, sid, db_name=None):
        # Add a new mock statement if applicable.
        if mk_hash not in stmts_dict.keys():
            stmts_dict[mk_hash] = MockStatement(mk_hash)

        # Add the new evidence.
        mev = MockEvidence(src_api.lower(), raw_sid=sid, source_sub_id=db_name)
        stmts_dict[mk_hash].evidence.append(mev)

    # Handle the evidence from reading.
    res_rdg = db.select_all([db.Reading.reader,
                             db.RawUniqueLinks.pa_stmt_mk_hash,
                             db.RawUniqueLinks.raw_stmt_id],
                            *db.link(db.Reading, db.RawUniqueLinks))
    for src_api, mk_hash, sid in res_rdg:
        add_evidence(src_api, mk_hash, sid)

    # Handle the evidence from knowledge bases.
    res_dbs = db.select_all([db.DBInfo.source_api,
                             db.DBInfo.db_name,
                             db.RawUniqueLinks.pa_stmt_mk_hash,
                             db.RawUniqueLinks.raw_stmt_id],
                            *db.link(db.DBInfo, db.RawUniqueLinks))
    for src_api, db_name, mk_hash, sid in res_dbs:
        add_evidence(src_api, mk_hash, sid, db_name)

    # Get the support links and populate
    sup_links = db.select_all([db.PASupportLinks.supported_mk_hash,
                               db.PASupportLinks.supporting_mk_hash])
    populate_support(stmts_dict, sup_links)

    return list(stmts_dict.values())


def calculate_belief(stmts):
    scorer = SimpleScorer(subtype_probs={
        'biopax': {'pc11': 0.2, 'phosphosite': 0.01},
    })
    be = BeliefEngine(scorer=scorer)
    be.set_prior_probs(stmts)
    be.set_hierarchy_probs(stmts)
    return {s.matches_key(): s.belief for s in stmts}


def run(db=None):
    if db is None:
        db = get_primary_db()
    stmts = load_mock_statements(db)
    return calculate_belief(stmts)


if __name__ == '__main__':
    belief_dict = run()
    with open('belief_dict.pkl', 'wb') as f:
        pickle.dump(belief_dict, f)
