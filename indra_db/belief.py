"""This file contains tools to calculate belief scores for the database.

Scores are calculated using INDRA's belief engine, with MockStatements and
MockEvidence derived from shallow metadata on the database, allowing the entire
corpus to be processed locally in RAM, in very little time.
"""
import tqdm
import pickle
import logging
import argparse
from datetime import datetime

from indra_db import util as dbu
from indra_db.util import S3Path
from indra_db.util.dump_sif import upload_pickle_to_s3, S3_SUBDIR

logger = logging.getLogger('db_belief')

from indra.belief import BeliefEngine, SimpleScorer


class LoadError(Exception):
    pass


class MockStatement(object):
    """A class to imitate real INDRA Statements for calculating belief."""
    def __init__(self, mk_hash, evidence=None, supports=None,
                 supported_by=None):
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

    def get_hash(self, **kwargs):
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
    for link in tqdm.tqdm(links):
        invalid_idx = [idx for idx in link if idx not in stmt_dict.keys()]
        if invalid_idx:
            logger.warning("Found at least one invalid index %s, from support "
                           "link pair %s." % (invalid_idx, link))
            continue
        supped_idx, supping_idx = link
        stmt_dict[supping_idx].supports.append(stmt_dict[supped_idx])
        stmt_dict[supped_idx].supported_by.append(stmt_dict[supping_idx])
    return


def load_mock_statements(db, hashes=None, sup_links=None):
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
    logger.info('Querying for evidence from reading')
    q_rdg = db.filter_query([db.Reading.reader,
                             db.RawUniqueLinks.pa_stmt_mk_hash,
                             db.RawUniqueLinks.raw_stmt_id],
                            *db.link(db.Reading, db.RawUniqueLinks))
    if hashes is not None:
        q_rdg = q_rdg.filter(db.RawUniqueLinks.pa_stmt_mk_hash.in_(hashes))
    res_rdg = q_rdg.all()

    for src_api, mk_hash, sid in tqdm.tqdm(res_rdg):
        add_evidence(src_api, mk_hash, sid)

    # Handle the evidence from knowledge bases.
    logger.info('Querying for evidence from knowledge bases')
    q_dbs = db.filter_query([db.DBInfo.source_api,
                             db.DBInfo.db_name,
                             db.RawUniqueLinks.pa_stmt_mk_hash,
                             db.RawUniqueLinks.raw_stmt_id],
                            *db.link(db.DBInfo, db.RawUniqueLinks))
    if hashes is not None:
        q_dbs = q_dbs.filter(db.RawUniqueLinks.pa_stmt_mk_hash.in_(hashes))
    res_dbs = q_dbs.all()

    for src_api, db_name, mk_hash, sid in tqdm.tqdm(res_dbs):
        add_evidence(src_api, mk_hash, sid, db_name)

    # Get the support links and populate
    if sup_links is None:
        logger.info('Querying for support links')
        sup_link_q = db.filter_query([db.PASupportLinks.supported_mk_hash,
                                      db.PASupportLinks.supporting_mk_hash])
        if hashes is not None:
            # I only use a constraint on the supported hash. If this is used in
            # the usual way where the hashes are part of a connected component,
            # then adding another constraint would have no effect (but to slow
            # down the query). Otherwise this is a bit arbitrary.
            sup_link_q = sup_link_q.filter(
                db.PASupportLinks.supported_mk_hash.in_(hashes)
            )
        sup_links = sup_link_q.all()

    logger.info('Populating support')
    populate_support(stmts_dict, sup_links)

    return list(stmts_dict.values())


def calculate_belief(stmts):
    scorer = SimpleScorer(subtype_probs={
        'biopax': {'pc11': 0.2, 'phosphosite': 0.01},
    })
    be = BeliefEngine(scorer=scorer)
    be.set_prior_probs(stmts)
    be.set_hierarchy_probs(stmts)
    return {str(s.get_hash()): s.belief for s in stmts}


def get_belief(db=None, partition=True):
    if db is None:
        db = dbu.get_db('primary')

    if partition:
        logger.info('Running on partitioned components')
        import networkx as nx
        hashes = {h for h, in db.select_all(db.PAStatements.mk_hash)}
        link_pair = [db.PASupportLinks.supporting_mk_hash,
                     db.PASupportLinks.supported_mk_hash]
        links = {tuple(link) for link in db.select_all(link_pair)}
        g = nx.Graph()
        g.add_nodes_from(hashes)
        g.add_edges_from(links)

        group = set()
        beliefs = {}
        for c in tqdm.tqdm(nx.connected_components(g)):
            group |= c

            if len(group) >= 10000:
                sg = g.subgraph(group)
                stmts = load_mock_statements(db, hashes=group,
                                             sup_links=list(sg.edges))
                beliefs.update(calculate_belief(stmts))
                group = set()
        return beliefs
    else:
        logger.info('Running without partitioning, loading mock statements')
        stmts = load_mock_statements(db)
        return calculate_belief(stmts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DB Belief Score Dumper')
    parser.add_argument('--fname',
                        nargs='?',
                        type=str,
                        default='belief_dict.pkl',
                        help='Filename of the belief dict output')
    parser.add_argument('-s3',
                        action='store_true',
                        default=False,
                        help='Upload belief dict to the bigmech s3 bucket '
                             'instead of saving it locally')
    args = parser.parse_args()
    belief_dict = get_belief()
    if args.s3:
        key = '/'.join([datetime.utcnow().strftime('%Y-%m-%d'), args.fname])
        s3_path = S3Path(S3_SUBDIR, key)
        upload_pickle_to_s3(obj=belief_dict, s3_path=s3_path)
    else:
        with open(args.fname, 'wb') as f:
            pickle.dump(belief_dict, f)
