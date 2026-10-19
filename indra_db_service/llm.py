import logging
import sqlite3
from os import environ
from pathlib import Path


logger = logging.getLogger('llm')

configured_path = environ.get('INDRA_DB_API_LLM_CURATION_DB')
if configured_path:
    LLM_CURATION_DB_PATH = Path(configured_path).expanduser()
else:
    LLM_CURATION_DB_PATH = Path(
        '/data/indra_db/evidence_llm_20261005.sqlite'
    )
    if not LLM_CURATION_DB_PATH.is_file():
        LLM_CURATION_DB_PATH = (
            Path.home() / 'Downloads' / 'evidence_llm_20261005.sqlite'
        )


def _get_llm_statement_scores(connection, statement_hashes):
    """Return stored correctness counts keyed by statement hash."""
    has_statement_scores = connection.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type = 'table' AND name = 'llm_statement_correctness'
        """
    ).fetchone()
    if not has_statement_scores:
        return {}

    scores = {}
    statement_hashes = list({int(h) for h in statement_hashes})
    for start in range(0, len(statement_hashes), 900):
        batch = statement_hashes[start:start + 900]
        placeholders = ','.join('?' for _ in batch)
        rows = connection.execute(
            f"""
            SELECT statement_hash, correct_count, total_count,
                   correctness_percent
            FROM llm_statement_correctness
            WHERE statement_hash IN ({placeholders})
            """,
            batch,
        )
        scores.update({row[0]: row[1:] for row in rows})
    return scores


def add_llm_verifications(results, database_path=LLM_CURATION_DB_PATH):
    """Attach stored LLM curations to statement evidence JSON."""
    if not database_path.is_file():
        return

    try:
        database_uri = f'file:{database_path}?mode=ro'
        with sqlite3.connect(database_uri, uri=True) as connection:
            scores = _get_llm_statement_scores(connection, results)
            for statement_hash, statement in results.items():
                score = scores.get(int(statement_hash))
                if score:
                    statement['llm_correctness'] = {
                        'correct_count': score[0],
                        'total_count': score[1],
                        'percent': score[2],
                    }
                for evidence in statement['evidence']:
                    row = connection.execute(
                        """
                        SELECT judgment, error_category, explanation
                        FROM llm_evidence_curation
                        WHERE statement_hash = ? AND source_hash = ?
                          AND evidence_text != ''
                        """,
                        (int(statement_hash), int(evidence['source_hash'])),
                    ).fetchone()
                    if row:
                        evidence['llm_verification'] = {
                            'verdict': row[0],
                            'error_category': row[1],
                            'explanation': row[2],
                        }
    except sqlite3.Error as error:
        logger.warning("Could not read LLM curation database: %s", error)


def get_llm_verifications(statement_hashes, include_explanations=False,
                          database_path=LLM_CURATION_DB_PATH):
    """Return LLM evidence verifications grouped by statement hash."""
    if not database_path.is_file():
        return None

    statement_hashes = list(dict.fromkeys(int(h) for h in statement_hashes))
    results = {
        str(statement_hash): {
            'statement': None,
            'correctness': None,
            'verifications': [],
        }
        for statement_hash in statement_hashes
    }

    try:
        database_uri = f'file:{database_path}?mode=ro'
        with sqlite3.connect(database_uri, uri=True) as connection:
            scores = _get_llm_statement_scores(connection, statement_hashes)
            for statement_hash, score in scores.items():
                results[str(statement_hash)]['correctness'] = {
                    'correct_count': score[0],
                    'total_count': score[1],
                    'percent': score[2],
                }

            fields = ('statement_hash, source_hash, statement, evidence_text, '
                      'judgment, error_category'
                      + (', explanation' if include_explanations else ''))
            for start in range(0, len(statement_hashes), 900):
                batch = statement_hashes[start:start + 900]
                placeholders = ','.join('?' for _ in batch)
                rows = connection.execute(
                    f"""
                    SELECT {fields}
                    FROM llm_evidence_curation
                    WHERE statement_hash IN ({placeholders})
                      AND evidence_text != ''
                    ORDER BY statement_hash, source_hash
                    """,
                    batch,
                )
                for row in rows:
                    statement_result = results[str(row[0])]
                    if statement_result['statement'] is None:
                        statement_result['statement'] = row[2]
                    verification = {
                        'source_hash': str(row[1]),
                        'evidence_text': row[3],
                        'judgment': row[4],
                        'error_category': row[5],
                    }
                    if include_explanations:
                        verification['explanation'] = row[6]
                    statement_result['verifications'].append(verification)
    except sqlite3.Error as error:
        logger.warning("Could not read LLM curation database: %s", error)
        return None

    missing = [statement_hash for statement_hash, result in results.items()
               if result['correctness'] is None and not result['verifications']]
    return {'statements': results, 'missing': missing}


def add_llm_relation_correctness(results,
                                 database_path=LLM_CURATION_DB_PATH):
    """Attach the highest child correctness level to grouped relations."""
    if not database_path.is_file():
        return

    relation_hashes = {}
    all_hashes = set()
    for key, relation in results.items():
        hashes = relation.get('hashes')
        if hashes is None and relation.get('hash') is not None:
            hashes = [relation['hash']]
        hashes = tuple({int(statement_hash) for statement_hash in hashes or []})
        relation_hashes[key] = hashes
        all_hashes.update(hashes)

    if not all_hashes:
        return

    try:
        database_uri = f'file:{database_path}?mode=ro'
        with sqlite3.connect(database_uri, uri=True) as connection:
            scores = _get_llm_statement_scores(connection, all_hashes)

        for key, hashes in relation_hashes.items():
            child_scores = [scores[h] for h in hashes if h in scores]
            if not child_scores:
                continue
            if any(score[2] > 66.66 and score[1] >= 3
                   for score in child_scores):
                level = 'strong'
            elif any(score[2] > 33.33 for score in child_scores):
                level = 'medium'
            else:
                level = 'weak'
            results[key]['llm_correctness'] = {'level': level}
    except sqlite3.Error as error:
        logger.warning("Could not read LLM statement scores: %s", error)
