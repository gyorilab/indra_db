import unittest
import numpy as np
from os import path, remove
import pandas as pd

import indra_db.tests.util as tu
from indra_db.util.dump_sif import load_db_content, get_source_counts, \
    make_dataframe, NS_LIST, normalize_sif_names


class SifDumperTester(unittest.TestCase):
    def get_db(self, count=1000):
        # Get db
        return tu.get_filled_ro(count)

    def setUp(self):
        self.db = self.get_db()
        self.db_content = load_db_content(True, NS_LIST, None, self.db)
        self.df = make_dataframe(True, self.db_content, None)

    # Tests
    def test_get_content(self):
        """Checks content loading and its structure"""

        # Get first item
        r = list(self.db_content)[0]
        assert isinstance(r, tuple)
        assert len(r) == 6
        assert isinstance(r[0], int)  # mk_hash
        assert isinstance(r[1], str)  # db_name
        assert r[1] in NS_LIST
        assert isinstance(r[2], str)  # db_id
        assert isinstance(r[3], int)  # ag_num
        assert r[3] > -1
        assert isinstance(r[4], int)  # ev_count
        assert r[4] > 0
        assert isinstance(r[5], str)  # type

    def test_dataframe(self):
        """Checks a dataframe produced by make_dataframe"""

        # Check column names
        assert {'agA_id', 'agA_name', 'agA_ns', 'agB_id', 'agB_name', 'agB_ns',
                'evidence_count', 'stmt_hash', 'stmt_type'} == set(
            self.df.columns)

        # Check for None's
        assert sum(self.df['agA_name'] == None) == 0
        assert sum(self.df['agB_name'] == None) == 0

        # Check df types
        assert isinstance(self.df.head(1)['agA_ns'][0], str)
        assert isinstance(self.df.head(1)['agB_ns'][0], str)
        assert isinstance(self.df.head(1)['agA_id'][0], str)
        assert isinstance(self.df.head(1)['agB_id'][0], str)
        assert isinstance(self.df.head(1)['agA_name'][0], str)
        assert isinstance(self.df.head(1)['agB_name'][0], str)
        assert isinstance(self.df.head(1)['stmt_type'][0], str)
        assert isinstance(self.df.head(1)['evidence_count'][0], np.int64)
        assert isinstance(self.df.head(1)['stmt_hash'][0], np.int64)

        # Check that we don't have significant keyerrors from creating the df
        key_error_file = path.join(path.dirname(__file__), 'key_errors.csv')
        if path.exists(key_error_file):
            key_errors = pd.read_csv(key_error_file, sep=',',
                                     names=['stmt_hash', 'ag_num'], header=None)
            remove(key_error_file)
            missing_hashes = set(key_errors['stmt_hash'].values)
            df_hashes = set(self.df['stmt_hash'].values)

            assert len(missing_hashes.intersection(df_hashes)) / \
                len(df_hashes) < 0.5

    def test_stratified_evidence(self):
        """Check the stratified evidence dumper"""

        ev_dict = get_source_counts(ro=self.db)

        # Check if nested dict
        for k in ev_dict:
            assert isinstance(ev_dict[k], dict)
            break

        # Check that some keys exist in the df
        df_hashes = set(self.df['stmt_hash'].values)
        sd_hashes = set(ev_dict.keys())
        assert len(sd_hashes.intersection(df_hashes)) / len(sd_hashes) > 0.25


def test_normalize_names():
    sif_dict = {
        'agA_ns': ['HGNC', 'HGNC'],
        'agA_id': ['26128', '26128'],
        'agA_name': ['SPRING1', 'C12orf49'],
        'agB_ns': ['HGNC', 'HGNC'],
        'agB_id': ['11892', '3236'],
        'agB_name': ['TNF', 'EGFR'],
        'stmt_type': ['Activation', 'Phosphorylation'],
        'evidence_count': [10, 12],
        'stmt_hash': [1234567890, -9876543210],
        'residue': [None, None],
        'position': [None, None],
        'source_counts': [{'sparser': 6, 'reach': 4}, {'pc': 6, 'sparser': 6}],
        'belief': [0.998, 0.9999]
     }

    sif_df = pd.DataFrame(sif_dict)
    normalize_sif_names(sif_df)
    # Both names should now be SPRING1
    assert set(sif_df.agA_name.values) == {'SPRING1'}
