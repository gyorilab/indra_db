import boto3
import moto

from indra.statements import Phosphorylation, Agent, Activation, Inhibition, \
    Complex, Evidence, Conversion
from indra_db.config import get_s3_dump
from indra_db.managers import dump_manager as dm
from indra_db.tests.util import get_temp_db, get_temp_ro, insert_test_stmts
from indra_db.util import S3Path, insert_db_stmts, insert_raw_agents


def _build_s3_test_dump(structure):
    """Build an s3 dump for testing.

    The input is a structure of the following form:
    ```
    structure = {
       '2020-01-01': [
            'start',
            'readonly',
            'sif',
       ],
       '2020-02-01': [
            'start',
            'readonly',
            'sif',
            'belief',
            'end'
       ]
    }
    ```
    where the names given are the canonical names of dumpers (see class
    definitions or `dumpers` global for details).
    """
    s3 = boto3.client('s3')
    dump_head = get_s3_dump()
    s3.create_bucket(Bucket=dump_head.bucket)
    for date_stamp, contents in structure.items():
        for dump_name in contents:
            dumper_class = dm.dumpers[dump_name]
            dumper_class(date_stamp=date_stamp).shallow_mock_dump()


@moto.mock_s3
def test_list_dumps():
    """Test the dump listing feature."""
    _build_s3_test_dump({
        '2020-01-01': ['start'],         # a dump that is unfinished.
        '2020-02-01': ['start', 'end'],  # a dump that is finished.
        '2020-03-01': ['sif']            # something strange but possible.
    })

    dump_head = get_s3_dump()

    def check_list(dumps, expected_timestamps):
        assert all(isinstance(s3p, S3Path) for s3p in dumps)
        assert all(s3p.key.startswith(dump_head.key) for s3p in dumps)
        time_stamps = [s3p.key.split('/')[-2] for s3p in dumps]
        assert expected_timestamps == time_stamps,\
            f"Expected: {expected_timestamps}, Got: {time_stamps}"

    all_dumps = dm.list_dumps()
    check_list(all_dumps, ['2020-01-01', '2020-02-01', '2020-03-01'])

    started_dumps = dm.list_dumps(started=True)
    check_list(started_dumps, ['2020-01-01', '2020-02-01'])

    done_dumps = dm.list_dumps(started=True, ended=True)
    check_list(done_dumps, ['2020-02-01'])

    unfinished_dumps = dm.list_dumps(started=True, ended=False)
    check_list(unfinished_dumps, ['2020-01-01'])


@moto.mock_s3
def test_get_latest():
    """Test the function used to get the latest version of a dump file."""
    _build_s3_test_dump({
        '2019-12-01': ['start', 'readonly', 'sif', 'end'],
        '2020-01-01': ['start', 'readonly'],
        '2020-02-01': ['start', 'sif', 'end']
    })

    ro_dump = dm.get_latest_dump_s3_path('readonly')
    assert '2020-01-01' in ro_dump.key, ro_dump.key

    sif_dump = dm.get_latest_dump_s3_path('sif')
    assert '2020-02-01' in sif_dump.key, sif_dump.key


@moto.mock_s3
def test_list_dumps_empty():
    """Test list_dumps when there are no dumps."""
    _build_s3_test_dump({})
    assert dm.list_dumps() == []



@moto.mock_s3
def test_dump_build():
    """Test the dump pipeline.

    Method
    ------
    CREATE CONTEXT:
    - Create a local principal database with a small amount of content.
      Aim for representation of stmt motifs and sources.
    - Create a local readonly database.
    - Create a fake bucket (moto)

    RUN THE DUMP

    CHECK THE RESULTS
    """
    db = get_temp_db(clear=True)

    db.copy('text_ref', [        # trid
        ('1', 1, 'PMC1', 1),     # 1
        ('2', 2, 'PMC2', 2),     # 2
        ('3', 3, None, None),    # 3
        (None, None, 'PMC4', 4)  # 4
    ], ('pmid', 'pmid_num', 'pmcid', 'pmcid_num'))

    db.copy('mesh_ref_annotations', [
        (1, 11, False),
        (1, 13, False),
        (1, 12, True),
        (2, 12, True),
        (3, 13, False),
        (3, 33, True)
    ], ('pmid_num', 'mesh_num', 'is_concept'))

    db.copy('text_content', [              # tcid
        (1, 'pubmed', 'txt', 'abstract'),  # 1
        (1, 'pmc', 'xml', 'fulltext'),     # 2
        (2, 'pubmed', 'txt', 'title'),     # 3
        (3, 'pubmed', 'txt', 'abstract'),  # 4
        (3, 'pmc', 'xml', 'fulltext'),     # 5
        (4, 'pmc', 'xml', 'fulltext')      # 6
    ], ('text_ref_id', 'source', 'format', 'text_type'))

    db.copy('reading', [(tcid, rdr, 1, 'bogus', 'emtpy') for tcid, rdr in [
        # 1             2             3
        (1, 'reach'), (1, 'eidos'), (1, 'isi'),

        # 4
        (2, 'reach'),

        # 5             6            7
        (3, 'reach'), (3, 'eidos'), (3, 'trips'),

        # 8
        (4, 'reach'),

        # 9
        (5, 'reach'),

        # 10
        (6, 'reach')
    ]], ('text_content_id', 'reader', 'batch_id', 'reader_version', 'format'))

    db.copy('db_info', [
        ('signor', 'signor', 'Signor'),       # 1
        ('pc', 'biopax', 'Pathway Commons'),  # 2
        ('medscan', 'medscan', 'MedScan')     # 3
    ], ('db_name', 'source_api', 'db_full_name'))

    raw_stmts = {
        'reading': {
            2: [
                Inhibition(
                    Agent('Fever', db_refs={'TEXT': 'fever', 'MESH': 'D005334'}),
                    Agent('Cough', db_refs={'TEXT': 'cough', 'MESH': 'D003371'}),
                    evidence=Evidence(text="We found fever inhibits cough.")
                )
            ],
            4: [
                Phosphorylation(
                    Agent('MEK', db_refs={'FPLX': 'MEK', 'TEXT': 'mek'}),
                    Agent('ERK', db_refs={'FPLX': 'MEK', 'TEXT': 'erk'}),
                    evidence=Evidence(text="mek phosphorylates erk, so say I.")
                ),
                Activation(
                    Agent('MAP2K1', db_refs={'HGNC': '6840', 'TEXT': 'MEK1'}),
                    Agent('MAPK1', db_refs={'HGNC': '6871', 'TEXT': 'ERK1'}),
                    evidence=Evidence(text="MEK1 activates ERK1, or os I'm told.")
                ),
                Activation(
                    Agent('ERK', db_refs={'FPLX': 'ERK', 'TEXT': 'ERK'}),
                    Agent('JNK', db_refs={'FPLX': 'JNK', 'TEXT': 'JNK'}),
                    evidence=Evidence(text="ERK activates JNK, maybe.")
                ),
                Complex([
                    Agent('MEK', db_refs={'FPLX': 'MEK', 'TEXT': 'MAP2K'}),
                    Agent('ERK', db_refs={'FPLX': 'ERK', 'TEXT': 'MAPK'}),
                    Agent('RAF', db_refs={'FPLX': 'RAF', 'TEXT': 'RAF'})
                ], evidence=Evidence(text="MAP2K, MAPK, and RAF form a complex."))
            ],
            7: [
                Activation(
                    Agent('ERK', db_refs={'FPLX': 'ERK', 'TEXT': 'ERK'}),
                    Agent('JNK', db_refs={'FPLX': 'JNK', 'TEXT': 'JNK'}),
                    evidence=Evidence(text='ERK activates JNK, maybe.')
                )
            ],
            8: [
                Complex([
                    Agent('MEK', db_refs={'FPLX': 'MEK', 'TEXT': 'mek'}),
                    Agent('ERK', db_refs={'FPLX': 'ERK', 'TEXT': 'erk'})
                ], evidence=Evidence(text="...in the mek-erk complex."))
            ],
        },
        'databases': {
            2: [
                Conversion(
                    Agent('FRK', db_refs={'HGNC': '3955'}),
                    [Agent('ATP', db_refs={'MESH': 'D000255'})],
                    [Agent('hydron', db_refs={'CHEBI': 'CHEBI:15378'})]
                )
            ],
            3: [
                Phosphorylation(
                    Agent('MEK', db_refs={'FPLX': 'MEK', 'TEXT': 'MEK'}),
                    Agent('ERK', db_refs={'FPLX': 'ERK', 'TEXT': 'ERK'}),
                    evidence=Evidence(text="...MEK phosphorylates ERK medscan.")
                )
            ]
        }
    }
    insert_test_stmts(db, raw_stmts)


    ro = get_temp_ro(clear=True)
