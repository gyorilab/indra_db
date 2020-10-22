import boto3
import moto

from indra_db.config import get_s3_dump
from indra_db.managers import dump_manager as dm
from indra_db.util import S3Path


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
        '2020-02-01': ['start', 'end'],  # a dump the finished.
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
