from __future__ import absolute_import, print_function, unicode_literals
from builtins import dict, str

import pickle
import random
from os import path, chdir
from subprocess import check_call

import boto3
from nose.plugins.attrib import attr

from indra.util import zip_string
from indra.tools.reading.readers import SparserReader, EmptyReader
from indra.tools.reading.readers import get_reader_classes

from indra_db import util as dbu
from indra_db.reading import read_db as rdb
from indra_db.tests.util import get_db_with_pubmed_content
from indra_db.reading.submit_reading_pipeline import DbReadingSubmitter


# ==============================================================================
# Tests for NEW reading pipeline which uses the database.
# ==============================================================================


def get_id_dict(tr_list):
    idtype_list = ['pmid', 'pmcid', 'doi']
    id_dict = {id_type: [] for id_type in idtype_list}
    for tr in tr_list:
        random.shuffle(idtype_list)
        for idtype in idtype_list:
            if getattr(tr, idtype) is not None:
                break
        else:
            raise Exception("No id types found for text ref.")
        id_dict[idtype].append(getattr(tr, idtype))
    return id_dict


def get_readers(*names, **kwargs):
    kwargs['ResultClass'] = rdb.DatabaseReadingData
    return [reader_class(**kwargs) for reader_class in get_reader_classes()
            if (not names or reader_class.name in names)]


@attr('nonpublic')
def test_get_content():
    "Test that content queries are correctly formed."
    db = get_db_with_pubmed_content()
    tcids = {tcid for tcid, in db.select_all(db.TextContent.id)}
    readers = get_readers()
    for reader in readers:
        worker = rdb.DatabaseReader(tcids, reader, db=db)

        N_exp = db.filter_query(db.TextContent).count()
        N_1 = sum([1 for _ in worker.iter_over_content()])
        assert N_1 == N_exp,\
            "Expected %d results in our query, got %d." % (N_exp, N_1)

        # Test response to empyt dict.
        worker = rdb.DatabaseReader([], reader, db=db)
        assert not any([c for c in worker.iter_over_content()]), \
            "Expected no results when passing no ids."


@attr('nonpublic')
def test_get_reader_children():
    "Test method for getting reader objects."
    readers = get_readers()
    assert len(readers) == 3, \
        "Expected only 3 readers, but got %s." % str(readers)


@attr('slow', 'nonpublic')
def test_reading_content_insert():
    "Test the content primary through-put of make_db_readings."
    db = get_db_with_pubmed_content()

    print("Test reading")
    tcids = {tcid for tcid, in db.select_all(db.TextContent.id)}
    readers = get_readers()
    workers = [rdb.DatabaseReader(tcids, reader, verbose=True, db=db)
               for reader in readers]
    reading_output = []
    for worker in workers:
        worker.get_readings()

        expected_output_len = len(tcids)
        N_new = len(worker.new_readings)
        reading_output.extend(worker.new_readings)
        if isinstance(worker.reader, EmptyReader):
            assert N_new == 0, "An empty reader read something..."
        else:
            assert N_new == expected_output_len, \
                ("Not all text content successfully read by %s."
                 "Expected %d outputs, but got %d."
                 % (worker.reader.name, expected_output_len, N_new))

    print("Test reading insert")
    for worker in workers:
        worker.dump_readings_to_db()
    r_list = db.select_all(db.Reading)

    def is_complete_match(r_list, reading_output):
        return all([any([rd.matches(r) for r in r_list])
                    for rd in reading_output])

    assert is_complete_match(r_list, reading_output), \
        "Not all reading output posted."

    print("Test making statements")
    num_stmts = 0
    for worker in workers:
        worker.get_statements()
        if not isinstance(worker.reader, EmptyReader):
            assert worker.statement_outputs, "Did not get statement outputs."
        num_stmts += len(worker.statement_outputs)

        worker.dump_statements_to_db()

    num_db_sids = db.count(db.RawStatements.id)
    assert num_db_sids == num_stmts, \
        "Only %d/%d statements added." % (num_db_sids, num_stmts)
    assert len(db.select_all(db.RawAgents)), "No agents added."


@attr('nonpublic')
def test_read_db():
    "Test the low level make_db_readings functionality with various settings."
    # Prep the inputs.
    db = get_db_with_pubmed_content()
    tcids = {tcid for tcid, in db.select_all(db.TextContent.id)}
    reader = get_readers('SPARSER')[0]

    # Run the reading with default batch size and reading_mode set to 'unread'
    worker1 = rdb.DatabaseReader(tcids, reader, db=db, reading_mode='unread')
    worker1.get_readings()
    N1 = len(worker1.new_readings)
    N1_exp = len(tcids)
    assert N1 == N1_exp, \
        'Expected %d readings, but got %d.' % (N1_exp, N1)
    worker1.dump_readings_to_db()
    N1_db = len(db.select_all(db.Reading))
    assert N1_db == N1, \
        'Expected %d readings to be copied to db, only %d found.' % (N1, N1_db)

    # Run the reading with default batch size, reading_mode set to 'all'. (this
    # should produce new readings.)
    reader.reset()
    worker2 = rdb.DatabaseReader(tcids, reader, db=db, reading_mode='all')
    worker2.get_readings()

    N2_old = len(worker2.extant_readings)
    N2_new = len(worker2.new_readings)
    print(N2_old, N2_new, N1, N1_db)
    assert N2_old == 0,\
        "Got %d old readings despite reading_mode set to 'all'." % N2_old
    assert N1 == N2_new, \
        "Got %d readings from run 1 but %d from run 2." % (N1, N2_new)

    # Run the reading with default batch size, with reading_mode set to
    # 'unread', again. (this should NOT produce new readings.)
    reader.reset()
    worker3 = rdb.DatabaseReader(tcids, reader, db=db, reading_mode='unread')
    worker3.get_readings()

    N_new = len(worker3.new_readings)
    N_old = len(worker3.extant_readings)

    assert N_new == 0,\
        "Got new readings when reading_mode was 'unread' and readings existed."
    assert N_old == N1, \
        ("Missed old readings when reading_mode was 'unread' and readings "
         "existed: expected %d, but got %d." % (N1, N_old))


@attr('slow', 'nonpublic')
def test_produce_readings():
    "Comprehensive test of the high level production of readings."
    # Prep the inputs.
    db = get_db_with_pubmed_content()
    tcids = {tcid for tcid, in db.select_all(db.TextContent.id)}

    # Test with just sparser for tollerable speeds.
    readers = get_readers('SPARSER')

    # Test the reading_mode='none' option (should yield nothing, because there
    # aren't any readings yet.)
    workers = rdb.run_reading(readers, tcids, verbose=True, db=db,
                              reading_mode='none', stmt_mode='none')
    assert all(len(worker.new_readings) == 0 for worker in workers)
    assert all(len(worker.extant_readings) == 0 for worker in workers)

    # Test just getting a pickle file (Nothing should be posted to the db.).
    pkl_file = 'test_db_res.pkl'
    workers = rdb.run_reading(readers, tcids, verbose=True, db=db,
                              upload_readings=False, reading_pickle=pkl_file)
    N_new = len(workers[0].new_readings)
    N_old = len(workers[0].extant_readings)
    N_exp = len(readers)*len(tcids)
    assert N_new == N_exp, "Expected %d readings, got %d." % (N_exp, N_new)
    assert N_old == 0, "Found old readings, when there should be none."
    assert path.exists(pkl_file), "Pickle file not created."
    with open(pkl_file, 'rb') as f:
        N_pkl = len(pickle.load(f))
    assert N_pkl == N_exp, \
        "Expected %d readings in pickle, got %d." % (N_exp, N_pkl)
    N_readings = db.filter_query(db.Reading).count()
    assert N_readings == 0, \
        "There shouldn't be any readings yet, but found %d." % N_readings

    # Test reading and insert to the database.
    rdb.run_reading(readers, tcids, verbose=True, db=db)
    N_db = db.filter_query(db.Reading).count()
    assert N_db == N_exp, "Expected %d readings, got %d." % (N_exp, N_db)

    # Test reading again, without read_mode='all', ('unread' by default)
    workers = rdb.run_reading(readers, tcids, verbose=True, db=db)
    N_old = len(workers[0].extant_readings)
    N_new = len(workers[0].new_readings)
    assert N_old == N_exp, \
        "Got %d old readings, expected %d." % (N_old, N_exp)
    assert N_new == 0, \
        "Got %d new readings, when none should have been read." % N_new
    assert all([rd.reading_id is not None
                for rd in workers[0].extant_readings])

    # Test with read_mode='none' again.
    workers = rdb.run_reading(readers, tcids, verbose=True, db=db,
                              reading_mode='none')
    N_old = len(workers[0].extant_readings)
    assert N_old == N_exp
    assert all([rd.reading_id is not None
                for rd in workers[0].extant_readings])

    # Test the read_mode='all'.
    workers = rdb.run_reading(readers, tcids, verbose=True, db=db,
                              reading_mode='all')
    old = workers[0].extant_readings
    new = workers[0].new_readings
    assert len(new) == N_exp
    assert len(old) == 0
    assert all([rd.reading_id is not None for rd in new])


@attr('nonpublic')
def test_stmt_mode_unread():
    "Test whether we can only create statements from unread content."
    # Prep the inputs.
    db = get_db_with_pubmed_content()
    tcids = {tcid for tcid, in db.select_all(db.TextContent.id)}

    # Test with just sparser for tolerable speeds.
    readers = get_readers('SPARSER')

    # First create some readings.
    some_tcids = random.sample(tcids, len(tcids)//2)
    workers0 = rdb.run_reading(readers, some_tcids, db=db, verbose=True)
    pre_stmt_hash_set = {sd.statement.get_hash(shallow=False)
                         for sd in workers0[0].statement_outputs}

    # Now only make statements for the content that was not read.
    workers = rdb.run_reading(readers, tcids, db=db, verbose=True,
                              stmt_mode='unread')
    stmt_hash_set = {sd.statement.get_hash(shallow=False)
                     for sd in workers[0].statement_outputs}
    assert stmt_hash_set.isdisjoint(pre_stmt_hash_set), \
        "There were overlapping statements."


@attr('nonpublic')
def test_sparser_parallel():
    "Test running sparser in parallel."
    db = get_db_with_pubmed_content()
    sparser_reader = SparserReader(n_proc=2)
    tc_list = db.select_all(db.TextContent)
    result = sparser_reader.read([rdb.process_content(tc) for tc in tc_list],
                                 verbose=True, log=True)
    N_exp = len(tc_list)
    N_res = len(result)
    assert N_exp == N_res, \
        "Expected to get %d results, but got %d." % (N_exp, N_res)


@attr('nonpublic')
def test_sparser_parallel_one_batch():
    "Test that sparser runs with multiple procs with batches of 1."
    db = get_db_with_pubmed_content()
    sparser_reader = SparserReader(n_proc=2)
    tc_list = db.select_all(db.TextContent)
    result = sparser_reader.read([rdb.process_content(tc) for tc in tc_list],
                                 verbose=True, n_per_proc=1)
    N_exp = len(tc_list)
    N_res = len(result)
    assert N_exp == N_res, \
        "Expected to get %d results, but got %d." % (N_exp, N_res)


@attr('slow', 'nonpublic')
def test_multi_batch_run():
    "Test that reading works properly with multiple batches run."
    db = get_db_with_pubmed_content()
    readers = get_readers()
    tcids = {tcid for tcid, in db.select_all(db.TextContent.id)}
    rdb.run_reading(readers, tcids, batch_size=len(tcids)//2, db=db,
                    stmt_mode='none')

    # This should catch any repeated readings.
    num_readings = db.filter_query(db.Reading).count()
    num_expected = len([r for r in readers if not isinstance(r, EmptyReader)])*len(tcids)
    assert num_readings == num_expected, \
        "Expected %d readings, only found %d." % (num_expected, num_readings)


@attr('slow', 'nonpublic')
def test_multiproc_statements():
    "Test the multiprocessing creation of statements."
    db = get_db_with_pubmed_content()
    readers = get_readers()
    tcids = {tcid for tcid, in db.select_all(db.TextContent.id)}
    workers = rdb.run_reading(readers, tcids, db=db)
    assert not any(worker.extant_readings for worker in workers)
    outputs = [rd for worker in workers for rd in worker.new_readings]
    stmts = rdb.make_statements(outputs, 2)
    assert len(stmts)


@attr('nonpublic')
def test_db_reading_help():
    chdir(path.expanduser('~'))
    check_call(['python', '-m', 'indra_db.reading.read_db_aws',
                '--help'])


@attr('nonpublic')
def test_normal_db_reading_call():
    s3 = boto3.client('s3')
    chdir(path.expanduser('~'))
    # Put some basic stuff in the test databsae
    N = 6
    db = dbu.get_test_db()
    db._clear(force=True)
    db.copy('text_ref', [(i, 'PMID80945%d' % i) for i in range(N)],
            cols=('id', 'pmid'))
    text_content = [
        (i, i, 'pubmed', 'text', 'abstract',
         zip_string('MEK phosphorylates ERK in test %d.' % i))
        for i in range(N)
        ]
    text_content += [
        (N, N-1, 'pmc_oa', 'text', 'fulltext',
         zip_string('MEK phosphorylates ERK. EGFR activates SHC.'))
        ]
    db.copy('text_content', text_content,
            cols=('id', 'text_ref_id', 'source', 'format', 'text_type',
                  'content'))

    # Put an id file on s3
    basename = 'local_db_test_run'
    s3_prefix = 'reading_results/%s/' % basename
    s3.put_object(Bucket='bigmech', Key=s3_prefix + 'id_list',
                  Body='\n'.join(['%d' % i for i in range(len(text_content))]))

    # Call the reading tool
    sub = DbReadingSubmitter(basename, ['sparser'])
    job_name, cmd = sub._make_command(0, len(text_content))
    cmd += ['--test']
    check_call(cmd)
    sub.produce_report()

    # Remove garbage on s3
    res = s3.list_objects(Bucket='bigmech', Prefix=s3_prefix)
    for entry in res['Contents']:
        print("Removing %s..." % entry['Key'])
        s3.delete_object(Bucket='bigmech', Key=entry['Key'])
    return
