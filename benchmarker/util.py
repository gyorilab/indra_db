__all__ = ['benchmark', 'list_apis', 'list_stacks', 'save_results']

import os
import json

import boto3
import logging
from datetime import datetime
from inspect import getmembers, isfunction, isclass, ismethod
from importlib.util import spec_from_file_location, module_from_spec

from numpy import array


logger = logging.getLogger('benchmark_tools')

BUCKET = 'bigmech'
PREFIX = 'indra-db/benchmarks/'


def run_test(test_name, test_func, num_runs):
    test_results = dict.fromkeys(['passed', 'error_type', 'error_str',
                                  'duration', 'deviation', 'times'])
    test_results['passed'] = False
    test_results['error_type'] = [None]*num_runs
    test_results['error_str'] = [None]*num_runs
    print(test_name)
    print('-' * len(test_name))
    durations = []
    for i in range(num_runs):
        print("LOGS:")
        start = datetime.now()
        try:
            test_func()
            print('-' * len(test_name))
            print("PASSED!")
            test_results['passed'] += True
        except Exception as e:
            print('-' * len(test_name))
            print("FAILED!", type(e), e)
            logger.exception(e)
            test_results['passed'] += False
            test_results['error_type'][i] = str(type(e))
            test_results['error_str'][i] = str(e)
        finally:
            end = datetime.now()
            durations.append((end - start).total_seconds())
            print()
    dur_array = array(durations)
    test_results['times'] = durations
    test_results['duration'] = dur_array.mean()
    test_results['deviation'] = dur_array.std()
    test_results['passed'] = test_results['passed'] / num_runs
    return test_results


def benchmark(test_selection=None, base_name=None, num_runs=1):
    """Run a benchmark of the REST service using a given test corpus.

    Parameters
    ----------
    test_selection : Optional[str]
        Specify the location of the test or tests you wish to run, using the
        standard formalism: "path/to/test.py:specific_test", where any less
        specification will result in a search for things that start with "test_"
        recursively, as usual.
    base_name : Optional[str]
        Give this benchmark a base name.
    num_runs : Optional[int]
        Specify how many times the tests should be run.
    """
    # By default, just run in this directory
    if test_selection is None:
        test_selection = os.path.abspath('.')

    # Extract a function name, if it was included.
    if test_selection.count(':') == 0:
        func_name = None
    elif test_selection.count(':') == 1:
        test_selection, func_name = test_selection.split(':')
    else:
        raise ValueError(f"Invalid loc: {test_selection}")
    mod_name = os.path.basename(test_selection).replace('.py', '')
    if base_name:
        mod_name = base_name + '.' + mod_name

    # Check if the location exists, and whether it is a directory or file.
    # Handle the file case by recursively calling this function for each file.
    results = {}
    if not os.path.exists(test_selection):
        raise ValueError(f"No such file or directory: {test_selection}")
    elif os.path.isdir(test_selection):
        if func_name is not None:
            raise ValueError("To specify function, location must be a file.")
        for file in os.listdir(test_selection):
            new_path = os.path.join(test_selection, file)
            if ('test' in file and os.path.isfile(new_path)
                    and new_path.endswith('.py')):
                results.update(benchmark(new_path, base_name=mod_name,
                                         num_runs=num_runs))
        return results

    # Handle the case a file is specified.
    if not test_selection.endswith('.py'):
        raise ValueError(f"Location {test_selection} is not a python file.")
    print("=" * len(test_selection))
    print(test_selection)
    print('-' * len(test_selection))
    spec = spec_from_file_location(mod_name, test_selection)
    test_module = module_from_spec(spec)
    try:
        spec.loader.exec_module(test_module)
    except KeyboardInterrupt:
        raise
    except Exception as err:
        logger.error(f"Failed to load {test_selection}, skipping...")
        logger.exception(err)
        return results

    # Run test functions
    tests = [f for f, _ in getmembers(test_module, isfunction) if 'test' in f]
    for test_name in tests:
        test = getattr(test_module, test_name)
        results[f'{mod_name}.{test_name}'] = run_test(test_name, test, num_runs)

    # Run test classes
    test_classes = [c for c, _ in getmembers(test_module, isclass)
                    if c.lower().startswith('test')]
    for class_name in test_classes:
        cls = getattr(test_module, class_name)
        obj = cls()
        test_methods = [m for m, _ in getmembers(obj, ismethod)
                        if m.lower().startswith('test')
                        or m.lower() == 'run_test']
        for method_name in test_methods:
            obj.setUp()
            test = getattr(obj, method_name)
            if method_name == 'run_test' and len(test_methods) == 1:
                results[f'{mod_name}.{class_name}'] = \
                    run_test(class_name, test, num_runs)
            else:
                results[f'{mod_name}.{class_name}.{method_name}'] = \
                    run_test(method_name, test, num_runs)
            obj.tearDown()

    return results


def list_apis():
    """List the current API names on s3."""
    s3 = boto3.client('s3')
    res = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX, Delimiter='/')
    return [e['Prefix'][len(PREFIX):-1] for e in res['CommonPrefixes']]


def list_stacks():
    """List the stacks represented on s3."""
    s3 = boto3.client('s3')
    stack_names = set()
    for api_name in list_apis():
        try:
            api_prefix = f'{PREFIX}{api_name}/'
            res = s3.list_objects_v2(Bucket=BUCKET, Prefix=api_prefix,
                                     Delimiter='/')
            stack_names |= {e['Prefix'][len(api_prefix):-1]
                            for e in res['CommonPrefixes']}
        except KeyError:
            logger.error(f"Failed to inspect {api_prefix}: likely malformed "
                         f"content was added to s3.")
            continue
    return list(stack_names)


def save_results(start_time, api_name, stack_name, results):
    """Save the result of a test on s3."""
    s3 = boto3.client('s3')
    data_key = f'{PREFIX}{api_name}/{stack_name}/{start_time}.json'
    s3.put_object(Bucket=BUCKET, Key=data_key, Body=json.dumps(results))
    return
