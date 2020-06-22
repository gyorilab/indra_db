import os
import json

import boto3
import logging
from datetime import datetime
from collections import defaultdict
from inspect import getmembers, isfunction
from importlib.util import spec_from_file_location, module_from_spec


logger = logging.getLogger('benchmark_tools')

BUCKET = 'bigmech'
PREFIX = 'indra-db/benchmarks/'


def benchmark(loc, base_name=None):
    # By default, just run in this directory
    if loc is None:
        loc = os.path.abspath('.')

    # Extract a function name, if it was included.
    if loc.count(':') == 0:
        func_name = None
    elif loc.count(':') == 1:
        loc, func_name = loc.split(':')
    else:
        raise ValueError(f"Invalid loc: {loc}")
    mod_name = os.path.basename(loc).replace('.py', '')
    if base_name:
        mod_name = base_name + '.' + mod_name

    # Check if the location exists, and whether it is a directory or file.
    # Handle the file case by recursively calling this function for each file.
    results = {}
    if not os.path.exists(loc):
        raise ValueError(f"No such file or directory: {loc}")
    elif os.path.isdir(loc):
        if func_name is not None:
            raise ValueError("To specify function, location must be a file.")
        for file in os.listdir(loc):
            new_path = os.path.join(loc, file)
            if ('test' in file and os.path.isfile(new_path)
                    and new_path.endswith('.py')):
                results.update(benchmark(new_path, base_name=mod_name))
        return results

    # Handle the case a file is specified.
    if not loc.endswith('.py'):
        raise ValueError(f"Location {loc} is not a python file.")
    print("="*len(loc))
    print(loc)
    print('-'*len(loc))
    spec = spec_from_file_location(mod_name, loc)
    test_module = module_from_spec(spec)
    try:
        spec.loader.exec_module(test_module)
    except KeyboardInterrupt:
        raise
    except Exception as err:
        logger.error(f"Failed to load {loc}, skipping...")
        logger.exception(err)
        return results

    # Run tests
    tests = (f for f, _ in getmembers(test_module, isfunction) if 'test' in f)
    for test_name in tests:
        test_results = dict.fromkeys(['passed', 'error_type', 'error_str',
                                      'duration'])
        print(test_name)
        print('-'*len(test_name))
        print("LOGS:")
        test = getattr(test_module, test_name)
        start = datetime.now()
        try:
            test()
            print('-'*len(test_name))
            print("PASSED!")
            test_results['passed'] = True
        except Exception as e:
            print('-'*len(test_name))
            print("FAILED!", type(e), e)
            test_results['passed'] = False
            test_results['error_type'] = str(type(e))
            test_results['error_str'] = str(e)
        finally:
            end = datetime.now()
            test_results['duration'] = (end - start).total_seconds()
            print()
            results[f'{mod_name}.{test_name}'] = test_results

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
        api_prefix = f'{PREFIX}{api_name}/'
        print(api_prefix)
        res = s3.list_objects_v2(Bucket=BUCKET, Prefix=api_prefix,
                                 Delimiter='/')
        print(res)
        stack_names |= {e['Prefix'][len(api_prefix):-1]
                        for e in res['CommonPrefixes']}
    return list(stack_names)


def save_results(start_time, api_name, stack_name, results):
    """Save the result of a test on s3."""
    s3 = boto3.client('s3')
    data_key = f'{PREFIX}{api_name}/{stack_name}/{start_time}.json'
    s3.put_object(Bucket=BUCKET, Key=data_key, Body=json.dumps(results))
    return
