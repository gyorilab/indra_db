import os
import subprocess
import webbrowser

from numpy import array
from datetime import datetime
from collections import defaultdict
from typing import Iterable

import click

from benchmarker.util import benchmark, list_apis, list_stacks, save_results


HERE = os.path.dirname(os.path.abspath(__file__))


@click.group()
def main():
    """The benchmarker CLI.

    The benchmarker tool allows stack deployments to be
    compared based on the time taken to run existing test corpora that utilize
    the web service.
    """


@main.command('list')
@click.argument("list_scope", type=click.Choice(["apis", "stacks"]),
                required=False)
def print_list(list_scope):
    """List the apis or stacks that have already been used."""
    def print_apis():
        print()
        print("Existing API Test Corpora")
        print("-------------------------")
        for api in list_apis():
            print(api)

    def print_stacks():
        print()
        print("Existing Tested Stacks")
        print("----------------------")
        for stack_name in list_stacks():
            print(stack_name)

    if list_scope == 'apis':
        print_apis()
    elif list_scope == 'stacks':
        print_stacks()
    else:
        print_apis()
        print_stacks()


@main.command()
@click.argument("location")
@click.argument("stack_name")
@click.argument("api_name")
@click.option("-r", "--inner-runs", default=1,
              type=click.IntRange(1, 100),
              help="Select the number of times to repeat the test in a row.")
@click.option("-R", "--outer-runs", default=1,
              type=click.IntRange(1, 100),
              help=("Select the number of times to repeat the entire suite of "
                    "tests."))
def run(location, stack_name, api_name, inner_runs, outer_runs):
    """Run the benchmarker and save the aggregate the results."""
    start_time = datetime.utcnow()

    # Run the benchmarker. Run it `outer_run` times, and we will aggregate
    # the results below.
    result_list = []
    test_names = []
    for i in range(outer_runs):
        run_result = benchmark(location, num_runs=inner_runs)
        if not test_names:
            test_names = list(run_result.keys())
        result_list.append(run_result)

    # Aggregate the results from above, either adding values to the list
    # or extending a list.
    results = {}
    for test_name in test_names:
        test_results = defaultdict(list)
        for this_result in result_list:
            test_data = this_result[test_name]
            for data_name, data_val in test_data.items():
                if isinstance(data_val, Iterable):
                    test_results[data_name].extend(data_val)
                else:
                    test_results[data_name].append(data_val)

        # Convert the default dict into a real dict.
        test_results = dict(test_results)

        # Turn the time data into an array, and calculate mean and std dev.
        time_data = array(test_results['times'])
        test_results['duration'] = time_data.mean()
        test_results['deviation'] = time_data.std()

        # Calculate the overall pass rate.
        test_results['passed'] = sum(test_results['passed'])/outer_runs

        # Add this test's aggregated results to the results object.
        results[test_name] = test_results

    for test, stats in results.items():
        print(test, stats['passed'], stats['duration'], stats['deviation'])
    save_results(start_time, api_name, stack_name, results)


@main.command()
def view():
    """Run the web service to view results."""
    basic_env = os.environ.copy()
    basic_env['FLASK_APP'] = os.path.join(HERE, "viewer_app/app.py:app")
    print("Starting web server...")
    p = subprocess.Popen(['flask', 'run', '--port', '5280'],
                         env=basic_env, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    print("Opening browser...")
    webbrowser.open("http://localhost:5280")
    print("Press Ctrl-C to exit.")
    p.wait()


if __name__ == "__main__":
    main()
