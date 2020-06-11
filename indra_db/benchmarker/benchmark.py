import os
from argparse import ArgumentParser


def benchmark(loc):
    # By default, just run in this directory
    if loc is None:
        loc = '.'

    # Extract a function name, if it was included.
    if loc.count(':') == 0:
        func_name = None
    elif loc.count(':') == 1:
        loc, func_name = loc.split(':')
    else:
        raise ValueError(f"Invalid loc: {loc}")

    # Check if the location exists, and whether it is a directory or file.
    if not os.path.exists(loc):
        raise ValueError(f"No such file or directory: {loc}")
    elif os.path.isdir(loc):
        # find all files
        pass
    elif os.path.isfile(loc):
        # handle this file
        pass
    pass


parser = ArgumentParser(description=('Run tests and benchmark time to run and '
                                     'errors.'))
parser.add_argument(dest='location')

if __name__ == '__main__':
    args = parser.parse_args()
    benchmark(args.location)
