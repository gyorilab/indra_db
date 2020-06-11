import os
from argparse import ArgumentParser


def benchmark(loc):
    if not os.path.exists(loc):
        raise ValueError(f"No such file or direcory: {loc}")
    elif os.path.isdir(loc):
        # find all dirs
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
