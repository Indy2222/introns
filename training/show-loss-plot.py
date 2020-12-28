#! /usr/bin/env python3

"""Display training loss plot.

Usage:
  show-loss-plot.py --model-dir <model-dir> [--last-epoch-only]
  show-loss-plot.py (-h | --help)
  show-loss-plot.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
"""

from pathlib import Path

from docopt import docopt

from lossplot import show_plot


def main():
    arguments = docopt(__doc__, version='Evaluation 1.0')
    model_dir = Path(arguments['<model-dir>'])
    show_plot(model_dir, arguments['--last-epoch-only'])


if __name__ == '__main__':
    main()
