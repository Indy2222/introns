#!/usr/bin/env python3


"""Classify splice sites with a neural network.

Usage:
  classify-splice-sites.py <data-file-name> <model-file-name> <window-inner> \
    <window-outer> <site> [-v] [-c]
  classify-splice-sites.py (-h | --help)
  classify-splice-sites.py --version

Options:
  -c --cpus     Dummy argument kept for compatibility reasons.
  -h --help     Show this screen.
  --version     Show version.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from docopt import docopt
from keras.models import load_model

DNA_SYMBOLS = np.array(['A', 'T', 'C', 'G', 'N'])


def main():
    arguments = docopt(__doc__, version='1.0')

    data_file = Path(arguments['<data-file-name>'])
    model_file = Path(arguments['<model-file-name>'])

    window_inner = int(arguments['<window-inner>'])
    assert window_inner == 200
    window_outer = int(arguments['<window-outer>'])
    assert window_outer == 200

    site = arguments['<site>']
    assert site in ('donor', 'acceptor')

    input_df = pd.read_csv(str(data_file), sep=';')
    model = load_model(str(model_file))

    inputs = prepare_inputs(input_df)
    predictions = model.predict(inputs)
    predictions = np.squeeze(predictions).round().astype(np.int32)
    predictions[predictions == 0] = -1

    output_df = input_df.assign(pred=pd.Series(list(predictions)))
    output_df.to_csv(sys.stdout, sep=';', index=False)


def prepare_inputs(input_df: pd.DataFrame):
    sequences = []
    for sequence in input_df['sequence']:
        sequence = str(sequence)
        sequence = sequence.upper()
        assert set(sequence).issubset({'A', 'C', 'T', 'G', 'N'})
        sequence = np.array(list(sequence))
        sequences.append((sequence[2:, None] == DNA_SYMBOLS).astype(np.float32))

    return np.array(sequences)


if __name__ == '__main__':
    main()
