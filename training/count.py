#! /usr/bin/env python3

"""Count number of true and false acceptor sites for each organism.

Usage:
  count.py --positions-dir <position-dir>  --output <output-csv-path>

Options:
  -h --help                 Show this screen.
  --version                 Show version.
"""

import os

from pathlib import Path
import numpy as np
import pandas as pd
from docopt import docopt
from tqdm import tqdm

from data import ORGANISM_CSV_FILE, SampleType


def main():
    arguments = docopt(__doc__, version='Dataset Genration 1.0')
    position_dir = Path(arguments['<position-dir>'])
    output_csv_path = Path(arguments['<output-csv-path>'])

    organisms_df = pd.read_csv(ORGANISM_CSV_FILE)

    organism_ids = list(organisms_df['organism_id'])
    true_donors, true_acceptors, false_donors, false_acceptors = [], [], [], []
    for organism_id in tqdm(organism_ids):
        true_donors.append(count(
            position_dir, organism_id, SampleType.DONOR, True))
        true_acceptors.append(count(
            position_dir, organism_id, SampleType.ACCEPTOR, True))
        false_donors.append(count(
            position_dir, organism_id, SampleType.DONOR, False))
        false_acceptors.append(count(
            position_dir, organism_id, SampleType.ACCEPTOR, False))

    counts_df = pd.DataFrame({
        'organism_id': organism_ids,
        'true_donor': true_donors,
        'true_acceptor': true_acceptors,
        'false_donor': false_donors,
        'false_acceptor': false_acceptors,
    })
    counts_df.to_csv(output_csv_path, index=False)


def count(
    positions_dir: Path,
    organism_id: str,
    splice_site_type: SampleType,
    is_positive: bool
):
    if splice_site_type not in (SampleType.ACCEPTOR, SampleType.DONOR):
        raise ValueError(f'Splice site type has to be either donor or'
                         f' acceptor, got: {splice_site_type}')

    if is_positive:
        positivity = 'positive'
    else:
        positivity = 'negative'

    npz_file_name = f'{positivity}_{splice_site_type.value}_{organism_id}.npz'
    npz_file_path = positions_dir / npz_file_name

    if not os.path.exists(npz_file_path):
        return 0

    npz = np.load(npz_file_path)
    return sum(len(npz[k]) for k in npz.keys() if k.startswith('positions'))


if __name__ == '__main__':
    main()
