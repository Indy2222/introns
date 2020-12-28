"""This tool produces various statistics for annotated introns.

Usage:
  intron-stats.py \
    --organism-csv <organism-csv> \
    --feature-dir <feature-dir> \
    --output-dir <output-dir>
  intron-stats.py (-h | --help)
  intron-stats.py --version

Options:
  -h --help                  Show this screen.
  --version                  Show version.

"""

import json
from collections import defaultdict
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docopt import docopt

from intron_stats import __version__


NUM_BINS = 50
LAST_BIN = 150


def main():
    arguments = docopt(__doc__, version=__version__)
    organism_csv_path = Path(arguments['<organism-csv>'])
    feature_dir = Path(arguments['<feature-dir>'])
    output_dir = Path(arguments['<output-dir>'])

    total_num_introns = 0
    total_genes = 0
    total_hist = np.zeros((NUM_BINS,), dtype=np.float64)

    phyla = defaultdict(lambda: [])

    organism_df = pd.read_csv(str(organism_csv_path))
    for _, row in organism_df.iterrows():
        organism_id = row['organism_id']
        introns_path = feature_dir / f'introns_{organism_id}.csv'

        try:
            introns_df = pd.read_csv(str(introns_path))
        except pd.errors.EmptyDataError:
            # This organism does not have any annotated introns
            pass

        num_introns, hist, lenghts = organism_stats(introns_df)
        total_num_introns += num_introns
        total_hist += hist

        genes_path = feature_dir / f'genes_{organism_id}.csv'
        try:
            genes_df = pd.read_csv(str(genes_path))
        except pd.errors.EmptyDataError:
            # This organism does not have any annotated introns
            pass
        num_genes = genes_df.shape[0]
        total_genes += num_genes

        save_stats(output_dir / f'{organism_id}-stats.json', hist, num_introns,
                   num_genes)

        phulym_lens = phyla[row['taxonomy_phylum']]
        phulym_lens.append(lenghts)

    save_stats(output_dir / 'total-stats.json', total_hist, total_num_introns,
               total_genes)
    plot_distribution(output_dir, total_hist)

    box_plot_lens, labels = [], []
    for phylum, lens in phyla.items():
        box_plot_lens.append(np.concatenate(lens))
        labels.append(phylum)
    plot_boxes(output_dir, box_plot_lens, labels)


def organism_stats(introns: pd.DataFrame) -> (int, np.array):
    num_introns = len(introns)
    introns['len'] = introns['end'] - introns['start']
    hist = np.histogram(introns['len'], bins=NUM_BINS, range=(0, LAST_BIN))[0]
    return num_introns, hist, np.array(introns['len'])


def save_stats(
    path: Path,
    histogram: np.array,
    num_introns: int,
    num_genes: int,
):
    with open(path, 'w') as fp:
        json.dump({
            'numOfIntrons': num_introns,
            'numOfGenes': num_genes,
            'histogram': {
                'start': 0,
                'end': LAST_BIN,
                'values': [int(v) for v in histogram],
            }
        }, fp)


def plot_distribution(output_dir: Path, histogram: np.array):
    plt.figure()

    lengths = np.linspace(0, LAST_BIN, NUM_BINS)
    histogram /= histogram.sum()
    plt.plot(lengths, histogram)

    plt.xlabel('Intron length')
    plt.ylabel('Density')
    plt.title('Intron Length Distribution')

    plt.savefig(
        str(output_dir / 'length-hist.pdf'),
        transparent=True,
    )


def plot_boxes(output_dir: Path, lenghts: List[np.array], labels: List[str]):
    plt.figure()

    plt.boxplot(
        x=lenghts,
        vert=True,
        labels=labels,
        showfliers=False,
        whis=(5, 95),
    )

    plt.title('Per Phylum Intron Lenghts')
    plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.tight_layout()

    plt.savefig(
        str(output_dir / 'box-plot.pdf'),
        transparent=True,
    )


if __name__ == '__main__':
    main()
