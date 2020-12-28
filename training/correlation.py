#! /usr/bin/env python3

"""
Usage:
  correlation.py \
--prediction-dir <prediction-dir> \
--output-dir <output-dir> \
--organism-ids <organism-ids>
  correlation.py (-h | --help)
  correlation.py --version

Options:
  -h --help                  Show this screen.
  --version                  Show version.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docopt import docopt
from tqdm import tqdm

from data import SampleType


def main():
    arguments = docopt(__doc__, version='1.0')
    prediction_dir = Path(arguments['<prediction-dir>'])
    output_dir = Path(arguments['<output-dir>'])
    organism_ids = arguments['<organism-ids>'].split(',')

    correlations = {}

    pos_donors, neg_donors = [], []
    pos_acceptors, neg_acceptors = [], []

    for organism_id in tqdm(organism_ids):
        prediction_df = load_merged_predictions(
            prediction_dir=prediction_dir,
            organism_id=organism_id,
        )

        positive_df = prediction_df[prediction_df['is_positive_donor']]
        negative_df = prediction_df[~prediction_df['is_positive_donor']]
        negative_df = negative_df.groupby(['feature_id']).mean()

        pos_donors.append(positive_df['prediction_donor'])
        neg_donors.append(negative_df['prediction_donor'])
        pos_acceptors.append(positive_df['prediction_acceptor'])
        neg_acceptors.append(negative_df['prediction_acceptor'])

        positive_cor = np.corrcoef(
            positive_df['prediction_donor'],
            positive_df['prediction_acceptor']
        )[0, 1]
        negative_cor = np.corrcoef(
            negative_df['prediction_donor'],
            negative_df['prediction_acceptor']
        )[0, 1]

        donor_mask = positive_df['prediction_donor'] > 0.5
        acceptor_mask = positive_df['prediction_acceptor'] > 0.5
        both_mask = np.logical_and(donor_mask, acceptor_mask)
        introns_total = len(positive_df)
        introns_acceptor = sum(acceptor_mask)
        introns_donors = sum(donor_mask)
        introns_both = sum(both_mask)

        correlations[organism_id] = {
            'positive': positive_cor,
            'positiveCount': len(positive_df),
            'negative': negative_cor,
            'negativeCount': len(negative_df),
            'tpr': {
                'acceptor': introns_acceptor / introns_total,
                'donor': introns_donors / introns_total,
                'both': introns_both / introns_total,
                'mult': introns_acceptor * introns_donors / introns_total**2,
            },
        }

    with open(output_dir / 'stats.json', 'w') as fp:
        json.dump(correlations, fp)

    pos_donors = np.concatenate(pos_donors, axis=0)
    neg_donors = np.concatenate(neg_donors, axis=0)
    pos_acceptors = np.concatenate(pos_acceptors, axis=0)
    neg_acceptors = np.concatenate(neg_acceptors, axis=0)

    plot_dependency(
        a=pos_donors,
        b=pos_acceptors,
        path=output_dir / 'positive.pdf'
    )
    plot_dependency(
        a=neg_donors,
        b=neg_acceptors,
        path=output_dir / 'negative.pdf'
    )


def plot_dependency(a: np.array, b: np.array, path: Path):
    num_steps = 20
    step = 1.0 / num_steps
    ranges = [(step * i, (i + 1) * step) for i in range(num_steps)]

    a_mean = np.zeros((len(ranges),), dtype=np.float32)
    b_mean = np.zeros_like(a_mean, dtype=np.float32)
    b_median = np.zeros_like(a_mean, dtype=np.float32)
    b_std = np.zeros_like(a_mean, dtype=np.float32)

    for i, (start, end) in enumerate(ranges):
        mask = np.logical_and(a >= start, a < end)
        a_mean[i] = a[mask].mean()
        b_mean[i] = b[mask].mean()
        b_median[i] = np.median(b[mask])
        b_std[i] = np.std(b[mask])

    plt.figure()

    plt.plot(a_mean, b_mean, label='mean')
    plt.fill_between(
        a_mean,
        b_mean - b_std,
        b_mean + b_std,
        alpha=0.2,
        label=r'$\pm\sigma$'
    )
    plt.plot(a_mean, b_median, label='median')

    plt.xlabel('Donor Model Output')
    plt.ylabel('Acceptor Model Output')
    plt.title('Donor/Acceptor Models Dependency')
    plt.legend()
    plt.savefig(str(path), transparent=True)


def load_merged_predictions(
    prediction_dir: Path,
    organism_id: str,
):
    donor_df = load_predictions(
        prediction_dir=prediction_dir,
        sample_type=SampleType.DONOR,
        organism_id=organism_id,
    )
    acceptor_df = load_predictions(
        prediction_dir=prediction_dir,
        sample_type=SampleType.ACCEPTOR,
        organism_id=organism_id,
    )

    return donor_df.merge(
        right=acceptor_df,
        how='inner',
        on='feature_id',
        suffixes=['_donor', '_acceptor'],
    )


def load_predictions(
    prediction_dir: Path,
    sample_type: SampleType,
    organism_id: str,
):
    path = prediction_dir / f'{organism_id}_{sample_type.value}.csv'
    return pd.read_csv(path, dtype={
        'path': str,
        'sample_type_name': str,
        'is_positive': bool,
        'feature_id': str,
        'absolute_position': int,
        'scaffold_name': str,
        'prediction': float,
    })


if __name__ == '__main__':
    main()
