#! /usr/bin/env python3

"""This tool produces a set of plots showing Kullback–Leibler divergence of
model inferences on true input data and model inferences on data with single
nucleotide modifications to the input sequence on Y axis and distance from
splice-site on X axis.

Usage:
  position-dependence.py --data-directory <data-directory> \
--features-directory <features-directory> \
--json-path <json-path> \
--model-dir <model-dir> \
--output-dir <output-dir> \
[--intron-max-len <intron-min-len>] \
[--intron-min-len <intron-max-len>] \
[--organism-ids <organism-ids>]
  position-dependence.py (-h | --help)
  position-dependence.py --version

Options:
  -h --help                  Show this screen.
  --version                  Show version.

"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
from scipy.spatial import KDTree
from tensorflow.keras.models import load_model

from data import DatasetConfig
from loader import load_dataset_custom

NUM_SAMPLES = 1000
DNA_SYMBOLS = ['A', 'T', 'C', 'G', 'N']
UPSTREAM_LIMIT = 25
DOWNSTREAM_LIMIT = 75


def main():
    arguments = docopt(__doc__, version='Evaluation 1.0')
    data_directory = Path(arguments['<data-directory>'])
    features_directory = Path(arguments['<features-directory>'])
    json_path = Path(arguments['<json-path>'])
    model_dir = Path(arguments['<model-dir>'])
    model_path = model_dir / 'model.h5'
    output_dir = Path(arguments['<output-dir>'])

    intron_min_len = arguments['<intron-min-len>']
    if intron_min_len is not None:
        intron_min_len = int(intron_min_len)
    intron_max_len = arguments['<intron-max-len>']
    if intron_max_len is not None:
        intron_max_len = int(intron_max_len)

    dataset_config = DatasetConfig.from_path(json_path)

    organism_ids = arguments['<organism-ids>']
    if organism_ids is None:
        organism_ids = dataset_config.validation_organisms
    else:
        organism_ids = organism_ids.split(',')

    model = load_model(str(model_path))

    evaluate(
        model=model,
        data_directory=data_directory,
        features_directory=features_directory,
        output_dir=output_dir,
        dataset_config=dataset_config,
        organism_ids=organism_ids,
        intron_min_len=intron_min_len,
        intron_max_len=intron_max_len,
    )


def evaluate(
    model,
    data_directory: Path,
    features_directory: Path,
    output_dir: Path,
    dataset_config: DatasetConfig,
    organism_ids: Optional[list] = None,
    intron_min_len: Optional[int] = None,
    intron_max_len: Optional[int] = None,
):
    assert organism_ids
    per_organism_limit = NUM_SAMPLES / len(organism_ids)

    inputs, outputs = load_dataset_custom(
        data_directory=data_directory,
        features_directory=features_directory,
        organism_ids=organism_ids,
        sample_type=dataset_config.sample_type,
        per_organism_limit=per_organism_limit,
        positive_examples_ratio=0.5,
        batch_size=16,
        upstream_len=dataset_config.upstream_len,
        downstream_len=dataset_config.downstream_len,
        intron_min_len=intron_min_len,
        intron_max_len=intron_max_len,
    ).load_all()

    divergences = compute_divergence(
        model=model,
        inputs=inputs,
        outputs=outputs,
        upstream_len=dataset_config.upstream_len,
        downstream_len=dataset_config.downstream_len,
    )

    plt.close('all')
    for symbol, positions, positive, negative in divergences:
        plot_divergences(
            symbol=symbol,
            positions=positions,
            positive_divergence=positive,
            negative_divergence=negative,
            output_dir=output_dir,
        )


def plot_divergences(
    symbol: str,
    positions: np.array,
    positive_divergence: np.array,
    negative_divergence: np.array,
    output_dir: Path,
):
    plt.figure()

    plt.plot(positions, positive_divergence, label='positive samples')
    plt.plot(positions, negative_divergence, label='negative samples')

    plt.xlabel('distance from splice-site')
    plt.ylabel('Kullback–Leibler divergence')
    plt.title(f'Sensitivity to Swap to Symbol {symbol}')

    plt.legend()
    #plt.show()

    plt.savefig(
        str(output_dir / f'sensitivity-{symbol}.pdf'),
        transparent=True,
    )


def compute_divergence(
    model,
    inputs: np.array,
    outputs: np.array,
    upstream_len: int,
    downstream_len: int,
):
    assert len(inputs.shape) == 3
    assert inputs.shape[1] == downstream_len + upstream_len
    assert inputs.shape[2] == len(DNA_SYMBOLS)

    start = -min(UPSTREAM_LIMIT, upstream_len)
    stop = min(DOWNSTREAM_LIMIT, downstream_len)

    original_probas = np.squeeze(model.predict(inputs))
    positive_samples = outputs > 0.5
    positions = list(range(start, stop))
    np_positions = np.array(positions)

    max_positive_divergence = None
    max_negative_divergence = None
    min_positive_divergence = None
    min_negative_divergence = None

    for symbol in ('A', 'T', 'C', 'G'):
        positivie_divergence = np.zeros((len(positions)), dtype=np.float32)
        negative_divergence = np.zeros((len(positions)), dtype=np.float32)

        for position in positions:
            seq_index = position + upstream_len
            res_index = position - start
            perturbated = perturbate_data(inputs, symbol, seq_index)
            probas = np.squeeze(model.predict(perturbated))

            positivie_divergence[res_index] = estimate_divergence(
                original_probas[positive_samples], probas[positive_samples])
            negative_divergence[res_index] = estimate_divergence(
                original_probas[~positive_samples], probas[~positive_samples])

        if max_positive_divergence is None:
            max_positive_divergence = np.copy(positivie_divergence)
            max_negative_divergence = np.copy(negative_divergence)
            min_positive_divergence = np.copy(positivie_divergence)
            min_negative_divergence = np.copy(negative_divergence)
        else:
            max_positive_divergence = np.maximum(
                max_positive_divergence, positivie_divergence)
            max_negative_divergence = np.maximum(
                max_negative_divergence, negative_divergence)
            min_positive_divergence = np.minimum(
                min_positive_divergence, positivie_divergence)
            min_negative_divergence = np.minimum(
                min_negative_divergence, negative_divergence)

        yield (symbol, np_positions, positivie_divergence, negative_divergence)

    yield ('max(ATCG)', np_positions, max_positive_divergence,
           max_negative_divergence)
    yield ('min(ATCG)', np_positions, min_positive_divergence,
           min_negative_divergence)


def estimate_divergence(p_samples, q_samples):
    # https://www.princeton.edu/~kulkarni/Papers/Journals/j068_2009_WangKulVer_TransIT.pdf
    p_samples = p_samples.reshape((p_samples.shape[0], 1))
    q_samples = q_samples.reshape((q_samples.shape[0], 1))
    n = p_samples.shape[0]
    epsilon = 0.0000000001
    k = 10

    tree = KDTree(p_samples)
    p_dists, _ = tree.query(p_samples, k + 1)
    p_dists = np.maximum(np.array(list(p[k] for p in p_dists)), epsilon)
    q_dists, _ = tree.query(q_samples, k)
    q_dists = np.maximum(np.array(list(q[k - 1] for q in q_dists)), epsilon)

    divergence = np.log(q_dists / p_dists).sum() / n + np.log(n / (n - 1))
    return max(divergence, 0)


def perturbate_data(inputs: np.array, symbol: str, index: int) -> np.array:
    inputs = np.copy(inputs)
    vector = np.zeros((5,), dtype=np.float32)
    vector[DNA_SYMBOLS.index(symbol)] = 1.0
    inputs[:, index] = vector[np.newaxis, :]
    return inputs


if __name__ == '__main__':
    main()
