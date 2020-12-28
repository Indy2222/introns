#! /usr/bin/env python3

"""Generate plot with dependency of false positive length to intron length.

Usage:
  intron-len-fn.py \
--data-directory <data-directory> \
--features-directory <features-directory> \
--json-path <json-path> \
--model-dir <model-dir> \
--output-dir <output-dir> \
[--organism-ids <organism-ids>]
  intron-len-fn.py (-h | --help)
  intron-len-fn.py --version

Options:
  -h --help                  Show this screen.
  --version                  Show version.
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
from tensorflow.keras.models import load_model

from data import DatasetConfig
from loader import load_dataset_custom

NUMBER_OF_SAMPLES = 10000
MIN_INTRON_LEN = 20
MAX_INTRON_LEN = 250
STEP_SIZE = 5


def main():
    arguments = docopt(__doc__, version='Evaluation 1.0')
    data_directory = Path(arguments['<data-directory>'])
    features_directory = Path(arguments['<features-directory>'])
    json_path = Path(arguments['<json-path>'])
    model_dir = Path(arguments['<model-dir>'])
    model_path = model_dir / 'model.h5'
    output_dir = Path(arguments['<output-dir>'])

    dataset_config = DatasetConfig.from_path(json_path)
    organism_ids = arguments['<organism-ids>']
    if organism_ids is None:
        organism_ids = dataset_config.validation_organisms
    else:
        organism_ids = organism_ids.split(',')

    model = load_model(str(model_path))

    min_lengths = list(range(MIN_INTRON_LEN, MAX_INTRON_LEN, STEP_SIZE))
    lengths = np.array([
        m + (STEP_SIZE // 2) for m in min_lengths], dtype=np.uint32)
    mean_errors = np.zeros_like(lengths, dtype=np.float32)
    mean_error_sem = np.zeros_like(lengths, dtype=np.float32)
    fnr = np.zeros_like(lengths, dtype=np.float32)

    for i, min_len in enumerate(min_lengths):
        mean_errors[i], mean_error_sem[i], fnr[i] = get_mean_error_and_fnr(
            dataset_config=dataset_config,
            features_directory=features_directory,
            data_directory=data_directory,
            organism_ids=organism_ids,
            model=model,
            intron_min_len=min_len,
            intron_max_len=min_len + STEP_SIZE - 1,
        )

    plot(lengths, mean_errors, mean_error_sem, fnr, output_dir)


def plot(
    lengths: np.array,
    mean_errors: np.array,
    mean_error_sem: np.array,
    fnr: np.array,
    output_dir: Path
):
    plt.close('all')
    plt.figure()

    plt.plot(lengths, mean_errors, label='mean error')
    plt.fill_between(
        lengths,
        mean_errors - mean_error_sem,
        mean_errors + mean_error_sem,
        alpha=0.2,
        label=r'mean error SEM',
    )

    plt.plot(lengths, fnr, label='false negative rate')

    plt.xlabel('Intron Length')
    plt.ylabel('Mean Error')
    plt.title('Error and Intron Length Dependency')

    plt.legend()
    #plt.show()

    plt.savefig(
        str(output_dir / 'intron-lenght-error.pdf'),
        transparent=True,
    )


def get_mean_error_and_fnr(
    dataset_config: DatasetConfig,
    features_directory: Path,
    data_directory: Path,
    organism_ids: List[str],
    model,
    intron_min_len: int,
    intron_max_len: int,
):
    inputs, outputs = load_dataset_custom(
        features_directory=features_directory,
        data_directory=data_directory,
        organism_ids=organism_ids,
        sample_type=dataset_config.sample_type,
        per_organism_limit=max(1, NUMBER_OF_SAMPLES // len(organism_ids)),
        positive_examples_ratio=1,
        batch_size=16,
        sample_len=None,
        upstream_len=dataset_config.upstream_len,
        downstream_len=dataset_config.downstream_len,
        intron_min_len=intron_min_len,
        intron_max_len=intron_max_len,
    ).load_all()

    predictions = model.predict(inputs)
    predictions = np.squeeze(predictions)
    mean_error = (1 - predictions).mean()
    mean_error_sem = np.std(predictions) / np.sqrt(predictions.shape[0])
    fnr = (predictions <= 0.5).sum() / predictions.shape[0]
    return mean_error, mean_error_sem, fnr


if __name__ == '__main__':
    main()
