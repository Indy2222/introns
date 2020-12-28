#! /usr/bin/env python3

"""Generate plot with dependency of false positive rate to distance to closes
true splice-site.

Usage:
  intron-dist-fp.py \
--data-directory <data-directory> \
--features-directory <features-directory> \
--json-path <json-path> \
--model-dir <model-dir> \
--output-dir <output-dir> \
[--organism-ids <organism-ids>]
  intron-dist-fp.py (-h | --help)
  intron-dist-fp.py --version

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

NUMBER_OF_SAMPLES = 1000
MIN_DIST = -30
MAX_DIST = 30
STEP_SIZE = 1


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

    min_dists = list(
        m
        for m in range(MIN_DIST, MAX_DIST, STEP_SIZE)
        if STEP_SIZE > 1 or m != 0
    )
    dists = np.array([
        m + (STEP_SIZE // 2) for m in min_dists], dtype=np.int32)
    mean_errors = np.zeros_like(dists, dtype=np.float32)
    mean_error_sem = np.zeros_like(dists, dtype=np.float32)
    fpr = np.zeros_like(dists, dtype=np.float32)

    for i, min_dist in enumerate(min_dists):
        max_dist = min_dist + STEP_SIZE - 1
        mean_errors[i], mean_error_sem[i], fpr[i] = get_mean_error_and_fpr(
            dataset_config=dataset_config,
            features_directory=features_directory,
            data_directory=data_directory,
            organism_ids=organism_ids,
            model=model,
            splice_min_dist=min_dist,
            splice_max_dist=max_dist,
        )

    plot(dists, mean_errors, mean_error_sem, fpr, output_dir)


def plot(
    dists: np.array,
    mean_errors: np.array,
    mean_error_sem: np.array,
    fpr: np.array,
    output_dir: Path
):
    plt.close('all')
    plt.figure()

    plt.plot(dists, mean_errors, label='mean error')
    plt.fill_between(
        dists,
        mean_errors - mean_error_sem,
        mean_errors + mean_error_sem,
        alpha=0.2,
        label=r'mean error SEM',
    )

    plt.plot(dists, fpr, label='false positive rate')

    plt.xlabel('Distance to Nearest Splice Site')
    plt.ylabel('Mean Error')
    plt.title('Error Rate in Splice Site Neighborhood')

    plt.legend()
    #plt.show()

    plt.savefig(
        str(output_dir / 'splice-site-dist-error.pdf'),
        transparent=True,
    )


def get_mean_error_and_fpr(
    dataset_config: DatasetConfig,
    features_directory: Path,
    data_directory: Path,
    organism_ids: List[str],
    model,
    splice_min_dist: int,
    splice_max_dist: int,
):
    inputs, outputs = load_dataset_custom(
        features_directory=features_directory,
        data_directory=data_directory,
        organism_ids=organism_ids,
        sample_type=dataset_config.sample_type,
        per_organism_limit=max(1, NUMBER_OF_SAMPLES // len(organism_ids)),
        positive_examples_ratio=0,
        batch_size=16,
        sample_len=None,
        upstream_len=dataset_config.upstream_len,
        downstream_len=dataset_config.downstream_len,
        splice_min_dist=splice_min_dist,
        splice_max_dist=splice_max_dist,
    ).load_all()

    assert outputs.sum() == 0

    predictions = model.predict(inputs)
    predictions = np.squeeze(predictions)
    mean_error = predictions.mean()
    mean_error_sem = np.std(predictions) / np.sqrt(predictions.shape[0])

    fpr = (predictions > 0.5).sum() / predictions.shape[0]
    return mean_error, mean_error_sem, fpr


if __name__ == '__main__':
    main()
