#! /usr/bin/env python3

"""
Usage:
  predict-samples.py \
--data-directory <data-directory> \
--json-path <json-path> \
--model-dir <model-dir> \
--output-dir <output-dir> \
[--organism-ids <organism-ids>]
  predict-samples.py (-h | --help)
  predict-samples.py --version

Options:
  -h --help                  Show this screen.
  --version                  Show version.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from docopt import docopt
from tensorflow.keras.models import load_model
from tqdm import tqdm

from data import DatasetConfig, SampleType


def main():
    arguments = docopt(__doc__, version='1.0')
    data_directory = Path(arguments['<data-directory>'])
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

    for organism_id in tqdm(organism_ids):
        process_organism(
            model=model,
            data_directory=data_directory,
            output_dir=output_dir,
            dataset_config=dataset_config,
            organism_id=organism_id,
        )


def process_organism(
    model,
    data_directory: Path,
    output_dir: Path,
    dataset_config: DatasetConfig,
    organism_id: str,
):
    samples_df = load_samples(
        data_directory=data_directory,
        sample_type=dataset_config.sample_type,
        organism_id=organism_id,
    )

    center, start, end = None, None, None

    inputs = []
    for path in samples_df['path']:
        with np.load(path) as data:
            input_ = data['input']

        if center is None:
            center = input_.shape[0] // 2
            start = center - dataset_config.upstream_len
            assert start >= 0
            end = center + dataset_config.downstream_len
            assert end <= input_.shape[0]

        inputs.append(input_[start:end, :])

    inputs = np.array(inputs)
    predictions = np.squeeze(model.predict(inputs))

    store_predictions(
        output_dir=output_dir,
        sample_type=dataset_config.sample_type,
        samples_df=samples_df,
        predictions=predictions,
        organism_id=organism_id,
    )


def store_predictions(
    output_dir: Path,
    sample_type: SampleType,
    samples_df: pd.DataFrame,
    predictions: np.array,
    organism_id: str,
):
    assert len(predictions.shape) == 1
    assert len(samples_df) == len(predictions)
    samples_df['prediction'] = predictions

    output_path = output_dir / f'{organism_id}_{sample_type.value}.csv'
    samples_df.to_csv(
        output_path,
        columns=[
            'sample_type_name',
            'is_positive',
            'feature_id',
            'absolute_position',
            'scaffold_name',
            'prediction',
        ],
        index=False,
    )


def load_samples(
    data_directory: Path,
    sample_type: SampleType,
    organism_id: str,
):
    organism_dir = data_directory / organism_id

    samples_df = pd.read_csv(organism_dir / 'samples.csv', dtype={
        'path': str,
        'sample_type_name': str,
        'is_positive': bool,
        'feature_id': str,
        'absolute_position': int,
        'scaffold_name': str,
    })
    samples_df = samples_df.loc[
        samples_df['sample_type_name'] == sample_type.value]
    samples_df['path'] = samples_df['path'].apply(lambda x: organism_dir / x)
    return samples_df


if __name__ == '__main__':
    main()
