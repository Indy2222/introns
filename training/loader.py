import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence

from data import DatasetConfig, SampleType


class Dataset(Sequence):

    def __init__(
        self,
        npz_file_paths: List[Path],
        batch_size: int,
        sample_len: Optional[int] = None,
        upstream_len: Optional[int] = None,
        downstream_len: Optional[int] = None,
    ):
        assert len(set(npz_file_paths)) == len(npz_file_paths)

        self._npz_file_paths = npz_file_paths
        self.batch_size = batch_size
        self._sample_len = sample_len
        self._upstream_len = upstream_len
        self._downstream_len = downstream_len

        self.shuffle()

    def __len__(self):
        return int(np.ceil(len(self._npz_file_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size
        return self._load_samples(self._npz_file_paths[start:end])

    @property
    def window_size(self):
        if self._sample_len:
            return self._sample_len

        assert self._downstream_len is not None
        assert self._upstream_len is not None
        return self._downstream_len + self._upstream_len

    def load_all(self):
        return self._load_samples(self._npz_file_paths)

    def shuffle(self):
        print('Shuffling dataset...')
        random.shuffle(self._npz_file_paths)

    def _load_samples(self, paths):
        inputs = []
        outputs = []

        for path in paths:
            with np.load(path) as data:
                inputs.append(data['input'])
                outputs.append(data['output'])

        if self._sample_len is None:
            assert len({i.shape[0] for i in inputs}) == 1

            window_size = inputs[0].shape[0]
            centre = window_size // 2
            if self._upstream_len is not None:
                upstream_cutoff = centre - self._upstream_len
                assert upstream_cutoff >= 0
            else:
                upstream_cutoff = 0
            if self._downstream_len is not None:
                downstream_cutoff = centre + self._downstream_len
                assert downstream_cutoff <= window_size
            else:
                downstream_cutoff = window_size

            inputs = [i[upstream_cutoff:downstream_cutoff] for i in inputs]

        else:
            assert max(i.shape[0] for i in inputs) <= self._sample_len
            inputs = [
                np.pad(i, [(0, self._sample_len - i.shape[0]), (0, 0)])
                for i in inputs
            ]

        return np.array(inputs), np.array(outputs)


def load_datasets(
    features_directory: Path,
    data_directory: Path,
    dataset_config: DatasetConfig,
    batch_size: int,
):
    training_files = _get_npz_file_paths(
        features_directory=features_directory,
        data_directory=data_directory,
        organism_ids=dataset_config.training_organisms,
        sample_type=dataset_config.sample_type,
        per_organism_limit=dataset_config.training_per_organism_limit,
        positive_examples_ratio=dataset_config.positive_examples_ratio,
    )
    validation_files = _get_npz_file_paths(
        features_directory=features_directory,
        data_directory=data_directory,
        organism_ids=dataset_config.validation_organisms,
        sample_type=dataset_config.sample_type,
        per_organism_limit=dataset_config.validation_per_organism_limit,
        positive_examples_ratio=dataset_config.positive_examples_ratio,
    )

    print(
        f'Training dataset contains {len(dataset_config.training_organisms)} '
        f'organisms and {len(training_files)} samples. Validation dataset '
        f'contains {len(dataset_config.validation_organisms)} and '
        f'{len(validation_files)} samples.'
    )

    training_dataset = Dataset(training_files, batch_size,
                               dataset_config.max_sample_len,
                               dataset_config.upstream_len,
                               dataset_config.downstream_len)
    validation_dataset = Dataset(validation_files, batch_size,
                                 dataset_config.max_sample_len,
                                 dataset_config.upstream_len,
                                 dataset_config.downstream_len)

    return training_dataset, validation_dataset


def load_dataset_custom(
    features_directory: Path,
    data_directory: Path,
    organism_ids: List[str],
    sample_type: SampleType,
    per_organism_limit: int,
    positive_examples_ratio: float,
    batch_size: int,
    sample_len: Optional[int] = None,
    upstream_len: Optional[int] = None,
    downstream_len: Optional[int] = None,
    intron_min_len: Optional[int] = None,
    intron_max_len: Optional[int] = None,
    splice_min_dist: Optional[int] = None,
    splice_max_dist: Optional[int] = None,
):
    file_paths = _get_npz_file_paths(
        features_directory=features_directory,
        data_directory=data_directory,
        organism_ids=organism_ids,
        sample_type=sample_type,
        per_organism_limit=per_organism_limit,
        positive_examples_ratio=positive_examples_ratio,
        intron_min_len=intron_min_len,
        intron_max_len=intron_max_len,
        splice_min_dist=splice_min_dist,
        splice_max_dist=splice_max_dist,
    )
    return Dataset(file_paths, batch_size, sample_len=sample_len,
                   upstream_len=upstream_len, downstream_len=downstream_len)


def _get_npz_file_paths(
    features_directory: Path,
    data_directory: Path,
    organism_ids: List[str],
    sample_type: SampleType,
    per_organism_limit: int,
    positive_examples_ratio: float,
    # put these to some filter class
    intron_min_len: Optional[int] = None,
    intron_max_len: Optional[int] = None,
    splice_min_dist: Optional[int] = None,
    splice_max_dist: Optional[int] = None,
) -> List[Path]:
    npz_file_paths = []

    for organism_id in organism_ids:
        organism_dir = data_directory / organism_id

        samples_df = pd.read_csv(organism_dir / 'samples.csv', dtype={
            'path': str,
            'sample_type_name': str,
            'is_positive': bool,
        })

        if intron_min_len is not None or intron_max_len is not None:
            introns_csv = features_directory / f'introns_{organism_id}.csv'
            introns_df = pd.read_csv(introns_csv, index_col='id', dtype={
                'id': str,
                'scaffold name': str,
                'start': int,
                'end': int,
            })

            samples_df = samples_df.join(introns_df, on='feature_id')
            samples_df['feature_len'] = samples_df['end'] - samples_df['start']

            if intron_min_len is not None:
                samples_df = samples_df.loc[np.logical_or(
                    ~samples_df['is_positive'],
                    samples_df['feature_len'] >= intron_min_len
                )]
            if intron_max_len is not None:
                samples_df = samples_df.loc[np.logical_or(
                    ~samples_df['is_positive'],
                    samples_df['feature_len'] <= intron_max_len
                )]

        samples_df = _filter_splice_dist(
            organism_id=organism_id,
            sample_type=sample_type,
            features_directory=features_directory,
            samples_df=samples_df,
            splice_min_dist=splice_min_dist,
            splice_max_dist=splice_max_dist,
        )

        assert len(set(samples_df['path'])) == len(samples_df['path'])
        positive_samples_df = samples_df.loc[
            (samples_df['sample_type_name'] == sample_type.value)
            & samples_df['is_positive']
        ]
        negative_samples_df = samples_df.loc[
            (samples_df['sample_type_name'] == sample_type.value)
            & ~samples_df['is_positive']
        ]

        positive_samples_df, negative_samples_df = _clip_data(
            positive_examples=positive_samples_df,
            negative_examples=negative_samples_df,
            positive_ratio=positive_examples_ratio,
            total_limit=per_organism_limit,
        )

        if not positive_samples_df.empty:
            npz_file_paths.extend(
                positive_samples_df['path'].apply(lambda x: organism_dir / x))
        if not negative_samples_df.empty:
            npz_file_paths.extend(
                negative_samples_df['path'].apply(lambda x: organism_dir / x))

    return npz_file_paths


def _clip_data(
    positive_examples: pd.DataFrame,
    negative_examples: pd.DataFrame,
    positive_ratio: float,
    total_limit: int,
) -> Tuple[List[str], List[str]]:
    negative_ratio = 1 - positive_ratio
    positive_len, negative_len = len(positive_examples), len(negative_examples)
    total_len = positive_len + negative_len

    factor = 1.0
    positive_clip = total_len * positive_ratio
    negative_clip = total_len * negative_ratio

    if positive_clip > positive_len:
        factor = min(factor, positive_len / positive_clip)
    elif negative_clip > negative_len:
        factor = min(factor, negative_len / negative_clip)

    total_clip = positive_clip + negative_clip
    if total_limit < total_clip:
        factor = min(factor, total_limit / total_clip)

    positive_clip = int(min(positive_clip * factor, positive_len))
    negative_clip = int(min(negative_clip * factor, negative_len))

    if positive_clip > 0:
        positive_examples = positive_examples.sample(
            n=positive_clip, replace=False, random_state=456)
    else:
        positive_examples = positive_examples[:0]

    if negative_clip > 0:
        negative_examples = negative_examples.sample(
            n=negative_clip, replace=False, random_state=456)
    else:
        negative_examples = negative_examples[:0]

    return positive_examples, negative_examples


def _filter_splice_dist(
    organism_id: str,
    sample_type: SampleType,
    features_directory: Path,
    samples_df: pd.DataFrame,
    splice_min_dist: Optional[int] = None,
    splice_max_dist: Optional[int] = None,
):
    if splice_min_dist is None and splice_max_dist is None:
        return samples_df

    genes_csv = features_directory / f'genes_{organism_id}.csv'
    genes_df = pd.read_csv(genes_csv, index_col='id', dtype={
        'id': str,
        'scaffold name': str,
        'start': int,
        'end': int,
    })
    samples_df = samples_df.join(genes_df, on='feature_id')

    introns_csv = features_directory / f'introns_{organism_id}.csv'
    introns_df = pd.read_csv(introns_csv, index_col='id', dtype={
        'id': str,
        'scaffold name': str,
        'start': int,
        'end': int,
    })

    if sample_type == SampleType.DONOR:
        position_column = 'start'
    elif sample_type == SampleType.ACCEPTOR:
        position_column = 'end'
    else:
        raise ValueError(
            'Only splice-site candidates could be filtered by distance.')

    samples_df.sort_values(by='absolute_position', inplace=True)
    introns_df.sort_values(by=position_column, inplace=True)

    samples_df = pd.merge_asof(
        left=samples_df,
        right=introns_df,
        left_on='absolute_position',
        right_on=position_column,
        left_by='scaffold_name',
        right_by='scaffold name',
        direction='nearest',
        suffixes=['_sample', '']
    )

    samples_df['splice_site_dist'] = (
        samples_df[position_column] - samples_df['absolute_position'])

    if splice_min_dist is not None:
        samples_df = samples_df.loc[np.logical_or(
            samples_df['is_positive'],
            samples_df['splice_site_dist'] >= splice_min_dist
        )]
    if splice_max_dist is not None:
        samples_df = samples_df.loc[np.logical_or(
            samples_df['is_positive'],
            samples_df['splice_site_dist'] <= splice_max_dist
        )]

    return samples_df
