import json
import os
import random
from enum import Enum
from pathlib import Path
from typing import List, Optional

import pandas as pd

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
# TODO: make this an argument?
ORGANISM_CSV_FILE = PROJECT_DIRECTORY / 'data' / 'organisms.csv'


class SampleType(Enum):

    ACCEPTOR = 'acceptor'
    DONOR = 'donor'
    INTRON = 'intron'


class DatasetConfig:

    @classmethod
    def from_path(
        cls,
        path: Path,
    ):
        with open(path) as fp:
            config_dict = json.load(fp)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(
        cls,
        config_dict: dict,
    ):
        training_organisms = config_dict['training']
        validation_organisms = config_dict['validation']
        sample_type = SampleType(config_dict['sampleType'])
        training_per_organism_limit = int(
            config_dict['trainingPerOrganismLimit'])
        validation_per_organism_limit = int(
            config_dict['validationPerOrganismLimit'])
        positive_examples_ratio = float(config_dict['positiveExamplesRatio'])

        try:
            upstream_len = int(config_dict['upstreamLen'])
        except KeyError:
            upstream_len = None
        try:
            downstream_len = int(config_dict['downstreamLen'])
        except KeyError:
            downstream_len = None
        try:
            max_sample_len = int(config_dict['maxSampleLen'])
        except KeyError:
            max_sample_len = None

        return cls(
            training_organisms=training_organisms,
            validation_organisms=validation_organisms,
            sample_type=sample_type,
            training_per_organism_limit=training_per_organism_limit,
            validation_per_organism_limit=validation_per_organism_limit,
            positive_examples_ratio=positive_examples_ratio,
            upstream_len=upstream_len,
            downstream_len=downstream_len,
            max_sample_len=max_sample_len
        )

    def __init__(
        self,
        training_organisms: List[str],
        validation_organisms: List[str],
        sample_type: SampleType,
        training_per_organism_limit: int,
        validation_per_organism_limit: int,
        positive_examples_ratio: float,
        upstream_len: Optional[int] = None,
        downstream_len: Optional[int] = None,
        max_sample_len: Optional[int] = None,
    ):
        self.training_organisms = training_organisms
        self.validation_organisms = validation_organisms
        self.sample_type = sample_type
        self.training_per_organism_limit = training_per_organism_limit
        self.validation_per_organism_limit = validation_per_organism_limit
        self.positive_examples_ratio = positive_examples_ratio
        self.upstream_len = upstream_len
        self.downstream_len = downstream_len
        self.max_sample_len = max_sample_len

    def to_dict(self):
        result = {
            'training': self.training_organisms,
            'validation': self.validation_organisms,
            'sampleType': self.sample_type.value,
            'trainingPerOrganismLimit': self.training_per_organism_limit,
            'validationPerOrganismLimit': self.validation_per_organism_limit,
            'positiveExamplesRatio': self.positive_examples_ratio,
        }

        if self.upstream_len is not None:
            result['upstreamLen'] = self.upstream_len
        if self.downstream_len is not None:
            result['downstreamLen'] = self.downstream_len
        if self.max_sample_len is not None:
            result['maxSampleLen'] = self.max_sample_len

        return result


def generate_dataset(
    data_directory: Path,
    json_path: Path,
    sample_type: SampleType,
    validation_fraction: float,
    training_per_organism_limit: int,
    validation_per_organism_limit: int,
    positive_examples_ratio: float,
    downstream_len: int,
    upstream_len: int,
    filter_column_name: Optional[str] = None,
    filter_column_values: Optional[List[str]] = None,
):
    """Create training and validation datasets. Validation dataset is created
    number of organisms making :param:`validation_fraction` fraction of all
    organisms  limited to :param:`validation_per_organism_limit` samples per
    organism. Training dataset is created from the remaining organisms and
    limited to :param:`training_per_organism_limit` samples per organism.

    Description of the generated dataset is stored to a JSON file which can be
    used for model training and evaluation.

    :param data_directory: a path to a directory with training and validation
        samples.
    :param json_path: path to JSON file where description of the
        dataset will be stored.
    :param sample_type: dataset is created from splice-site donors, splice-site
        acceptors or introns only.
    :param validation_fraction: number of matched organisms multiplied by this
        number gives number of organisms included in validation dataset.
    :param training_per_organism_limit: maximum number of positive and
        negative (sum) samples for a single organism in training dataset.
    :param positive_ratio: how large fraction of total generated examples will
        be positive examples. For example set this to 0.5 to generate a
        balanced dataset.
    :param upstream_len: number of nucleotides upstream from splice-site to
        include in samples.
    :param downstream_len: number of nucleotides downstream from splice-site to
        include in samples.
    :param filter_column_name: both training and validation datasets are
        limited to organisms with this SQL column equal to one of given values.
    :param filter_column_values: a list of possible values of
        :param:`filter_column_name`, non-matching organisms are not included.
    """

    if validation_fraction <= 0 or validation_fraction >= 1:
        raise ValueError(
            '`validation_fraction` must be between 0 and 1 exclusive.')

    organisms_df = pd.read_csv(ORGANISM_CSV_FILE)

    if filter_column_name:
        query_expr = ' OR '.join(
            f'{filter_column_name} == "{v}"'
            for v in filter_column_values
        )
        organisms_df.query(query_expr, inplace=True)

    organism_ids = [str(o) for o in organisms_df['organism_id']]

    # TODO: remove this once all organisms are generated
    organism_ids = [
        d for d in organism_ids
        if os.path.exists(data_directory / d)
    ]

    if len(organism_ids) < 2:
        raise Exception(f'Number of matched organisms is {len(organism_ids)} '
                        f'but at least 2 organisms must be matched.')

    random.shuffle(organism_ids)

    num_validation_organism = len(organism_ids) * validation_fraction
    num_validation_organism = int(min(
        len(organism_ids) - 1, max(1, num_validation_organism)))
    assert 1 <= num_validation_organism < len(organism_ids)

    training_organisms = organism_ids[num_validation_organism:]
    validation_organisms = organism_ids[:num_validation_organism]

    dataset_config = DatasetConfig(
        training_organisms=training_organisms,
        validation_organisms=validation_organisms,
        sample_type=sample_type,
        training_per_organism_limit=training_per_organism_limit,
        validation_per_organism_limit=validation_per_organism_limit,
        positive_examples_ratio=positive_examples_ratio,
        upstream_len=upstream_len,
        downstream_len=downstream_len,
    )

    with open(json_path, 'w') as fp:
        json.dump(dataset_config.to_dict(), fp)
