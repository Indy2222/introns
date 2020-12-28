#! /usr/bin/env python3

"""Dataset Generation tool.

Usage:
  dataset.py --data-directory <data-directory> \
    --json-path <json-path> \
    --sample-type <sample-type> \
    --training-per-organism-limit <training-per-organism-limit> \
    --validation-per-organism-limit <validation-per-organism-limit> \
    --positive-examples-ratio <positive-examples-ratio> \
    --upstream-len <upstream-len> \
    --downstream-len <downstream-len> \
    [--filter-column-name <filter-column-name>] \
    [--filter-column-values <filter-column-values>] \
    [--validation-fraction <validation-fraction>]
  dataset.py (-h | --help)
  dataset.py --version

Options:
  -h --help                 Show this screen.
  --version                 Show version.
  --sample-type             One of `donor`, `acceptor` and `intron`.
  --filter-column-name      Organisms are stored an SQL table. Filter to
                            organisms whose column <filter-column-name> is
                            equal to one of <filter-column-values>.
  --filter-column-values    Comma separated list of values.
  --validation-fraction     Multiple of total matched organisms to be used in
                            validation dataset. Remaining organisms are used in
                            training dataset. This has to be a number between 0
                            and 1 (exclusive). Defaults to 0.1.
  --upstream-len            Number of nucleotides upstream from splice-site to
                            include in input to the NN.
  --downstream-len          Number of nucleotides downstream from splice-site
                            to include in input to the NN.
"""

from pathlib import Path

from docopt import docopt

from data import SampleType, generate_dataset


def main():
    arguments = docopt(__doc__, version='Dataset Genration 1.0')

    data_directory = Path(arguments['<data-directory>'])
    json_path = Path(arguments['<json-path>'])
    sample_type = SampleType(arguments['<sample-type>'])

    training_per_organism_limit = int(
        arguments['<training-per-organism-limit>'])
    assert training_per_organism_limit > 0
    validation_per_organism_limit = int(
        arguments['<validation-per-organism-limit>'])
    assert validation_per_organism_limit > 0

    positive_examples_ratio = float(arguments['<positive-examples-ratio>'])
    assert 0.0 < positive_examples_ratio < 1.0

    filter_column_name = arguments['<filter-column-name>']
    if filter_column_name is not None:
        filter_column_values = arguments['<filter-column-values>'].split(',')
    else:
        filter_column_values = None

    validation_fraction = arguments['<validation-fraction>']
    if validation_fraction is None:
        validation_fraction = 0.1
    else:
        validation_fraction = float(validation_fraction)
    assert 0.0 < validation_fraction < 1.0

    upstream_len = int(arguments['<upstream-len>'])
    downstream_len = int(arguments['<downstream-len>'])

    generate_dataset(
        data_directory=data_directory,
        json_path=json_path,
        sample_type=sample_type,
        validation_fraction=validation_fraction,
        training_per_organism_limit=training_per_organism_limit,
        validation_per_organism_limit=validation_per_organism_limit,
        positive_examples_ratio=positive_examples_ratio,
        upstream_len=upstream_len,
        downstream_len=downstream_len,
        filter_column_name=filter_column_name,
        filter_column_values=filter_column_values,
    )


if __name__ == '__main__':
    main()
