#! /usr/bin/env python3

"""Model evaluation tool. This tool generates various statistics and plots for
all validation organisms separately and together to sub-directories in output
directory.

Usage:
  evaluate.py --data-directory <data-directory> \
    --features-directory <features-directory> \
    --json-path <json-path> \
    --model-dir <model-dir> \
    --output-dir <output-dir> \
    [--organism-ids <organism-ids>]
  evaluate.py (-h | --help)
  evaluate.py --version

Options:
  -h --help                  Show this screen.
  --version                  Show version.
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
from tensorflow.keras.models import load_model
from scipy.stats import chi2
from sklearn.metrics import auc, confusion_matrix, roc_curve

import lossplot
from data import DatasetConfig, SampleType
from loader import load_dataset_custom


# This needs to be enough to measure improvements with statistical significance
NUM_DESIRED_SAMPLES = 90000


def main():
    arguments = docopt(__doc__, version='Evaluation 1.0')
    data_directory = Path(arguments['<data-directory>'])
    features_directory = Path(arguments['<features-directory>'])
    json_path = Path(arguments['<json-path>'])
    model_dir = Path(arguments['<model-dir>'])
    model_path = model_dir / 'model.h5'
    output_dir = Path(arguments['<output-dir>'])

    lossplot.store_plot(model_dir, output_dir)
    dataset_config = DatasetConfig.from_path(json_path)

    organism_ids = arguments['<organism-ids>']
    if organism_ids is None:
        organism_ids = dataset_config.validation_organisms
    else:
        organism_ids = organism_ids.split(',')

    model = load_model(str(model_path))

    evaluate(
        dataset_config=dataset_config,
        features_directory=features_directory,
        data_directory=data_directory,
        output_dir=output_dir / 'all',
        evaluation_name='all_organisms',
        organism_ids=organism_ids,
        model=model,
        ritch=True,
    )

    for organism_id in organism_ids:
        evaluate(
            dataset_config=dataset_config,
            data_directory=data_directory,
            features_directory=features_directory,
            output_dir=output_dir / 'per_organism',
            evaluation_name=organism_id,
            organism_ids=[organism_id],
            model=model,
            ritch=False,
        )


def evaluate(
    dataset_config: DatasetConfig,
    features_directory: Path,
    data_directory: Path,
    output_dir: Path,
    evaluation_name: str,
    organism_ids: List[str],
    model,
    ritch: bool,
):
    print(f'Going to evaluate {evaluation_name}...')

    if dataset_config.sample_type == SampleType.INTRON:
        # TODO: this should not be used once masking is implemented
        sample_len = 200
    else:
        sample_len = None

    if ritch:
        per_organism_limit = NUM_DESIRED_SAMPLES // len(organism_ids)
    else:
        per_organism_limit = 1000 // len(organism_ids)

    inputs, outputs = load_dataset_custom(
        features_directory=features_directory,
        data_directory=data_directory,
        organism_ids=organism_ids,
        sample_type=dataset_config.sample_type,
        per_organism_limit=per_organism_limit,
        positive_examples_ratio=0.5,
        batch_size=16,
        sample_len=sample_len,
        upstream_len=dataset_config.upstream_len,
        downstream_len=dataset_config.downstream_len,
    ).load_all()
    print(f'Loaded {len(outputs)} samples from {len(organism_ids)} '
          f'organisms...')

    if len(outputs) < 100:
        print(f'Cannot generate statistics for {evaluation_name} because '
              f'insufficient number of samples.')
        return

    output_dir = output_dir / evaluation_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    predictions = model.predict(inputs)
    predictions = np.squeeze(predictions)

    plt.close('all')

    generate_confusion_matrix(output_dir, outputs, predictions,
                              json_only=not ritch)
    generate_roc_curve(output_dir, outputs, predictions,
                       json_only=not ritch)

    if not ritch:
        return

    generate_prediction_hist(output_dir, outputs, predictions)
    generate_prediction_cdf(output_dir, outputs, predictions)
    compute_effect_of_misreads(output_dir, inputs, outputs, predictions)


def generate_roc_curve(
    output_dir: Path,
    outputs: np.array,
    predictions: np.array,
    json_only: Optional[bool] = False,
):
    fpr, tpr, _ = roc_curve(outputs, predictions)
    auc_ = auc(fpr, tpr)

    with open(output_dir / 'auc.json', 'w') as fp:
        json.dump(auc_, fp)

    if json_only:
        return

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')

    #plt.show()

    plt.savefig(
        str(output_dir / 'roc-curve.pdf'),
        transparent=True,
    )


def generate_confusion_matrix(
    output_dir: Path,
    outputs: np.array,
    predictions: np.array,
    threshold: Optional[float] = 0.5,
    json_only: Optional[bool] = False,
):
    cm = confusion_matrix(
        outputs > 0.5, predictions > threshold, normalize='true')
    tn, fp, fn, tp = cm.ravel()

    matrix_dict = {
        'tp': float(tp),
        'fp': float(fp),
        'tn': float(tn),
        'fn': float(fn),
    }

    with open(output_dir / 'confusion-matrix.json', 'w') as fp:
        json.dump(matrix_dict, fp)

    if json_only:
        return

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.xticks(np.arange(2), ['Negative', 'Positive'])
    plt.yticks(np.arange(2), ['Negative', 'Positive'],
               rotation=90, va='center')
    plt.ylim((1.5, -0.5))

    labels = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            text = f'{labels[i][j]} = {cm[i][j] * 100:0.01f}%'
            plt.text(j, i, text, va='center', ha='center')

    #plt.show()

    plt.savefig(
        str(output_dir / 'confusion-matrix.pdf'),
        transparent=True,
    )


def generate_prediction_hist(output_dir, outputs, predictions):
    positions, positive, negative = _generate_hist(
        output_dir, outputs, predictions)

    positive_prob, positive_low, positive_high = positive
    negative_prob, negative_low, negative_high = negative

    plt.figure()

    plt.plot(positions, positive_prob, label='positive')
    plt.fill_between(
        positions,
        positive_low,
        positive_high,
        alpha=0.2,
        label=r'positive 95% CI',
    )

    plt.plot(positions, negative_prob, label='negative')
    plt.fill_between(
        positions,
        negative_low,
        negative_high,
        alpha=0.2,
        label=r'negative 95% CI',
    )

    plt.yscale('log')
    plt.xlabel('Prediction')
    plt.ylabel('Density')
    plt.title('Prediction Distribution')

    plt.legend()
    #plt.show()

    plt.savefig(
        str(output_dir / 'prediction-hist.pdf'),
        transparent=True,
    )


def generate_prediction_cdf(output_dir, outputs, predictions):
    positions, (positive_prob, _, _), (negative_prob, _, _) = _generate_hist(
        output_dir, outputs, predictions)

    positive_cdf = np.cumsum(positive_prob)
    negative_cdf = np.cumsum(negative_prob)

    positions = np.insert(positions, 0, 0.0)
    positive_cdf = np.insert(positive_cdf, 0, 0.0)
    negative_cdf = np.insert(negative_cdf, 0, 0.0)

    max_index = np.argmax(negative_cdf - positive_cdf)
    max_x = positions[max_index]
    max_y = positive_cdf[max_index]
    max_dy = negative_cdf[max_index] - positive_cdf[max_index]

    plt.figure()

    plt.plot(positions, positive_cdf, label='positive')
    plt.plot(positions, negative_cdf, label='negative')

    plt.arrow(max_x, max_y, 0, max_dy, head_width=0.02,
              length_includes_head=True)
    plt.annotate(
        f'max $\Delta$ {max_dy:0.2f} at {max_x:0.2f}',
        va='center',
        ha='right',
        xy=(max_x - 0.01, max_y + 0.5 * max_dy),
        rotation=90
    )

    plt.xlabel('Prediction')
    plt.ylabel('Cumulative Distribution')
    plt.title('Prediction CDF')

    plt.legend()
    #plt.show()

    plt.savefig(
        str(output_dir / 'prediction-cdf.pdf'),
        transparent=True,
    )


def _generate_hist(output_dir, outputs, predictions):
    num_bins = 25
    positive_hist, positive_bins = np.histogram(
        predictions[outputs == 1.0], bins=num_bins, range=(0.0, 1.0))
    negative_hist, negative_bins = np.histogram(
        predictions[outputs == 0.0], bins=num_bins, range=(0.0, 1.0))

    positive_low, positive_high = poisson_ci_histogram(positive_hist, 0.05)
    negative_low, negative_high = poisson_ci_histogram(negative_hist, 0.05)

    positive_total = np.sum(positive_hist)
    negative_total = np.sum(negative_hist)
    positive_prob = positive_hist / positive_total
    negative_prob = negative_hist / negative_total
    positive_low = positive_low / positive_total
    positive_high = positive_high / positive_total
    negative_low = negative_low / negative_total
    negative_high = negative_high / negative_total

    half_step = 1.0 / (2 * num_bins)
    positions = np.linspace(half_step, 1.0 - half_step, num=num_bins)

    positive = (positive_prob, positive_low, positive_high)
    negative = (negative_prob, negative_low, negative_high)

    return positions, positive, negative


def poisson_ci_histogram(
    histogram: np.array,
    alpha: float,
) -> Tuple[np.array, np.array]:
    low = np.zeros_like(histogram, dtype=np.float32)
    high = np.zeros_like(histogram, dtype=np.float32)

    for i, k in enumerate(histogram):
        low[i], high[i] = poisson_ci(k, alpha)

    return low, high


def poisson_ci(k: int, alpha: float) -> float:
    # https://en.wikipedia.org/wiki/Poisson_distribution#Confidence_interval
    low = chi2.ppf(alpha / 2, 2 * k) / 2
    high = chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2
    if k == 0:
        low = 0.0
    return low, high


def compute_effect_of_misreads(output_dir, inputs, outputs, predictions):

    outputs = outputs.squeeze()
    predictions = predictions.squeeze()

    def mse(mask):
        return np.square(outputs[mask] - predictions[mask]).mean()

    # TODO: What if misreads are biased towards some organisms?
    has_misreads = np.any(np.array(inputs).argmax(axis=-1) == 4, axis=-1)
    mse_without_misreads = mse(np.logical_not(has_misreads))
    mse_with_misreads = mse(has_misreads)

    misreads_dict = {
        'numSamples': len(outputs),
        'numSamplesWithMisreads': int(has_misreads.sum()),
        'mseWithoutMisreads': float(mse_without_misreads),
        'mseWithMisreads': float(mse_with_misreads),
    }

    with open(output_dir / 'misreads.json', 'w') as fp:
        json.dump(misreads_dict, fp)


if __name__ == '__main__':
    main()
