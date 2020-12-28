import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


def store_plot(model_dir: Path, output_dir: Path):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    _generate_plot(model_dir, False)
    plt.savefig(
        str(output_dir / 'training-loss.pdf'),
        transparent=True,
    )


def show_plot(model_dir: Path, last_epoch_only: bool):
    _generate_plot(model_dir, last_epoch_only)
    plt.show()


def _generate_plot(model_dir: Path, last_epoch_only: bool):
    with open(model_dir / 'loss.json') as fp:
        history = json.load(fp)

    samples_seen_train, losses_train = zip(*history['training'])
    samples_seen_train = np.array(samples_seen_train)
    losses_train = np.array(losses_train)

    validation = history['validation']
    if len(validation) > 0:
        samples_seen_val, losses_val = zip(*validation)
        samples_seen_val = np.array(samples_seen_val)
        losses_val = np.array(losses_val)
    else:
        samples_seen_val, losses_val = None, None

    # Clip the uninteresting / high-loss beginning.
    clip_samples_count = 10000
    if last_epoch_only and samples_seen_val is not None:
        clip_samples_count = int(samples_seen_val[-1])
        samples_seen_val, losses_val = samples_seen_val[-1:], losses_val[-1:]

    clip_point = np.argmax(samples_seen_train > clip_samples_count)
    samples_seen_train = samples_seen_train[clip_point:]
    losses_train = losses_train[clip_point:]

    losses_lowess = sm.nonparametric.lowess(
        endog=losses_train,
        exog=samples_seen_train,
        frac=0.3,
        is_sorted=True,
        return_sorted=False,
    )

    samples_seen_train = samples_seen_train // 1000
    plt.scatter(samples_seen_train[::100], losses_train[::100], color='blue',
                label='training loss', s=0.9)
    plt.plot(samples_seen_train, losses_lowess, color='green',
             label='LOWESS of training loss')

    if samples_seen_val is not None:
        samples_seen_val = samples_seen_val // 1000
        plt.scatter(samples_seen_val, losses_val, color='red', marker='x',
                    label='validation loss')

    plt.xlabel('Number of Samples [Thousands]')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
