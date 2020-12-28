import json
from pathlib import Path
from statistics import mean

from tensorflow.keras.callbacks import Callback


class ProgressLogger(Callback):

    # Flush every stats_rate samples
    stats_rate = 2000

    def __init__(self, output_dir: Path, batch_size: int):
        self._output_dir = output_dir
        self._batch_size = batch_size
        self._num_samples = 0
        self._buffer = []
        self._training_history = []
        self._validation_history = []

    def on_batch_end(self, batch, logs):
        self._num_samples += self._batch_size
        self._buffer.append(float(logs['loss']))

        if len(self._buffer) * self._batch_size > self.stats_rate:
            loss = mean(self._buffer)
            self._training_history.append((self._num_samples, loss))
            self._buffer = []
            self.flush()

    def on_epoch_end(self, epoch, logs):
        loss = float(logs['val_loss'])
        self._validation_history.append((self._num_samples, loss))
        self.flush()

    def flush(self):
        loss_dict = {
            'validation': self._validation_history,
            'training': self._training_history,
        }

        with open(self._output_dir / 'loss.json', 'w') as fp:
            json.dump(loss_dict, fp)
