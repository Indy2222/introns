#! /usr/bin/env python3

"""Model training tool.

Usage:
  train.py --data-directory <data-directory> \
    --features-directory <features-directory> \
    --json-path <json-path> \
    --output-dir <output-dir>
  train.py (-h | --help)
  train.py --version

Options:
  -h --help                  Show this screen.
  --version                  Show version.
"""

import os
from pathlib import Path

from docopt import docopt
from tensorflow import keras
from tensorflow.keras.callbacks import (EarlyStopping, LambdaCallback,
                                        ModelCheckpoint, ReduceLROnPlateau)
from tensorflow.keras.layers import (LSTM, Add, Bidirectional, Concatenate,
                                     Conv1D, Dense, Dropout, Flatten, Input,
                                     LeakyReLU, Masking)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

from data import DatasetConfig, SampleType
from loader import load_datasets
from progress import ProgressLogger


def main():
    arguments = docopt(__doc__, version='Training 1.0')
    data_directory = Path(arguments['<data-directory>'])
    features_directory = Path(arguments['<features-directory>'])
    json_path = Path(arguments['<json-path>'])
    output_dir = Path(arguments['<output-dir>'])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset_config = DatasetConfig.from_path(json_path)

    batch_size = 16

    training_dataset, validation_dataset = load_datasets(
        features_directory=features_directory,
        data_directory=data_directory,
        dataset_config=dataset_config,
        batch_size=batch_size,
    )

    callbacks = []
    callbacks.append(LambdaCallback(
        on_epoch_end=lambda epoch, logs: training_dataset.shuffle(),
    ))

    if dataset_config.sample_type == SampleType.INTRON:
        model = build_intron_model(training_dataset.window_size)
    else:
        model = build_splice_site_model(training_dataset.window_size)
    print(model.summary())

    keras.utils.plot_model(
        model,
        to_file=output_dir / 'architecture.pdf',
        show_shapes=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=300,
    )

    optimizer = SGD(
        learning_rate=0.01,
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    callbacks.append(ModelCheckpoint(
        filepath=str(output_dir / 'model.h5'),
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        period=1,
        verbose=1
    ))

    callbacks.append(ModelCheckpoint(
        filepath=str(output_dir / 'model-{epoch:03d}.h5'),
        save_best_only=False,
        monitor='val_loss',
        mode='min',
        period=1,
        verbose=1
    ))

    callbacks.append(EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=10,
    ))

    callbacks.append(ReduceLROnPlateau(
        monitor='val_loss',
        min_delta=0.001,
        factor=0.2,
        patience=1,
        min_lr=0.00001,
        verbose=1,
    ))

    callbacks.append(ProgressLogger(output_dir, batch_size))

    print('Training is starting...')
    model.fit(
        training_dataset,
        shuffle=False,
        validation_data=validation_dataset,
        epochs=1000,
        callbacks=callbacks,
    )


def build_splice_site_model(window_size: int):
    layer = input_ = Input(shape=(window_size, 5))

    layer = Concatenate(axis=2)([layer] * 12)
    for _ in range(2):
        intermediate = Conv1D(
            filters=60,
            kernel_size=4,
            strides=1,
            padding='same',
            activation='linear'
        )(layer)
        intermediate = LeakyReLU(alpha=0.01)(intermediate)

        intermediate = Conv1D(
            filters=60,
            kernel_size=1,
            strides=1,
            padding='same',
            activation='linear'
        )(intermediate)
        intermediate = LeakyReLU(alpha=0.01)(intermediate)
        layer = Add()([layer, intermediate])

    layer = Bidirectional(LSTM(10, return_sequences=True))(layer)

    layer = Flatten()(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(1, activation='sigmoid')(layer)

    return Model(inputs=[input_], outputs=[layer])


def build_intron_model(window_size):
    layer = input_ = Input(shape=(window_size, 5))
    layer = Masking(mask_value=0., input_shape=(window_size, 5))(layer)
    layer = Conv1D(
        filters=10,
        kernel_size=3,
        strides=1,
        padding='valid',
        activation='linear'
    )(layer)
    layer = LeakyReLU(alpha=0.01)(layer)
    layer = Bidirectional(LSTM(10, return_sequences=False))(layer)
    layer = Dense(1, activation='sigmoid')(layer)
    return Model(inputs=[input_], outputs=[layer])


if __name__ == '__main__':
    main()
