Segmentation of CDS in Fungi DNA
================================

This is repository of Martin Indra's master's thesis focusing on coding
sequence segmentation in DNA sequences of fungi.

Compiled PDF of the thesis is available on
[mgn.cz/introns.pdf](https://mgn.cz/introns.pdf).

The goal of the thesis is detection of introns in long fungal DNA sequences. Id
est detection of start and end position of each intron. This is done in two
steps:

 1. Intron [splice-site detection](#splice-site-detection) predicts confidence
    of donor and acceptor splice-site on each consensus dinucleotide (`GT` and
    `AG` respecitvely) in a DNA sequence.

 1. Intron detection produces a list of intron start and end positions withing
    a sequence. This step takes output from the previous step as input.

## Thesis

The text is placed inside `/thesis`.

## Data

Data for the project as well as trained neural networks and other large binary
materials are available in [Google Cloud Storage
(GCS)](https://cloud.google.com/storage/) at
[gs://thesis.mgn.cz/data](https://console.cloud.google.com/storage/browser/thesis.mgn.cz/data/).
The GCS is not publicly accessibly because it is privately funded by the
author. Feel to request access from the author if necessary.

Part of data can be downloaded automatically with `data.sh` script.

# Splice Site Detection<a name="splice-site-detection"></a>

Splice sites are detected with a recurrent convolutional neural network (RCNN).
The RCNN takes a fixed-length DNA sequence on its input and predicts whether
there is a splice-site at a fixed position in the input sequence. Output of the
RCNN is a number between 0 and 1 (inclusive).

The RCNN is trained and experiments are done with the following pipeline:

 1. Training data are prepared with a pre-processing program placed in
    [/preprocess](/preprocess) directory. See `--help` test of the program to
    learn more about data pre-processing.

    Numpy NPZ files with training/validation/test data are generated at the end
    of the pre-processing pipeline. Each NPZ file contains arrays `input`
    (one-hot-encoded input DNA sequence), `output` (either 1.0 or 0.0) and
    `position`.

 1. This steps generates a two CSV files with file paths to training and
    validation examples, id est paths to a subset of NPZ paths from the previous
    steps. This step takes the following inputs:

    * Splice-site type, which is either `donor` or `acceptor`.
    * Positive example ratio, which is a number between 0 and 1 (exclusive).
    * Maximum number of examples per organism. Note that there might be less
      than this number of example for organisms with short DNA sequences. This
      value is set separately for training and validation dataset.
    * A filter on organisms, which could be used in the dataset.
    * Number of validation organisms. The remaining organisms are included in
      training dataset.

 1. This steps train a RCNN on input data generated in previous steps. Training
    and validation loss is stored in a log file during training and model is
    stored in a HDF5 file.

 1. This steps evaluates performance of the neural network on validation
    dataset. It also produces performance statistics on each organism from
    training and validation datasets.

 1. This steps produces a comprehensive PDF report from data generated in the
    previous step.
