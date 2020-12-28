#! /usr/bin/env python3

"""Model evaluation report tool. This tool generates a report from evaluation
generated data.

Usage:
  report.py --counts <count-csv-path> --evaluation-dir <evaluation-dir> \
    --data-json-path <data-json-path> --output-dir <output-dir>
  report.py (-h | --help)
  report.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
"""

import datetime
import json
import os
from pathlib import Path
from shutil import copyfile
from subprocess import check_call

import pandas as pd
from docopt import docopt

from data import DatasetConfig, SampleType

REPORT_TEMPLATE = """\\documentclass[11pt,a4paper]{{article}}

\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}

\\begin{{document}}
\\title{{Evaluation Report}}
\\author{{Martin Indra}}
\\date{{{date:%B %d}}}
\\maketitle

\\section{{Dataset}}

{dataset_latex}

\\section{{Overall Performance}}

{section_overall_latex}

\\section{{Individual Organisms}}

{section_organisms_latex}

\\end{{document}}
"""


def main():
    arguments = docopt(__doc__, version='Report 1.0')
    count_csv_path = Path(arguments['<count-csv-path>'])
    evaluation_dir = Path(arguments['<evaluation-dir>'])
    dataset_config_path = Path(arguments['<data-json-path>'])
    output_dir = Path(arguments['<output-dir>'])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset_config = DatasetConfig.from_path(dataset_config_path)
    sample_type = dataset_config.sample_type

    counts = None
    if sample_type in (SampleType.DONOR, SampleType.ACCEPTOR):
        counts = pd.read_csv(count_csv_path)
        counts = pd.DataFrame({
            'organism_id': counts['organism_id'],
            'true': counts[f'true_{sample_type.value}'],
            'false': counts[f'false_{sample_type.value}'],
        })
        counts.set_index('organism_id', inplace=True)

    dataset_latex = generate_dataset_section(dataset_config)

    section_overall_latex = generate_overall_section(
        evaluation_dir, output_dir, dataset_config, counts)
    section_organisms_latex = generate_organisms_section(
        evaluation_dir, counts)

    latex_text = REPORT_TEMPLATE.format(
        date=datetime.datetime.utcnow().date(),
        dataset_latex=dataset_latex,
        section_overall_latex=section_overall_latex,
        section_organisms_latex=section_organisms_latex,
    )

    with open(output_dir / 'report.tex', 'w') as fp:
        fp.write(latex_text)

    check_call(['pdflatex', 'report.tex'], cwd=output_dir)
    check_call(['pdflatex', 'report.tex'], cwd=output_dir)


def generate_dataset_section(dataset_config: DatasetConfig) -> str:
    number_of_organisms = len(dataset_config.training_organisms)
    per_organism_limit = dataset_config.training_per_organism_limit
    positive_examples_ratio = dataset_config.positive_examples_ratio

    # TODO: window size
    # TODO: window offset

    organism_ids_latex = [
        o.replace('_', '\\_') for o in dataset_config.training_organisms
    ]

    return (
        '\\begin{itemize}\n'
        f'\\item Sample type: {dataset_config.sample_type.value}\n'
        f'\\item Number of training organisms: {number_of_organisms}\n'
        f'\\item Max samples per organism: {per_organism_limit}\n'
        f'\\item Positive examples: {positive_examples_ratio * 100:0.1f}\\%\n'
        '\\end{itemize}\n\n'
        '\\subsection{{Training Organism IDs}}\n\n'
        f'{", ".join(organism_ids_latex)}\n\n'
    )


def generate_overall_section(
    evaluation_dir: Path,
    output_dir: Path,
    dataset_config: DatasetConfig,
    counts: pd.DataFrame,
) -> str:
    training_loss_path = _copy_figure(
        evaluation_dir, output_dir, 'training-loss.pdf')

    evaluation_dir = evaluation_dir / 'all' / 'all_organisms'

    roc_curve_path = _copy_figure(evaluation_dir, output_dir, 'roc-curve.pdf')
    confusion_matrix_path = _copy_figure(
        evaluation_dir, output_dir, 'confusion-matrix.pdf')
    prediction_hist_path = _copy_figure(
        evaluation_dir, output_dir, 'prediction-hist.pdf')
    prediction_cdf_path = _copy_figure(
        evaluation_dir, output_dir, 'prediction-cdf.pdf')

    with open(evaluation_dir / 'auc.json') as fp:
        auc = json.load(fp)
    assert isinstance(auc, float)

    with open(evaluation_dir / 'confusion-matrix.json') as fp:
        confusion_matrix = json.load(fp)
    tp = confusion_matrix['tp']
    tn = confusion_matrix['tn']
    fp = confusion_matrix['fp']
    fn = confusion_matrix['fn']

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    precision_corrected = None
    if counts is not None:
        organism_ids = dataset_config.validation_organisms
        counts = counts[counts.index.isin(organism_ids)]
        tp_corrected = tp * counts['true'].sum()
        fp_corrected = fp * counts['false'].sum()
        precision_corrected = tp_corrected / (tp_corrected + fp_corrected)

    overall_latex = (
        '\\begin{itemize}\n'
        f'\\item Accuracy on balanced dataset {accuracy * 100:0.1f}\\%\n'
        f'\\item AUC {auc * 100:0.1f}\\%\n'
        f'\\item Precision on balanced dataset {precision * 100:0.1f}\\%\n'
        f'\\item Recall {recall * 100:0.1f}\\%\n'
    )

    if precision_corrected is not None:
        overall_latex += f'\\item Precision Corrected {precision_corrected * 100:0.1f}\\%\n'

    overall_latex += (
        '\\end{itemize}\n'
        '\n'
        '\\begin{figure}[!ht]\n'
        '\\centering\n'
        f'\\includegraphics[width=0.7\\textwidth]{{{training_loss_path}}}\n'
        '\\end{figure}\n'
        '\n'
        '\\begin{figure}[!ht]\n'
        '\\centering\n'
        f'\\includegraphics[width=0.7\\textwidth]{{{roc_curve_path}}}\n'
        '\\end{figure}\n'
        '\n'
        '\\begin{figure}[!ht]\n'
        '\\centering\n'
        f'\\includegraphics[width=0.7\\textwidth]{{{confusion_matrix_path}}}\n'
        '\\end{figure}\n'
        '\n'
        '\\begin{figure}[!ht]\n'
        '\\centering\n'
        f'\\includegraphics[width=0.7\\textwidth]{{{prediction_hist_path}}}\n'
        '\\end{figure}\n'
        '\n'
        '\\begin{figure}[!ht]\n'
        '\\centering\n'
        f'\\includegraphics[width=0.7\\textwidth]{{{prediction_cdf_path}}}\n'
        '\\end{figure}\n'
        '\\clearpage\n'
    )

    return overall_latex


def _copy_figure(source_dir: Path, output_dir: Path, file_name: str):
    figures_dir = output_dir / 'figures'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    copyfile(source_dir / file_name, figures_dir / file_name)
    return f'figures/{file_name}'


def generate_organisms_section(evaluation_dir: Path, counts: pd.DataFrame):
    evaluation_dir = evaluation_dir / 'per_organism'

    rows = []
    for organism_id in os.listdir(evaluation_dir):
        organism_dir = evaluation_dir / organism_id

        with open(organism_dir / 'auc.json') as fp:
            auc = json.load(fp)
        assert isinstance(auc, float)

        with open(organism_dir / 'confusion-matrix.json') as fp:
            confusion_matrix = json.load(fp)
        tp = confusion_matrix['tp']
        fp = confusion_matrix['fp']
        fn = confusion_matrix['fn']

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        row = [organism_id, auc, precision, recall]

        if counts is not None:
            tp_corrected = counts.loc[organism_id, 'true'] * tp
            fp_corrected = counts.loc[organism_id, 'false'] * fp
            precision_corrected = tp_corrected / (tp_corrected + fp_corrected)
            row.append(precision_corrected)

        # TODO: add more stats, add Phylum, class etc (from DB)
        rows.append(row)

    columns = [
        'Organism ID',
        'AUC',
        'Precision',
        'Recall',
    ]

    if counts is not None:
        columns.append('Precision Corr.')

    df = pd.DataFrame.from_records(
        data=rows,
        index='Organism ID',
        columns=columns,
    )
    df.sort_values(by='Organism ID', inplace=True)

    table_latex = df.to_latex(
        longtable=True,
        formatters={
            'AUC': lambda v: f'{v * 100:0.1f}%',
            'Precision': lambda v: f'{v * 100:0.1f}%',
            'Recall': lambda v: f'{v * 100:0.1f}%',
            'Precision Corr.': lambda v: f'{v * 100:0.1f}%',
        }
    )

    text = (
        'The following table reports model performance on individual '
        'organisms. Precision and recall are calculated on a perfectly '
        'balanced dataset.'
    )

    return text + '\n\n' + table_latex


if __name__ == '__main__':
    main()
