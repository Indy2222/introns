import importlib
from pathlib import Path

import numpy as np
import pandas as pd

# Needs to be imported like this due to hyphens in the module name.
module = importlib.import_module('classify-splice-sites')


def test_load_data():
    test_csv_path = Path(__file__).parent / 'test_data.csv'
    test_df = pd.read_csv(str(test_csv_path), sep=';')
    test_data = module.prepare_inputs(test_df)

    assert test_data.shape == (2, 7, 5)
    assert test_data.dtype == np.float32

    expected_data = np.array([
        [  # TGACNGG
            [0., 1., 0., 0., 0.],
            [0., 0., 0., 1., 0.],
            [1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 1.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 1., 0.],
        ],
        [  # AAAAAAA
            [1., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0.],
        ],
    ], dtype=np.float32)
    np.testing.assert_array_equal(test_data, expected_data)
