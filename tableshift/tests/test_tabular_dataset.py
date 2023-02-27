"""
Tests for TabularDatasets.

To run tests: python -m unittest tableshift/tests/test_tabular_dataset.py -v
"""
import tempfile
import unittest
from tableshift.core import TabularDataset, RandomSplitter, \
    TabularDatasetConfig, PreprocessorConfig


class TestGerman(unittest.TestCase):
    def test_init_dataset(self):
        """
        Initialize the German dataset. Checks that an exception is not raised.
        """
        with tempfile.TemporaryDirectory() as td:
            _ = TabularDataset(name="german",
                               config=TabularDatasetConfig(cache_dir=td),
                               grouper=None,
                               splitter=RandomSplitter(val_size=0.25,
                                                       random_state=68594,
                                                       test_size=0.25),
                               preprocessor_config=PreprocessorConfig(
                                   passthrough_columns="all"))
        return

    def test_get_pandas(self):
        """
        Initialize the Adult dataset and check for nonempty train/test/val DFs.
        """
        with tempfile.TemporaryDirectory() as td:
            dset = TabularDataset(name="adult",
                                  config=TabularDatasetConfig(cache_dir=td),
                                  grouper=None,
                                  splitter=RandomSplitter(val_size=0.25,
                                                          random_state=68594,
                                                          test_size=0.25),
                                  preprocessor_config=PreprocessorConfig(
                                      passthrough_columns="all"))
            for split in ("train", "validation", "test"):
                X, y, _, _ = dset.get_pandas(split)
                assert len(X)
                assert len(X) == len(y)
        return
