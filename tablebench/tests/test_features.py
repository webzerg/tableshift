"""
Tests for Feature objects.

To run tests: python -m unittest tablebench/tests/test_features.py -v
"""
import copy
import string
import unittest

import pandas as pd
import numpy as np

from tablebench.core import features
from tablebench.core.features import Feature, cat_dtype, PreprocessorConfig


class TestFeatureFillNA(unittest.TestCase):
    def test_fillna_int(self):
        data = pd.Series(np.arange(10), dtype=int)
        feature = Feature("my_feature", int, na_values=(3, 4, 5))
        data_na = feature.fillna(data)
        self.assertEqual(pd.isnull(data_na).sum(), 3)

    def test_fillna_float(self):
        data = pd.Series(np.arange(10), dtype=float)
        feature = Feature("my_feature", float, na_values=(2., 7.))
        data_na = feature.fillna(data)
        self.assertEqual(pd.isnull(data_na).sum(), 2)

    def test_fillna_int_with_float(self):
        """Tests case of int feature with float na_values."""
        data = pd.Series(np.arange(10), dtype=int)
        feature = Feature("my_feature", int, na_values=(3., 4., 5.))
        data_na = feature.fillna(data)
        self.assertEqual(pd.isnull(data_na).sum(), 3)

    def test_fillna_float_with_int(self):
        """Tests case of float feature with int na_values."""
        data = pd.Series(np.arange(10), dtype=float)
        feature = Feature("my_feature", float, na_values=(2, 4, 6, 8))
        data_na = feature.fillna(data)
        self.assertEqual(pd.isnull(data_na).sum(), 4)

    def test_fillna_categorical(self):
        """Tests case of categorical feature with string na_values."""
        letters = list("abcdefg")
        data = pd.Categorical(letters * 2, categories=letters)
        feature = Feature("my_feature", cat_dtype, na_values=("a", "b"))
        data_na = feature.fillna(data)
        self.assertEqual(pd.isnull(data_na).sum(), 4)

    def test_fillna_int_categorical(self):
        """Tests case of categorical feature with int na_values."""
        numbers = list(range(5))
        data = pd.Categorical(numbers * 2, categories=numbers)
        feature = Feature("my_feature", cat_dtype, na_values=(2, 4))
        data_na = feature.fillna(data)
        self.assertEqual(pd.isnull(data_na).sum(), 4)


class TestColumnIsOfType(unittest.TestCase):
    def test_categorical_type_check(self):
        letters = list("abcdefg")
        numbers = list(range(5))
        numeric_cat_data = pd.Series(
            pd.Categorical(numbers * 2, categories=numbers))

        string_cat_data = pd.Series(
            pd.Categorical(letters * 2, categories=letters))

        self.assertTrue(features.column_is_of_type(string_cat_data, cat_dtype))
        self.assertTrue(features.column_is_of_type(numeric_cat_data, cat_dtype))
        self.assertFalse(features.column_is_of_type(numeric_cat_data, int))
        self.assertFalse(features.column_is_of_type(numeric_cat_data, float))

    def test_numeric_type_check(self):
        int_data = pd.Series(np.arange(10, dtype=int))
        self.assertTrue(features.column_is_of_type(int_data, int))
        self.assertFalse(features.column_is_of_type(int_data, float))
        float_data = pd.Series(np.arange(10, dtype=float))
        self.assertTrue(features.column_is_of_type(float_data, float))
        self.assertFalse(features.column_is_of_type(float_data, int))


class TestPreprocessor(unittest.TestCase):
    def setUp(self) -> None:
        n = 100
        self.df = pd.DataFrame({
            "int_a": np.arange(0, n, dtype=int),
            "int_b": np.arange(-n, 0, dtype=int),
            "float_a": np.random.uniform(size=n),
            "float_b": np.random.uniform(-1., 1., size=n),
            "string_a": np.random.choice(list(string.ascii_lowercase), size=n),
            "cat_a": pd.Categorical(
                np.random.choice(["typea", "typeb"], size=n)),
        })
        return

    def test_passthrough_all(self):
        """Test case with no transformations."""
        data = copy.deepcopy(self.df)
        preprocessor = PreprocessorConfig(passthrough_columns="all")
        train_idxs = list(range(50))
        transformed = preprocessor.fit_transform(data, train_idxs=train_idxs)
        np.testing.assert_array_equal(data.values, transformed.values)
        return
