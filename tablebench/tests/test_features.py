"""
Tests for Features and FeatureLists.

To run tests: python -m unittest tablebench/tests/test_features.py -v
"""
import unittest

import pandas as pd
import numpy as np

from tablebench.core.features import Feature, FeatureList, cat_dtype


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

    def test_fillna_int_object(self):
        """Tests case of categorical feature with int na_values."""
        letters = list("abcdefg")
        raise NotImplementedError("incomplete.")
        data = pd.Series(letters * 2).astype()
        feature = Feature("my_feature", cat_dtype, na_values=("a", "b"))
        data_na = feature.fillna(data)
        self.assertEqual(pd.isnull(data_na).sum(), 4)
