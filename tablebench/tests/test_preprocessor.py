"""
Tests for Preprocessor objects.

To run tests: python -m unittest tablebench/tests/test_preprocessor.py -v
"""

import copy
import string
import unittest
import numpy as np
import pandas as pd
from tablebench.core.features import Preprocessor, PreprocessorConfig, \
    FeatureList, Feature


class TestPreprocessor(unittest.TestCase):
    def setUp(self) -> None:
        n = 100
        self.df = pd.DataFrame({
            "int_a": np.arange(0, n, dtype=int),
            "int_b": np.arange(-n, 0, dtype=int),
            "float_a": np.random.uniform(size=n),
            "float_b": np.random.uniform(-1., 1., size=n),
            "string_a": np.random.choice(["a", "b", "c"], size=n),
            "cat_a": pd.Categorical(
                np.random.choice(["typea", "typeb"], size=n)),
        })
        return

    def test_passthrough_all(self):
        """Test case with no transformations (passthrough="all")."""
        data = copy.deepcopy(self.df)
        preprocessor = Preprocessor(
            config=PreprocessorConfig(passthrough_columns="all"))
        train_idxs = list(range(50))
        transformed = preprocessor.fit_transform(data, train_idxs=train_idxs)
        np.testing.assert_array_equal(data.values, transformed.values)
        return

    def test_passthrough_numeric(self):
        """Test case with no numeric transformations."""
        data = copy.deepcopy(self.df)
        preprocessor = Preprocessor(
            config=PreprocessorConfig(numeric_features="passthrough"))
        train_idxs = list(range(50))
        transformed = preprocessor.fit_transform(data, train_idxs=train_idxs)
        numeric_cols = ["int_a", "int_b", "float_a", "float_b"]
        np.testing.assert_array_equal(data[numeric_cols].values,
                                      transformed[numeric_cols].values)
        return

    def test_passthrough_categorical(self):
        """Test case with no categorical transformations."""
        data = copy.deepcopy(self.df)
        preprocessor = Preprocessor(
            config=PreprocessorConfig(categorical_features="passthrough"))
        train_idxs = list(range(50))
        transformed = preprocessor.fit_transform(data, train_idxs=train_idxs)
        categorical_cols = ["string_a", "cat_a"]
        np.testing.assert_array_equal(data[categorical_cols].values,
                                      transformed[categorical_cols].values)
        return

    def test_map_values(self):
        """Test case where a value map is used for features."""
        value_map_string_a = {
            "a": "Diagnosis of disease A",
            "b": "Diagnosis of disease B",
            "c": "Diagnosis of disease C"}
        value_map_int_a = {x: -x for x in range(len(self.df))}
        feature_list = FeatureList([
            Feature("int_a", int, value_mapping=value_map_int_a),
            Feature("int_b", int),
            Feature("float_a", float),
            Feature("float_b", float),
            Feature("string_a", str, value_mapping=value_map_string_a)
        ])
        data = copy.deepcopy(self.df)
        preprocessor = Preprocessor(
            config=PreprocessorConfig(
                categorical_features="map_values",
                numeric_features="map_values"),
            feature_list=feature_list)
        train_idxs = list(range(50))
        transformed = preprocessor.fit_transform(data, train_idxs=train_idxs)

        self.assertTrue(np.all(np.isin(transformed["string_a"].values,
                                       list(value_map_string_a.values()))))

        self.assertTrue(np.all(np.isin(transformed["int_a"].values,
                                       list(value_map_int_a.values()))))
