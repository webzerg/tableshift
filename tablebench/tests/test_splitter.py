"""
Tests for splitters.

To run tests: python -m unittest tablebench/tests/test_splitter.py -v
"""
import unittest
import pandas as pd
import numpy as np

from tablebench.core.splitter import DomainSplitter

np.random.seed(54329)


class TestDomainSplitter(unittest.TestCase):

    def setUp(self) -> None:
        n = 5000
        self.data = pd.DataFrame({
            "values1": np.arange(n),
            "values2": np.random.choice([1, 2, 3], n),
            "domain": (["a"] * int(n / 2) + ["b"] * int(n / 2))
        })

        self.groups = pd.DataFrame({
            "group_var_a": np.random.choice([0, 1], n),
            "group_var_b": np.random.choice([0, 1], n),
        })

        self.labels = pd.Series(np.random.choice([0, 1], n))

    def test_disjoint_splits(self):
        data = self.data
        groups = self.groups
        labels = self.labels
        ood_vals = ["a"]
        splitter = DomainSplitter(id_test_size=0.5,
                                  val_size=0.1,
                                  domain_split_varname="domain",
                                  domain_split_ood_values=ood_vals,
                                  ood_val_size=0.25,
                                  random_state=45378)

        splits = splitter(data, labels, groups=groups,
                          domain_labels=data["domain"])

        train_domains = data.iloc[splits["train"]]["domain"]
        val_domains = data.iloc[splits["validation"]]["domain"]
        ood_val_domains = data.iloc[splits["ood_validation"]]["domain"]
        id_test_domains = data.iloc[splits["id_test"]]["domain"]
        ood_test_domains = data.iloc[splits["ood_test"]]["domain"]

        # Check that OOD splits only contain OOD values
        self.assertTrue(np.all(ood_test_domains.isin(ood_vals)))
        self.assertTrue(np.all(ood_val_domains.isin(ood_vals)))

        # Check that ID splits do not contain any OOD values
        self.assertFalse(np.any(train_domains.isin(ood_vals)))
        self.assertFalse(np.any(val_domains.isin(ood_vals)))
        self.assertFalse(np.any(id_test_domains.isin(ood_vals)))

        # Check for proper partitioning of id/ood
        assert set(train_domains) == set(id_test_domains)
        self.assertTrue(set(train_domains).isdisjoint(set(ood_test_domains)))
        self.assertTrue(set(val_domains).isdisjoint(set(ood_val_domains)))
        self.assertTrue(set(id_test_domains).isdisjoint(set(ood_test_domains)))

        # Check that output size is same as input
        self.assertEqual(sum(len(x) for x in splits.values()), len(data))

        # Check that every index is somewhere in splits
        all_idxs = set(idx for split_idxs in splits.values()
                       for idx in split_idxs)
        self.assertEqual(all_idxs, set(data.index.tolist()))

    def test_no_grouper(self):
        data = self.data
        labels = self.labels
        ood_vals = ["a"]
        splitter = DomainSplitter(id_test_size=0.5,
                                  val_size=0.1,
                                  domain_split_varname="domain",
                                  domain_split_ood_values=ood_vals,
                                  ood_val_size=0.25,
                                  random_state=45378)

        splits = splitter(data, labels, groups=None,
                          domain_labels=data["domain"])

        # Check that output size is same as input
        self.assertEqual(sum(len(x) for x in splits.values()), len(data))

        # Check that every index is somewhere in splits
        all_idxs = set(idx for split_idxs in splits.values()
                       for idx in split_idxs)
        self.assertEqual(all_idxs, set(data.index.tolist()))
