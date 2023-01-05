import pytest
import pandas as pd
import numpy as np

from tablebench.core.splitter import DomainSplitter

np.random.seed(54329)


class TestDomainSplitter:

    def test_disjoint_splits(self):
        ood_vals = ["a"]
        splitter = DomainSplitter(id_test_size=0.5,
                                  val_size=0.1,
                                  domain_split_varname="domain",
                                  domain_split_ood_values=ood_vals,
                                  ood_val_size=0.25,
                                  random_state=45378)
        n = 5000
        data = pd.DataFrame({
            "values1": np.arange(n),
            "values2": np.random.choice([1, 2, 3], n),
            "domain": (["a"] * int(n / 2) + ["b"] * int(n / 2))
        })

        groups = pd.DataFrame({
            "group_var_a": np.random.choice([0, 1], n),
            "group_var_b": np.random.choice([0, 1], n),
        })

        labels = pd.Series(np.random.choice([0, 1], n))

        splits = splitter(data, labels, groups=groups,
                          domain_labels=data["domain"])

        train_domains = data.iloc[splits["train"]]["domain"]
        val_domains = data.iloc[splits["validation"]]["domain"]
        ood_val_domains = data.iloc[splits["ood_validation"]]["domain"]
        id_test_domains = data.iloc[splits["id_test"]]["domain"]
        ood_test_domains = data.iloc[splits["ood_test"]]["domain"]

        # Check that OOD splits only contain OOD values
        assert np.all(ood_test_domains.isin(ood_vals))
        assert np.all(ood_val_domains.isin(ood_vals))

        # Check that ID splits do not contain any OOD values
        assert not np.any(train_domains.isin(ood_vals))
        assert not np.any(val_domains.isin(ood_vals))
        assert not np.any(id_test_domains.isin(ood_vals))

        # Check for proper partitioning of id/ood
        assert set(train_domains) == set(id_test_domains)
        assert not set(train_domains).intersection(set(ood_test_domains))
        assert not set(val_domains).intersection(set(ood_val_domains))
        assert not set(id_test_domains).intersection(set(ood_test_domains))

        # Check that output size is same as input
        assert sum(len(x) for x in splits.values()) == len(data)

        # Check that every index is somewhere in splits
        all_idxs = set(idx for split_idxs in splits.values()
                       for idx in split_idxs)
        assert all_idxs == set(data.index.tolist())