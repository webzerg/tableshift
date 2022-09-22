from tablebench.core import FixedSplitter, Grouper, TabularDataset, \
    TabularDatasetConfig, PreprocessorConfig

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

cache_dir = "./tmp"

dataset_config = TabularDatasetConfig(
    cache_dir=cache_dir,
    download=True,
    random_seed=12479)

preprocessor_config = PreprocessorConfig()

splitter = FixedSplitter(val_size=0.25, random_state=29746)
grouper = Grouper({"Race": ["White", ], "Sex": ["Male", ]}, drop=False)
adult = TabularDataset("adult",
                       config=dataset_config,
                       splitter=splitter,
                       grouper=grouper,
                       preprocessor_config=preprocessor_config)

X_tr, y_tr, G_tr = adult.get_pandas(split="train")

estimator = HistGradientBoostingClassifier()
estimator.fit(X_tr, y_tr)

X_te, y_te, G_te = adult.get_pandas(split="test")

y_hat_te = estimator.predict(X_te)
test_accuracy = accuracy_score(y_te, y_hat_te)
print(f"test accuracy is: {test_accuracy:.3f}")