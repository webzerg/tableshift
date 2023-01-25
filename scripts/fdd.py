from collections import defaultdict
from tqdm import tqdm
import argparse
import logging
from tablebench.core import TabularDataset, TabularDatasetConfig, CachedDataset
import matplotlib.pyplot as plt
from tablebench.models.training import train
from scipy.linalg import sqrtm
from numpy import iscomplexobj, trace

import numpy as np

from tablebench.configs.experiment_configs import EXPERIMENT_CONFIGS
from tablebench.models.utils import get_estimator
from tablebench.models.training import train
from tablebench.models.config import get_default_config
from tablebench.models.rtdl import MLPModelWithHook
from tablebench.models.compat import OPTIMIZER_ARGS

from tablebench.notebook_lib import EXPERIMENTS_LIST

experiment = 'anes'
uid = 'anesdomain_split_varname_VCF0112domain_split_ood_value_3.0'
cache_dir = './tmp'

LOG_LEVEL = logging.DEBUG

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=LOG_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')

expt_config = EXPERIMENT_CONFIGS[experiment]

logging.info(f"experiment is {experiment}")
logging.info(f"uid is {uid}")
logging.info(f"constructing cached dataset from {cache_dir}")

dset = CachedDataset(cache_dir=cache_dir, name=experiment, uid=uid)

# dataset_config = TabularDatasetConfig(cache_dir=cache_dir)
# tabular_dataset_kwargs = expt_config.tabular_dataset_kwargs
# if "name" not in tabular_dataset_kwargs:
#     tabular_dataset_kwargs["name"] = experiment

# dset = TabularDataset(config=dataset_config,
#                       splitter=expt_config.splitter,
#                       grouper=expt_config.grouper,
#                       preprocessor_config=expt_config.preprocessor_config,
#                       **tabular_dataset_kwargs)


model = "mlp"
config = get_default_config(model, dset)

logging.info(f"config is {config}")

kwargs = config

estimator = MLPModelWithHook(d_in=kwargs["d_in"],
                             d_layers=[kwargs["d_hidden"]] * kwargs[
                                 "num_layers"],
                             d_out=1,
                             dropouts=kwargs["dropouts"],
                             activation=kwargs["activation"],
                             **{k: kwargs[k] for k in OPTIMIZER_ARGS})

estimator = train(estimator, dset, device="cpu", config=config)

split_activations = defaultdict(list)

for split in ('id_test', 'ood_test'):
    print(split)
    loader = dset.get_dataloader(split)
    for x, y, _, _ in tqdm(loader):
        activations = estimator.get_activations(x)
        split_activations[split].append(activations.detach().cpu().numpy())

id_activations = np.row_stack(split_activations['id_test'])
ood_activations = np.row_stack(split_activations['ood_test'])


# See https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
def fdd(id_activations, ood_activations):
    Sigma_id = np.cov(id_activations, rowvar=False)
    Sigma_ood = np.cov(ood_activations, rowvar=False)

    covmean = sqrtm(Sigma_id.dot(Sigma_ood))

    if iscomplexobj(covmean):
        covmean = covmean.real

    return np.linalg.norm(
        id_activations.mean(axis=0) - ood_activations.mean(axis=0),
        ord=2) ** 2 + \
           trace(Sigma_id + Sigma_ood - 2. * covmean)


fdd_val = fdd(id_activations, ood_activations)
print(f"fdd for uid {uid} is {fdd_val}")
