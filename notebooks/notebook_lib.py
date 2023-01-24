from dataclasses import dataclass
import glob
import os
import os.path as osp
import re
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns

from tablebench.models.compat import PYTORCH_MODEL_NAMES

group_cols = ['task', 'estimator', 'domain_split_ood_values']


@dataclass
class ExperimentInfo:
    uid: str
    task: str
    ood_vals: str


# UIDs for the benchmark tasks.
EXPERIMENTS_LIST = [
    ExperimentInfo(
        "acsfoodstampsdomain_split_varname_DIVISIONdomain_split_ood_value_06",
        'acsfoodstamps_region',
        "['06']"),
    ExperimentInfo(
        'acsincomedomain_split_varname_DIVISIONdomain_split_ood_value_01',
        'acsincome_region',
        "['01']"),
    ExperimentInfo(
        'acspubcovdomain_split_varname_DISdomain_split_ood_value_1.0',
        'acspubcov_disability',
        "['1.0']"),
    ExperimentInfo(
        'acsunemploymentdomain_split_varname_SCHLdomain_split_ood_value_010203040506070809101112131415',
        'acsunemployment_edlvl',
        "['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']"),
    ExperimentInfo(
        'anesdomain_split_varname_VCF0112domain_split_ood_value_3.0',
        'anes_region',
        "['3.0']"),
    ExperimentInfo(
        'brfss_diabetesdomain_split_varname_PRACE1domain_split_ood_value_23456domain_split_id_values_1',
        'brfss_diabetes_race',
        '[2, 3, 4, 5, 6]'),
    ExperimentInfo(
        'brfss_blood_pressuredomain_split_varname_BMI5CATdomain_split_ood_value_3.04.0',
        'brfss_blood_pressure_bmi',
        "['3.0', '4.0']"),
    ExperimentInfo(
        'diabetes_readmissiondomain_split_varname_admission_source_iddomain_split_ood_value_7',
        'diabetes_admsrc',
        '(7,)'),
    ExperimentInfo(
        'helocdomain_split_varname_ExternalRiskEstimateLowdomain_split_ood_value_0',
        'heloc_externalrisk',
        '[0]'),
    ExperimentInfo(
        'mimic_extract_los_3domain_split_varname_insurancedomain_split_ood_value_Medicare',
        'mimic_extract_los_3_ins',
        "['Medicare']", ),
    ExperimentInfo(
        'mimic_extract_mort_hospdomain_split_varname_insurancedomain_split_ood_value_MedicareMedicaid',
        'mimic_extract_hosp_mort',
        "['1', '2']"),
    ExperimentInfo(
        'nhanes_leaddomain_split_varname_INDFMPIRBelowCutoffdomain_split_ood_value_1.0',
        'nhanes_lead_poverty',
        '[1.0]'),
    ExperimentInfo(
        'physionetICULOSgt47.0',
        'physionet_los47',
        "['all_ood_test_subdomains']"),
]

UIDS_LIST = [e.uid for e in EXPERIMENTS_LIST]


def extract_name_from_uid(uid: str) -> str:
    return re.search('^(\w+)domain_split_varname.*', uid).group(1)


# various plot-related utilities
def values_to_colors(ser: pd.Series) -> Tuple[pd.Series, dict]:
    """Helper function for mapping a categorical variable to colors for matplotlib."""
    color_labels = ser.unique()
    rgb_values = sns.color_palette("colorblind", len(color_labels))
    color_map = dict(zip(color_labels, rgb_values))
    return ser.map(color_map), color_map


# a list of RGB values for deterministic categorical plotting
rgblist = list(sns.color_palette("colorblind", 12))


def best_results_by_metric(
        df,
        metric='validation_accuracy',
        group_cols=group_cols):
    df.reset_index(inplace=True, drop=True)
    vals = df.groupby(group_cols)[metric].idxmax().tolist()
    df_out = df.loc[vals]
    assert np.all(df_out.groupby(group_cols).size() == 1)
    return df_out


def read_tableshift_results(tableshift_results_dir):
    """Read the non-baseline results."""
    for expt in EXPERIMENTS_LIST:
        for model in PYTORCH_MODEL_NAMES:
            search_glob = os.path.join(tableshift_results_dir, expt, '*', f'.*{model}.csv')
            files = glob.glob(search_glob)
            most_recent = sorted(files)[0]

def read_domain_shift_results(ds_results_dir):
    """Read the baseline results (CPU models: xgb and lightgbm only)."""
    df_list = []

    for shift in os.listdir(ds_results_dir):
        if shift.startswith('.'):
            continue
        runs = os.listdir(osp.join(ds_results_dir, shift))
        most_recent_run = sorted(runs)[-1]

        fileglob = os.path.join(ds_results_dir, shift, most_recent_run,
                                '*_full.csv')
        full_results = glob.glob(fileglob)
        if full_results:
            #         print(f'reading results for shift {shift}')
            filename = full_results[0]
            df = pd.read_csv(filename)
            df['task'] = shift
            df['filename'] = filename
            df_list.append(df)
        else:
            print(f'no results for shift {shift} matching {fileglob}')
    return pd.concat(df_list).reset_index()
