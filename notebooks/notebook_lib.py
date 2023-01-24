import glob
import os
import os.path as osp
import re
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns

group_cols = ['task', 'estimator', 'domain_split_ood_values']

# UIDs for the benchmark tasks.
UIDS_LIST = [
    "acsfoodstampsdomain_split_varname_DIVISIONdomain_split_ood_value_06",
    'acsincomedomain_split_varname_DIVISIONdomain_split_ood_value_01',
    'acspubcovdomain_split_varname_DISdomain_split_ood_value_1.0',
    'acsunemploymentdomain_split_varname_SCHLdomain_split_ood_value_010203040506070809101112131415',
    'anesdomain_split_varname_VCF0112domain_split_ood_value_3.0',
    'brfss_diabetesdomain_split_varname_PRACE1domain_split_ood_value_23456domain_split_id_values_1',
    'brfss_blood_pressuredomain_split_varname_BMI5CATdomain_split_ood_value_3.04.0',
    'diabetes_readmissiondomain_split_varname_admission_source_iddomain_split_ood_value_7',
    'helocdomain_split_varname_ExternalRiskEstimateLowdomain_split_ood_value_0',
    'mimic_extract_los_3domain_split_varname_insurancedomain_split_ood_value_Medicare',
    'mimic_extract_mort_hospdomain_split_varname_insurancedomain_split_ood_value_MedicareMedicaid',
    'nhanes_leaddomain_split_varname_INDFMPIRBelowCutoffdomain_split_ood_value_1.0',
    'physionetICULOSgt47.0'
]


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


def read_domain_shift_results(ds_results_dir):
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
