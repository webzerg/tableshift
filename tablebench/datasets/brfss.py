"""

Utilities for working with BRFSS dataset.

Accessed via https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system.
Raw Data: https://www.cdc.gov/brfss/annual_data/annual_data.htm
Data Dictionary: https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf
"""

import re

import pandas as pd

from tablebench.core.features import Feature, FeatureList, cat_dtype

BRFSS_STATE_LIST = [
    ['1.0', '10.0', '11.0', '12.0', '13.0', '15.0', '16.0', '17.0', '18.0',
     '19.0', '2.0', '20.0', '21.0', '22.0', '23.0', '24.0', '25.0', '26.0',
     '27.0', '28.0', '29.0', '30.0', '31.0', '32.0', '33.0', '34.0', '35.0',
     '36.0', '37.0', '38.0', '39.0', '4.0', '40.0', '41.0', '42.0', '44.0',
     '45.0', '46.0', '47.0', '48.0', '49.0', '5.0', '50.0', '51.0', '53.0',
     '54.0', '55.0', '56.0', '6.0', '66.0', '72.0', '8.0', '9.0']
]

# Brief feature descriptions below; for the full question/description
# see the data dictionary linked above. Note that in the original data,
# some of the feature names are preceded by underscores (these are
# "calculated variables"; see data dictionary). These leading
# underscores, where present, are removed in the preprocess_brfss() function
# due to limitations on naming in the sklearn transformers module.

BRFSS_FEATURES = FeatureList([
    ################ Target ################
    Feature("DIABETE3", int, is_target=True),  # (Ever told) you have diabetes

    # ################ Demographics/sensitive attributes. ################
    # Also see "INCOME2", "MARITAL", "EDUCA" features below.
    Feature("STATE", cat_dtype),
    # Was there a time in the past 12 months when you needed to see a doctor
    # but could not because of cost?
    Feature("MEDCOST", cat_dtype),
    # Preferred race category; note that ==1 is equivalent to
    # "White non-Hispanic race group" variable _RACEG21
    Feature("PRACE1", cat_dtype),
    # Indicate sex of respondent.
    Feature("SEX", cat_dtype),

    # Below are a set of indicators for known risk factors for diabetes.

    ################ General health ################
    # for how many days during the past 30 days was your
    # physical health not good?
    Feature("PHYSHLTH", float),
    ################ High blood pressure ################
    # Adults who have been told they have high blood pressure by a
    # doctor, nurse, or other health professional
    Feature("RFHYPE5", cat_dtype),
    ################ High cholesterol ################
    # Cholesterol check within past five years
    Feature("CHOLCHK", cat_dtype),
    # Have you EVER been told by a doctor, nurse or other health
    # professional that your blood cholesterol is high?
    Feature("TOLDHI2", cat_dtype),
    ################ BMI/Obesity ################
    # Calculated Body Mass Index (BMI)
    Feature("BMI5", float),
    # Four-categories of Body Mass Index (BMI)
    Feature("BMI5CAT", cat_dtype),
    ################ Smoking ################
    # Have you smoked at least 100 cigarettes in your entire life?
    Feature("SMOKE100", cat_dtype),
    # Do you now smoke cigarettes every day, some days, or not at all?
    Feature("SMOKDAY2", cat_dtype),
    ################ Other chronic health conditions ################
    # (Ever told) you had a stroke.
    Feature("CVDSTRK3", cat_dtype),
    # ever reported having coronary heart disease (CHD)
    # or myocardial infarction (MI)
    Feature("MICHD", cat_dtype),
    ################ Diet ################
    # Consume Fruit 1 or more times per day
    Feature("FRTLT1", cat_dtype),
    # Consume Vegetables 1 or more times per day
    Feature("VEGLT1", cat_dtype),
    ################ Alcohol Consumption ################
    # Calculated total number of alcoholic beverages consumed per week
    Feature("DRNKWEK", float),
    # Binge drinkers (males having five or more drinks on one occasion,
    # females having four or more drinks on one occasion)
    Feature("RFBING5", cat_dtype),
    ################ Exercise ################
    # Adults who reported doing physical activity or exercise
    # during the past 30 days other than their regular job
    Feature("TOTINDA", cat_dtype),
    # Minutes of total Physical Activity per week
    Feature("PA1MIN_", float),
    ################ Household income ################
    # annual household income from all sources
    Feature("INCOME2", cat_dtype),
    ################ Marital status ################
    Feature("MARITAL", cat_dtype),
    ################ Time since last checkup
    # About how long has it been since you last visited a
    # doctor for a routine checkup?
    Feature("CHECKUP1", cat_dtype),
    ################ Education ################
    # highest grade or year of school completed
    Feature("EDUCA", cat_dtype),
    ################ Health care coverage ################
    # Respondents aged 18-64 who have any form of health care coverage
    Feature("HCVU651", cat_dtype),
    ################ Mental health ################
    # for how many days during the past 30
    # days was your mental health not good?
    Feature("MENTHLTH", float),
])

# Raw names of the input features. Useful to subset before preprocessing,
# since some features contain near-duplicate versions (i.e. calculated
# and not-calculated versions, differing only be precending underscore).
BRFSS_INPUT_FEATURES = [
    "DIABETE3", "_STATE", "MEDCOST", "_HCVU651", "_PRACE1", "SEX",
    "PHYSHLTH", "_RFHYPE5", "_CHOLCHK", "TOLDHI2", "_BMI5", "_BMI5CAT",
    "SMOKE100", "SMOKDAY2", "CVDSTRK3", "_MICHD", "_FRTLT1", "_VEGLT1",
    "_DRNKWEK", "_RFBING5", "_TOTINDA", "PA1MIN_", "INCOME2", "MARITAL",
    "CHECKUP1", "EDUCA", "_HCVU651", "MENTHLTH"]


def preprocess_brfss(df: pd.DataFrame):
    # Label
    df["DIABETE3"].replace({2: 0, 3: 0, 4: 0}, inplace=True)
    # Drop 1k missing/not sure, plus one missing observation
    df = df[~(df["DIABETE3"].isin([7, 9]))].dropna(subset=["DIABETE3"])

    # Sensitive columns
    # Drop no preferred race/not answered/don't know/not sure
    df = df[~(df["_PRACE1"].isin([7, 8, 77, 99]))]
    df["_PRACE1"] = (df["_PRACE1"] == 1).astype(int)
    df["SEX"] = (df["SEX"] - 1)  # Map sex to male=0, female=1

    # PHYSHLTH, POORHLTH, MENTHLTH are measured in days, but need to
    # map 88 to 0 because it means zero (i.e. zero bad health days)
    df["PHYSHLTH"] = df["PHYSHLTH"].replace({88: 0})
    df["MENTHLTH"] = df["MENTHLTH"].replace({88: 0})

    # Drop rows where drinks per week is unknown/refused/missing;
    # this uses a different missingness code from other variables.
    df = df[~(df["_DRNKWEK"] == 99900)]

    # Some questions are not asked for various reasons
    # (see notes under "BLANK" for that question in data dictionary);
    # create an indicator for these due to large fraction of missingness.
    df["SMOKDAY2"] = df["SMOKDAY2"].fillna("NOTASKED_MISSING").astype(str)
    df["TOLDHI2"] = df["TOLDHI2"].fillna("NOTASKED_MISSING").astype(str)

    NUMERIC_COLS = ("_BMI5", "_DRNKWEK", "PHYSHLTH", "MENTHLTH", "PA1MIN_")

    # For these categorical columns, drop respondents who were not sure,
    # refused, or had missing responses. This is also useful because
    # sometimes those responses (dk/refuse/missing) are lumped into
    # a single category (e.g. "_TOTINDA").
    DROP_MISSING_REFUSED_COLS = (
        "MEDCOST", "PHYSHLTH", "_RFHYPE5", "_CHOLCHK", "SMOKE100",
        "SMOKDAY2", "TOLDHI2", "CVDSTRK3", "_TOTINDA", "_FRTLT1",
        "_VEGLT1", "_RFBING5", "PA1MIN_", "INCOME2", "MARITAL", "CHECKUP1",
        "EDUCA", "_MICHD", "_BMI5", "_BMI5CAT")

    for c in DROP_MISSING_REFUSED_COLS:
        if c not in NUMERIC_COLS:
            # Apply coded values for missing/refused/idk, for categorical cols.
            # Note that 88 is sometimes used for for these, but 8 is NOT
            # and constitutes a valid value in the above columns.
            df = df[~(df[c].isin([7, 9, 77, 88, 99]))]
        # Drop actual missing values, for all column dtypes
        df.dropna(subset=[c], inplace=True)
        # print("filtered {} remaining rows ({:.4f}%) using column {}".format(
        #     start_sz - len(df), 100 * (start_sz - len(df))/start_sz, c))

    # Cast columns to categorical; since some columns have mixed type,
    # we cast the entire column to string.
    for c in df.columns:
        if c not in NUMERIC_COLS and c not in (
                "_PRACE_1", "SEX", BRFSS_FEATURES.target):
            df[c] = df[c].apply(str).astype("category")

    # Remove leading underscores from column names
    renames = {c: re.sub("^_", "", c) for c in df.columns if c.startswith("_")}
    df.rename(columns=renames, inplace=True)

    # Select features and reset the index after subsampling;
    # resetting ensures that splitting happens correctly.
    df = df.loc[:, BRFSS_FEATURES.names].reset_index(drop=True)
    return df
