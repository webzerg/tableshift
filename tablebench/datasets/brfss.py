"""

Utilities for working with BRFSS dataset.

Accessed via https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system.
Raw Data: https://www.cdc.gov/brfss/annual_data/annual_data.htm
Data Dictionary: https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf
"""

import re

import pandas as pd

from tablebench.core.features import Feature, FeatureList, cat_dtype

BRFSS_YEARS = (2011, 2012, 2013, 2014, 2015,)

BRFSS_STATE_LIST = [
    '1.0', '10.0', '11.0', '12.0', '13.0', '15.0', '16.0', '17.0', '18.0',
    '19.0', '2.0', '20.0', '21.0', '22.0', '23.0', '24.0', '25.0', '26.0',
    '27.0', '28.0', '29.0', '30.0', '31.0', '32.0', '33.0', '34.0', '35.0',
    '36.0', '37.0', '38.0', '39.0', '4.0', '40.0', '41.0', '42.0', '44.0',
    '45.0', '46.0', '47.0', '48.0', '49.0', '5.0', '50.0', '51.0', '53.0',
    '54.0', '55.0', '56.0', '6.0', '66.0', '72.0', '8.0', '9.0'
]

# Brief feature descriptions below; for the full question/description
# see the data dictionary linked above. Note that in the original data,
# some of the feature names are preceded by underscores (these are
# "calculated variables"; see data dictionary). These leading
# underscores, where present, are removed in the preprocess_brfss() function
# due to limitations on naming in the sklearn transformers module.

BRFSS_FEATURES = FeatureList([
    ################ Target ################
    Feature("DIABETES", int, is_target=True),  # (Ever told) you have diabetes

    # Derived feature for year; keep as categorical dtype so normalization
    # is not applied.
    Feature("IYEAR", int, "Year of BRFSS dataset."),
    # ################ Demographics/sensitive attributes. ################
    # Also see "INCOME2", "MARITAL", "EDUCA" features below.
    Feature("STATE", cat_dtype),
    # Was there a time in the past 12 months when you needed to see a doctor
    # but could not because of cost?
    Feature("MEDCOST", cat_dtype),
    # Preferred race category; note that ==1 is equivalent to
    # "White non-Hispanic race group" variable _RACEG21
    Feature("PRACE1", int),
    # Indicate sex of respondent.
    Feature("SEX", int),

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
    Feature("CHOL_CHK_PAST_5_YEARS", cat_dtype),
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
    Feature("VEG_ONCE_PER_DAY", cat_dtype),
    ################ Alcohol Consumption ################
    # Calculated total number of alcoholic beverages consumed per week
    Feature("DRNK_PER_WEEK", float),
    # Binge drinkers (males having five or more drinks on one occasion,
    # females having four or more drinks on one occasion)
    Feature("RFBING5", cat_dtype),
    ################ Exercise ################
    # Adults who reported doing physical activity or exercise
    # during the past 30 days other than their regular job
    Feature("TOTINDA", cat_dtype),
    ################ Household income ################
    # annual household income from all sources
    Feature("INCOME", cat_dtype),
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

def align_brfss_features(df):
    """Map BRFSS column names to a consistent format over years.

    Some questions are asked over years, but while the options are identical,
    the value labels change (specifically, the interviewing instructions change,
    e.g. "Refused—Go to Section 06.01 CHOLCHK3" for BPHIGH6 in 2021 survey
    https://www.cdc.gov/brfss/annual_data/2021/pdf/codebook21_llcp-v2-508.pdf
    vs. "Refused—Go to Section 05.01 CHOLCHK1" in 2017 survey
    https://www.cdc.gov/brfss/annual_data/2017/pdf/codebook17_llcp-v2-508.pdf
    despite identical questions, and values.

    This function addresses these different names by mapping a set of possible
    variable names for the same question, over survey years, to a single shared
    name.
    """
    mapping = {
        # Question: Consume Fruit 1 or more times per day
        "FRUIT_ONCE_PER_DAY": (
            "_FRTLT1",  # 2013, 2015
            "_FRTLT1A",  # 2017, 2019, 2021
        ),
        # Question: Consume Vegetables 1 or more times per day
        "VEG_ONCE_PER_DAY": (
            "_VEGLT1",  # 2013, 2015
            "_VEGLT1A",  # 2017, 2019, 2021
        ),
        # Question: Cholesterol check within past five years (calculated)
        "CHOL_CHK_PAST_5_YEARS": (
            "_CHOLCHK",  # 2013, 2015
            "_CHOLCH1",  # 2017
            "_CHOLCH2",  # 2019
            "_CHOLCH3",  # 2021
        ),
        # Question: (Ever told) you have diabetes (If ´Yes´ and respondent is
        # female, ask ´Was this only when you were pregnant?´. If Respondent
        # says pre-diabetes or borderline diabetes, use response code 4.)
        "DIABETES": (
            "DIABETE3",  # 2013, 2015, 2017
            "DIABETE4",  # 2019, 2021
        ),
        # Question:  Calculated total number of alcoholic beverages consumed
        # per week
        "DRNK_PER_WEEK": (
            "_DRNKWEK",  # 2015, 2017
            "_DRNKWK1",  # 2019, 2021
        ),
        # Question: Indicate sex of respondent.
        "SEX": (
            "SEX",  # 2015, 2017
            "SEXVAR",  # 2019
        ),
        # Question: Was there a time in the past 12 months when you needed to
        # see a doctor but could not because {2015-2019: of cost/ 2021: you
        # could not afford it}?
        "MEDCOST": (
            "MEDCOST",  # 2015, 2017, 2019
            "MEDCOST1",  # 2021
        ),
        # Question: Is your annual household income from all sources: (If 
        # respondent refuses at any income level, code ´Refused.´) Note: 
        # higher levels/new codes added in 2021. 
        "INCOME": (
            "INCOME2",  # 2015, 2017, 2019
            "INCOME3",  # 2021
        )
    }
    for outname, input_names in mapping.items():
        
        assert len(set(df.columns).intersection(set(input_names))), \
            f"none of {input_names} detected in dataframe with " \
            f"columns {sorted(df.columns)}"
        
        df.rename(columns={old: outname for old in input_names}, inplace=True)
        assert outname in df.columns
    return df


def preprocess_brfss(df: pd.DataFrame):
    # Label
    df["DIABETES"].replace({2: 0, 3: 0, 4: 0}, inplace=True)
    # Drop 1k missing/not sure, plus one missing observation
    df = df[~(df["DIABETES"].isin([7, 9]))].dropna(subset=["DIABETES"])

    # Sensitive columns
    # Drop no preferred race/not answered/don't know/not sure
    df = df[~(df["_PRACE1"].isin([7, 8, 77, 99]))]
    df["_PRACE1"] = (df["_PRACE1"] == 1).astype(int)
    df["SEX"] = (df["SEX"] - 1).astype(int)  # Map sex to male=0, female=1

    # PHYSHLTH, POORHLTH, MENTHLTH are measured in days, but need to
    # map 88 to 0 because it means zero (i.e. zero bad health days)
    df["PHYSHLTH"] = df["PHYSHLTH"].replace({88: 0})
    df["MENTHLTH"] = df["MENTHLTH"].replace({88: 0})

    # Drop rows where drinks per week is unknown/refused/missing;
    # this uses a different missingness code from other variables.
    df = df[~(df["DRNK_PER_WEEK"] == 99900)]

    # Some questions are not asked for various reasons
    # (see notes under "BLANK" for that question in data dictionary);
    # create an indicator for these due to large fraction of missingness.
    df["SMOKDAY2"] = df["SMOKDAY2"].fillna("NOTASKED_MISSING").astype(str)
    df["TOLDHI2"] = df["TOLDHI2"].fillna("NOTASKED_MISSING").astype(str)

    # IYEAR is poorly coded, as e.g. "b'2015'"; here we parse it back to int.
    df["IYEAR"] = df["IYEAR"].apply(
        lambda x: re.search("\d+", x).group()).astype(int)

    NUMERIC_COLS = ("_BMI5", "DRNK_PER_WEEK", "PHYSHLTH", "MENTHLTH", "PA1MIN_",
                    "IYEAR")

    # For these categorical columns, drop respondents who were not sure,
    # refused, or had missing responses. This is also useful because
    # sometimes those responses (dk/refuse/missing) are lumped into
    # a single category (e.g. "_TOTINDA").
    DROP_MISSING_REFUSED_COLS = (
        "MEDCOST", "PHYSHLTH", "_RFHYPE5", "CHOL_CHK_PAST_5_YEARS", "SMOKE100",
        "SMOKDAY2", "TOLDHI2", "CVDSTRK3", "_TOTINDA", "FRUIT_ONCE_PER_DAY",
        "VEG_ONCE_PER_DAY", "_RFBING5", "PA1MIN_", "INCOME", "MARITAL", "CHECKUP1",
        "EDUCA", "_MICHD", "_BMI5", "_BMI5CAT")

    for c in DROP_MISSING_REFUSED_COLS:
        if c not in NUMERIC_COLS:
            # Apply coded values for missing/refused/idk, for categorical cols.
            # Note that 88 is sometimes used for for these, but 8 is NOT
            # and constitutes a valid value in the above columns.
            df = df[~(df[c].isin([7, 9, 77, 88, 99]))]
        # Drop actual missing values, for all column dtypes
        df.dropna(subset=[c], inplace=True)

    # Cast columns to categorical; since some columns have mixed type,
    # we cast the entire column to string.
    for c in df.columns:
        if c not in NUMERIC_COLS and c != BRFSS_FEATURES.target:
            df[c] = df[c].apply(str).astype("category")

    # Remove leading underscores from column names
    renames = {c: re.sub("^_", "", c) for c in df.columns if c.startswith("_")}
    df.rename(columns=renames, inplace=True)

    # Select features and reset the index after subsampling;
    # resetting ensures that splitting happens correctly.
    df = df.loc[:, BRFSS_FEATURES.names].reset_index(drop=True)
    return df
