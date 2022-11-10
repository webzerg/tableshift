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
# Features shared across BRFSS prediction tasks.
BRFSS_SHARED_FEATURES = FeatureList(features=[
    # Derived feature for year; keep as categorical dtype so normalization
    # is not applied.
    Feature("IYEAR", int, "Year of BRFSS dataset."),
    # ################ Demographics/sensitive attributes. ################
    # Also see "INCOME2", "MARITAL", "EDUCA" features below.
    Feature("STATE", cat_dtype),
    # Was there a time in the past 12 months when you needed to see a doctor
    # but could not because of cost?
    Feature("MEDCOST", cat_dtype, na_values=(7, 9)),
    # Preferred race category; note that ==1 is equivalent to
    # "White non-Hispanic race group" variable _RACEG21
    Feature("PRACE1", int, na_values=(77, 99)),
    # Indicate sex of respondent.
    Feature("SEX", int),
])

BRFSS_DIET_FEATURES = [
    # Consume Fruit 1 or more times per day
    Feature("FRUIT_ONCE_PER_DAY", cat_dtype, na_values=(9,)),
    # Consume Vegetables 1 or more times per day
    Feature("VEG_ONCE_PER_DAY", cat_dtype, na_values=(9,)),
]

BRFSS_ALCOHOL_FEATURES = [
    # Calculated total number of alcoholic beverages consumed per week
    Feature("DRNK_PER_WEEK", float, na_values=(99900,)),
    # Binge drinkers (males having five or more drinks on one occasion,
    # females having four or more drinks on one occasion)
    Feature("RFBING5", cat_dtype, na_values=(9,)),
]

BRFSS_SMOKE_FEATURES = [
    # Have you smoked at least 100 cigarettes in your entire life?
    Feature("SMOKE100", cat_dtype, na_values=(7, 9)),
    # Do you now smoke cigarettes every day, some days, or not at all?
    Feature("SMOKDAY2", cat_dtype, na_values=(7, 9)),
]

# Brief feature descriptions below; for the full question/description
# see the data dictionary linked above. Note that in the original data,
# some of the feature names are preceded by underscores (these are
# "calculated variables"; see data dictionary). These leading
# underscores, where present, are removed in the preprocess_brfss() function
# due to limitations on naming in the sklearn transformers module.

BRFSS_DIABETES_FEATURES = FeatureList([
    ################ Target ################
    Feature("DIABETES", int, is_target=True, na_values=(7, 9)),
    # (Ever told) you have diabetes

    # Below are a set of indicators for known risk factors for diabetes.
    ################ General health ################
    # for how many days during the past 30 days was your
    # physical health not good?
    Feature("PHYSHLTH", float, na_values=(77, 99)),
    ################ High blood pressure ################
    # Adults who have been told they have high blood pressure by a
    # doctor, nurse, or other health professional
    Feature("HIGH_BLOOD_PRESS", cat_dtype, na_values=(9,)),
    ################ High cholesterol ################
    # Cholesterol check within past five years
    Feature("CHOL_CHK_PAST_5_YEARS", cat_dtype, na_values=(9,)),
    # Have you EVER been told by a doctor, nurse or other health
    # professional that your blood cholesterol is high?
    Feature("TOLDHI2", cat_dtype, na_values=(7, 9)),
    ################ BMI/Obesity ################
    # Calculated Body Mass Index (BMI)
    Feature("BMI5", float),
    # Four-categories of Body Mass Index (BMI)
    Feature("BMI5CAT", cat_dtype),
    ################ Smoking ################
    *BRFSS_SMOKE_FEATURES,
    ################ Other chronic health conditions ################
    # (Ever told) you had a stroke.
    Feature("CVDSTRK3", cat_dtype, na_values=(7, 9)),
    # ever reported having coronary heart disease (CHD)
    # or myocardial infarction (MI)
    Feature("MICHD", cat_dtype),
    ################ Diet ################
    *BRFSS_DIET_FEATURES,
    ################ Alcohol Consumption ################
    *BRFSS_ALCOHOL_FEATURES,
    ################ Exercise ################
    # Adults who reported doing physical activity or exercise
    # during the past 30 days other than their regular job
    Feature("TOTINDA", cat_dtype, na_values=(9,)),
    ################ Household income ################
    # annual household income from all sources
    Feature("INCOME", cat_dtype, na_values=(77, 99)),
    ################ Marital status ################
    Feature("MARITAL", cat_dtype, na_values=(9,)),
    ################ Time since last checkup
    # About how long has it been since you last visited a
    # doctor for a routine checkup?
    Feature("CHECKUP1", cat_dtype, na_values=(7, 9)),
    ################ Education ################
    # highest grade or year of school completed
    Feature("EDUCA", cat_dtype, na_values=(9,)),
    ################ Health care coverage ################
    # Respondents aged 18-64 who have any form of health care coverage
    # Note: we keep missing values (=9) for this column since they are grouped
    # with respondents aged over 64; otherwise dropping the observations
    # with this value would exclude all respondents over 64.
    Feature("HEALTH_COV", cat_dtype),
    ################ Mental health ################
    # for how many days during the past 30
    # days was your mental health not good?
    Feature("MENTHLTH", float, na_values=(77, 99)),
]) + BRFSS_SHARED_FEATURES

BRFSS_BLOOD_PRESSURE_FEATURES = FeatureList(features=[
    Feature("HI_BP", int, """Have you ever been told by a doctor, nurse or 
    other health professional that you have high blood pressure?""",
            is_target=True),

    # Indicators for high blood pressure; see
    # https://www.nhlbi.nih.gov/health/high-blood-pressure/causes

    ################ Age ################
    Feature("AGEG5YR", int, "Fourteen-level age category",
            na_values=(14,)),
    ################ Family history and genetics ################
    # No questions related to this risk factor.
    ################ Lifestyle habits ################
    *BRFSS_DIET_FEATURES,
    *BRFSS_ALCOHOL_FEATURES,
    # Adults who reported doing physical activity or exercise
    # during the past 30 days other than their regular job
    Feature("TOTINDA", cat_dtype, na_values=(9,)),
    *BRFSS_SMOKE_FEATURES,
    ################ Medicines ################
    # No questions related to this risk factor.
    ################ Other medical conditions ################
    Feature("CHCSCNCR", cat_dtype, "(Ever told) (you had) skin cancer?",
            na_values=(7, 9)),
    Feature("CHCOCNCR", cat_dtype,
            "(Ever told) you had any other types of cancer?",
            na_values=(7, 9)),

    ################ Race/ethnicity ################
    # Covered in BRFSS_SHARED_FEATURES.
    ################ Sex ################
    # Covered in BRFSS_SHARED_FEATURES.
    ################ Social and economic factors ################
    # Income
    Feature("INCOME", cat_dtype, na_values=(77, 99)),
    # Type job status; related to early/late shifts which is a risk factor.
    Feature("EMPLOY1", cat_dtype, "Are you currently…?",
            na_values=(9,)),
    # Additional relevant features in BRFSS_SHARED_FEATURES.
]) + BRFSS_SHARED_FEATURES

# Some features have different names over years, due to changes in prompts or
# interviewer instructions. Here we map these different names to a single shared
# name that is consistent across years.
BRFSS_CROSS_YEAR_FEATURE_MAPPING = {
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
    ),
    # Question: Adults who have been told they have high blood pressure by a
    # doctor, nurse, or other health professional
    "HIGH_BLOOD_PRESS": (
        "_RFHYPE5",  # 2015, 2017, 2019
        "_RFHYPE6",  # 2021
    ),
    # Question: Respondents aged 18-64 who have any form of health insurance
    "HEALTH_COV": (
        "_HCVU651",  # 2015, 2017, 2019
        "_HCVU652",  # 2021
    )
}

# Raw names of the input features used in BRFSS. Useful to
# subset before preprocessing, since some features contain near-duplicate
# versions (i.e. calculated and not-calculated versions, differing only by a
# precending underscore).
_BRFSS_INPUT_FEATURES = list(
    set(['_AGEG5YR', 'CHECKUP1', 'CHCSCNCR', 'CHCOCNCR', 'CVDSTRK3', 'EDUCA',
         'IYEAR', 'MARITAL', 'MEDCOST', 'MENTHLTH', 'PHYSHLTH', 'SEX',
         'SMOKDAY2', 'SMOKE100', 'TOLDHI2', '_BMI5', '_BMI5CAT', '_MICHD',
         '_PRACE1', '_RFBING5', '_STATE', '_TOTINDA'] +
        list(BRFSS_CROSS_YEAR_FEATURE_MAPPING.keys())))


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

    for outname, input_names in BRFSS_CROSS_YEAR_FEATURE_MAPPING.items():
        assert len(set(df.columns).intersection(set(input_names))), \
            f"none of {input_names} detected in dataframe with " \
            f"columns {sorted(df.columns)}"

        df.rename(columns={old: outname for old in input_names}, inplace=True)
        assert outname in df.columns
    return df


def preprocess_brfss(df: pd.DataFrame, target_colname: str) -> pd.DataFrame:
    """Shared preprocessing function for BRFSS data tasks."""
    df = df[_BRFSS_INPUT_FEATURES]

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

    # Remove leading underscores from column names
    renames = {c: re.sub("^_", "", c) for c in df.columns if c.startswith("_")}
    df.rename(columns=renames, inplace=True)

    return df


def preprocess_brfss_diabetes(df: pd.DataFrame):
    df = preprocess_brfss(df, target_colname=BRFSS_DIABETES_FEATURES.target)

    df["DIABETES"].replace({2: 0, 3: 0, 4: 0}, inplace=True)

    # Reset the index after preprocessing to ensure splitting happens
    # correctly (splitting assumes sequential indexing).
    return df.reset_index(drop=True)


def preprocess_brfss_blood_pressure(df: pd.DataFrame) -> pd.DataFrame:
    df = preprocess_brfss(df, BRFSS_BLOOD_PRESSURE_FEATURES.target)

    # TODO(jpgard): this shouldnt be needed if na_values are dropped.
    # Drop samples where age is not reported; this has different coding for
    # the missing values than other numeric columns.
    df = df[~(df["_AGEG5YR"] != 14)]

    # Reset the index after preprocessing to ensure splitting happens
    # correctly (splitting assumes sequential indexing).
    return df.reset_index(drop=True)
