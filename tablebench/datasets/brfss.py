"""

Utilities for working with BRFSS dataset.

Accessed via https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system.
Raw Data: https://www.cdc.gov/brfss/annual_data/annual_data.htm
Data Dictionary: https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf
"""

import numpy as np
import re

import pandas as pd

from tablebench.core.features import Feature, FeatureList, cat_dtype
from tablebench.core.splitter import idx_where_not_in

# Features present in every year of BRFSS
BRFSS_GLOBAL_FEATURES = [
    'CHCOCNCR', 'CHCSCNCR', 'CHECKUP1', 'CVDSTRK3', 'EDUCA', 'EMPLOY1',
    'HIGH_BLOOD_PRESS', 'IYEAR', 'MARITAL', 'MEDCOST', 'MENTHLTH',
    'PHYSHLTH', 'SEX', 'SMOKDAY2', 'SMOKE100', '_AGEG5YR',
    '_BMI5', '_BMI5CAT', '_MICHD', '_PRACE1', '_RFBING5', '_STATE', '_TOTINDA']

# While BRFSS exists for every year back several decades, feature alignment
# is only implemented for these years due to "rotating core" features occurring
# only every other year and other changes prior to 2015; see comments below.
BRFSS_YEARS = (2015, 2017, 2019, 2021)

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
    # Derived feature for year.
    Feature("IYEAR", float, "Year of BRFSS dataset."),
    # ################ Demographics/sensitive attributes. ################
    # Also see "INCOME2", "MARITAL", "EDUCA" features below.
    Feature("STATE", cat_dtype, """State FIPS Code. Values: 1 Alabama, 
    4 Arizona, 5 Arkansas, 6 California, 8 Colorado, 9 Connecticut, 
    10 Delaware, 11 District of Columbia, 12 Florida, 13 Georgia, 15 Hawaii , 
    16 Idaho, 17, 18 Indiana, 19 Iowa, 20 Kansas, 21 Kentucky, 22 Louisiana , 
    23 Maine, 24 Maryland, 25 Massachusetts, 26 Michigan, 27 Minnesota, 
    28 Mississippi, 29 Missouri, 30 Montana, 31 Nebraska, 32 Nevada, 33 New 
    Hampshire, 34 New Jersey, 35 New Mexico, 36 New York, 37 North Carolina, 
    38 North Dakota, 39 Ohio, 40 Oklahoma, 41 Oregon, 42 Pennsylvania, 
    44 Rhode Island, 45 South Carolina, 46 South Dakota, 47 Tennessee, 
    48 Texas, 49 Utah, 50 Vermont, 51 Virginia, 53 Washington, 54 West 
    Virginia, 55 Wisconsin, 56 Wyoming, 66 Guam, 72 Puerto Rico"""),
    Feature("MEDCOST", cat_dtype, """Was there a time in the past 12 months 
    when you needed to see a doctor but could not because of cost?""",
            na_values=(7, 9)),
    # Preferred race category; note that ==1 is equivalent to
    # "White non-Hispanic race group" variable _RACEG21
    Feature("PRACE1", float, """Preferred race category. Values: 1 White, 
    2 Black or African American, 3  American Indian or Alaskan Native, 
    4:  Asian, 5:  Native Hawaiian or other Pacific Islander, 6: Other race, 
    7: No preferred race, 8: Multiracial but preferred race not answered, 
    77: Don’t know/Not sure, 99: refused.""",
            na_values=(7., 8., 77., 99.)),
    Feature("SEX", float, """Indicate sex of respondent. 1 Male, 2 Female."""),
])

BRFSS_DIET_FEATURES = [
    Feature("FRUIT_ONCE_PER_DAY", cat_dtype,
            """Consume Fruit 1 or more times per day. Values: 1 Consumed 
            fruit one or more times per day, 2: Consumed fruit less than one 
            time per day, 9 Don´t know, refused or missing values""",
            na_values=(9,)),
    Feature("VEG_ONCE_PER_DAY", cat_dtype,
            """Consume vegetables 1 or more times per day. Values: 1 Consumed 
            vegetables one or more times per day, 2: Consumed vegetables less 
            than one time per day, 9 Don´t know, refused or missing values""",
            na_values=(9,)),
]

BRFSS_ALCOHOL_FEATURES = [
    # Calculated total number of alcoholic beverages consumed per week
    Feature("DRNK_PER_WEEK", float,
            """Calculated total number of alcoholic beverages consumed per 
            week. Values: 0: Did not drink, 1 - 98999: Number of drinks per 
            week, 99900: Don’t know/Not sure/Refused/Missing""",
            na_values=(99900,)),
    Feature("RFBING5", cat_dtype,
            """Binge drinkers (males having five or more drinks on one 
            occasion, females having four or more drinks on one occasion). 
            Values: 1: No, 2: Yes,  9: Don’t know/Refused/Missing""",
            na_values=(9,)),
]

BRFSS_SMOKE_FEATURES = [
    # Have you smoked at least 100 cigarettes in your entire life?
    Feature("SMOKE100", cat_dtype,
            """Have you smoked at least 100 cigarettes in your entire life? 
            Values: 1: Yes, 2: N0, 7: Don’t know/Not Sure, 9: Refused, 
            BLANK Not asked or Missing.""",
            na_values=(7, 9)),

    Feature("SMOKDAY2", cat_dtype, """Do you now smoke cigarettes every day, 
    some days, or not at all? 1 Every day, 2 Some days, 3 Not at all, 7 Don´t 
    Know/Not Sure, 9 Refused, BLANK Not asked or Missing.""",
            na_values=(7, 9)),
]

# Brief feature descriptions below; for the full question/description
# see the data dictionary linked above. Note that in the original data,
# some of the feature names are preceded by underscores (these are
# "calculated variables"; see data dictionary). These leading
# underscores, where present, are removed in the preprocess_brfss() function
# due to limitations on naming in the sklearn transformers module.

BMI5CAT_FEATURE = Feature("BMI5CAT", cat_dtype, """Four-categories of Body Mass Index ( 
    BMI). Values: 1 Underweight (_BMI5 < 1850), 2 Normal Weight (1850 <= 
    _BMI5 < 2500), 3 Overweight (2500 <= _BMI5 < 3000), 4 Obese (3000 <= 
    _BMI5 < 9999), BLANK Don’t know/Refused/Missing.""")
BRFSS_DIABETES_FEATURES = FeatureList([
    ################ Target ################
    Feature("DIABETES", float,
            """(Ever told) you have diabetes (If "Yes" and respondent is 
            female, ask "Was this only when you were pregnant?". If 
            Respondent says pre-diabetes or borderline diabetes, use response 
            code 4.). Values: 1 Yes, 2 Yes, but female told only during 
            pregnancy, 3 No, 4 No, pre-diabetes or borderline diabetes, 
            7 Don’t know/Not Sure, 9 Refused, BLANK Not asked or Missing.""",
            is_target=True, na_values=(7, 9)),

    # Below are a set of indicators for known risk factors for diabetes.
    ################ General health ################
    Feature("PHYSHLTH", float,
            """Now thinking about your physical health, which includes 
            physical illness and injury, for how many days during the past 30 
            days was your physical health not good? Values: 1 - 30 Number of 
            days, 88 None, 77 Don’t know/Not sure, 99 Refused, BLANK Not 
            asked or Missing""",
            na_values=(77, 99)),
    ################ High blood pressure ################

    Feature("HIGH_BLOOD_PRESS", cat_dtype, na_values=(9,),
            description="""Adults who have been told they have high blood 
            pressure by a doctor, nurse, or other health professional. 1: No. 
            2: Yes. 9: Don’t know/Not Sure/Refused/Missing"""),
    ################ High cholesterol ################
    # Cholesterol check within past five years
    Feature("CHOL_CHK_PAST_5_YEARS", cat_dtype,
            """Aligned version of 'CHOLCHK*' features from 2015-2021; see 
            _align_chol_chk() below. Values: 1 Never, 2 Within the past year 
            (anytime less than 12 months ago), 3 Within the past 2 years (1 
            year but less than 2 years ago), 4 Within the past 5 years (2 
            years but less than 5 years ago), 5 5 or more years ago, 7 Don’t 
            know/Not Sure, 9 Refused—Go to next section, BLANK Not asked or 
            Missing.""",
            na_values=(9,)),

    Feature("TOLDHI", cat_dtype,
            """Have you EVER been told by a doctor, nurse or other health 
            professional that your blood cholesterol is high? Values: 1: Yes, 
            2: No, 7 Don’t know/Not Sure, 9 Refused, BLANK Not asked or 
            Missing.""",
            na_values=(7, 9)),
    ################ BMI/Obesity ################
    # Calculated Body Mass Index (BMI)
    Feature("BMI5", float, """Computed Body Mass Index (BMI). Values: 1 - 
    9999 1 or greater - Notes: WTKG3/(HTM4*HTM4) (Has 2 implied decimal 
    places); BLANK: Don’t know/Refused/Missing."""),
    # Four-categories of Body Mass Index (BMI)
    BMI5CAT_FEATURE,
    ################ Smoking ################
    *BRFSS_SMOKE_FEATURES,
    ################ Other chronic health conditions ################
    Feature("CVDSTRK3", cat_dtype,
            """(Ever told) you had a stroke.. Values: 1 Yes, 2 No, 7 Don’t 
            know/Not sure, 9 Refused, BLANK Not asked or Missing.""",
            na_values=(7, 9)),
    Feature("MICHD", cat_dtype, """Question: Respondents that have ever 
    reported having coronary heart disease (CHD) or myocardial infarction ( 
    MI). Values: 1 Reported having MI or CHD (Notes: CVDINFR4=1 OR 
    CVDCRHD4=1), 2 Did not report having MI or CHD (Notes: CVDINFR4=2 AND 
    CVDCRHD4=2), BLANK Not asked or Missing."""),
    ################ Diet ################
    *BRFSS_DIET_FEATURES,
    ################ Alcohol Consumption ################
    *BRFSS_ALCOHOL_FEATURES,
    ################ Exercise ################
    Feature("TOTINDA", cat_dtype,
            """Adults who reported doing physical activity or exercise during 
            the past 30 days other than their regular job. Values: 1 Had 
            physical activity or exercise, 2 No physical activity or exercise 
            in last 30 days, 9 Don’t know/Refused/Missing.""",
            na_values=(9,)),
    ################ Household income ################
    Feature("INCOME", cat_dtype,
            """Question: Is your annual household income from all sources: ( 
            If respondent refuses at any income level, code 'Refused'.) 
            Values: 1: Less than $10,000, 2: Less than $15,000 ($10,000 to 
            less than $15,000), 3: Less than $20,000 ($15,000 to less than 
            $20,000), 4: Less than $25,000 ($20,000 to less than $25,000), 
            5: Less than $35,000 ($25,000 to less than $35,000), 6: Less than 
            $50,000 ($35,000 to less than $50,000), 7: Less than $75, 
            000 ($50,000 to less than $75,000), 8 (BRFSS 2015-2019) $75, 
            000 or more / (BRFSS 2021) Less than $100,000? ($75,000 to < 
            $100,000), 9: Less than $150,000? ($100,000 to < $150,000)?, 
            10: Less than $200,000? ($150,000 to < $200,000), 11: $200,000 
            or more, 77 Don’t know/Not sure, 99 Refused, BLANK Not asked 
            or Missing.""",
            na_values=(77, 99)),
    ################ Marital status ################
    Feature("MARITAL", cat_dtype,
            """Question: Are you: (marital status) 1 Married, 2 Divorced, 
            3 Widowed, 4 Separated, 5 Never married, 6 A member of an 
            unmarried couple, 9 Refused, BLANK Not asked or Missing""",
            na_values=(9,)),
    ################ Time since last checkup
    # About how long has it been since you last visited a
    # doctor for a routine checkup?
    Feature("CHECKUP1", cat_dtype,
            """Question: About how long has it been since you last visited a 
            doctor for a routine checkup? [A routine checkup is a general 
            physical exam, not an exam for a specific injury, illness, 
            or condition.] Values: 1 Within past year (anytime < 12 months 
            ago), 2 Within past 2 years (1 year but < 2 years ago), 3 Within 
            past 5 years (2 years but < 5 years ago), 4 5 or more years ago, 
            7 Don’t know/Not sure, 8 Never, 9 Refused, BLANK Not asked or 
            Missing.""",
            na_values=(7, 9)),
    ################ Education ################
    # highest grade or year of school completed
    Feature("EDUCA", cat_dtype,
            """Question: What is the highest grade or year of school you 
            completed? Values: 1 Never attended school or only kindergarten, 
            2 Grades 1 through 8 (Elementary), 3 Grades 9 through 11 (Some 
            high school), 4 Grade 12 or GED (High school graduate), 5 College 
            1 year to 3 years (Some college or technical school), 6 College 4 
            years or more (College graduate), 9 Refused, BLANK Not asked or 
            Missing.""",
            na_values=(9,)),
    ################ Health care coverage ################
    # Note: we keep missing values (=9) for this column since they are grouped
    # with respondents aged over 64; otherwise dropping the observations
    # with this value would exclude all respondents over 64.
    Feature("HEALTH_COV", cat_dtype,
            """Respondents aged 18-64 who have any form of health care 
            coverage. Values: 1: Have health care coverage (Notes: 18 <= AGE <=64 and 
            HLTHPLN1 = 1), 2: Do not have health care coverage (Notes: 18 <= 
            AGE <=64 and HLTHPLN1 = 2), 9: Don’t know/Not Sure, Refused or 
            Missing (Notes: AGE > 64 or AGE = Missing or HLTHPLN1 = 7 or 9 or 
            Missing)."""),
    ################ Mental health ################
    # for how many days during the past 30
    # days was your mental health not good?
    Feature("MENTHLTH", float,
            """Now thinking about your mental health, which includes stress, 
            depression, and problems with emotions, for how many days during 
            the past 30 days was your mental health not good? Values: 1 - 30: 
            Number of days, 88: None, 77: Don’t know/Not sure, 99: Refused.""",
            na_values=(77, 99)),
]) + BRFSS_SHARED_FEATURES

BRFSS_BLOOD_PRESSURE_FEATURES = FeatureList(features=[
    Feature("HIGH_BLOOD_PRESS", int,
            """Have you ever been told by a doctor, nurse or other health 
            professional that you have high blood pressure? 0: No. 1: Yes. 8: 
            Don’t know/Not Sure/Refused/Missing (note: we subtract 1 from 
            original codebook values at preprocessing to create a binary 
            target variable).""",
            is_target=True),

    # Indicators for high blood pressure; see
    # https://www.nhlbi.nih.gov/health/high-blood-pressure/causes
    ################ BMI/Obesity ################
    # Four-categories of Body Mass Index (BMI)
    BMI5CAT_FEATURE,
    ################ Age ################
    Feature("AGEG5YR", float, """Fourteen-level age category. Values: 1: Age 
    18 to 24, 2: Age 25 to 29, 3 Age 30 to 34, 4 Age 35 to 39, 5 Age 40 to 
    44, 6 Age 45 to 49, 7 Age 50 to 54, 8 Age 55 to 59, 9 Age 60 to 64, 
    10 Age 65 to 69, 11 Age 70 to 74, 12 Age 75 to 79, 13 Age 80 or older, 
    14 Don’t know/Refused/Missing.""",
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
    Feature("CHCSCNCR", cat_dtype,
            """(Ever told) (you had) skin cancer? Values: 1 Yes, 2 No, 
            7 Don’t know / Not sure, 9 Refused, BLANK Not asked or Missing.""",
            na_values=(7, 9)),
    Feature("CHCOCNCR", cat_dtype,
            """(Ever told) you had any other types of cancer? Values: 1 Yes, 
            2 No, 7 Don’t know / Not sure, 9 Refused.""",
            na_values=(7, 9)),
    # 6 in 10 people suffering from diabetes also have high BP
    # source: https://www.cdc.gov/bloodpressure/risk_factors.htm
    Feature("DIABETES", float,
            """(Ever told) you have diabetes (If "Yes" and respondent is 
            female, ask "Was this only when you were pregnant?". If 
            Respondent says pre-diabetes or borderline diabetes, use response 
            code 4.). Values: 1 Yes, 2 Yes, but female told only during 
            pregnancy, 3 No, 4 No, pre-diabetes or borderline diabetes, 
            7 Don’t know/Not Sure, 9 Refused, BLANK Not asked or Missing.""",
            na_values=(7, 9)),

    ################ Race/ethnicity ################
    # Covered in BRFSS_SHARED_FEATURES.
    ################ Sex ################
    # Covered in BRFSS_SHARED_FEATURES.
    ################ Social and economic factors ################
    # Income
    Feature("POVERTY", int,
            description="Binary indicator for whether an individuals' income "
                        "falls below the 2021 poverty guideline for family of "
                        "four."),
    # Type job status; related to early/late shifts which is a risk factor.
    Feature("EMPLOY1", cat_dtype, """Question: Are you currently…? Values: 1 
    Employed for wages, 2 Self-employed, 3 Out of work for 1 year or more, 
    4 Out of work for less than 1 year, 5 A homemaker, 6 A student, 
    7 Retired, 8 Unable to work, 9 Refused.""",
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
    ),
    # Question: Have you ever been told by a doctor, nurse or other
    # health professional that your (TOLDHI2: blood) cholesterol is high?
    "TOLDHI": (
        "TOLDHI2",
        "TOLDHI3"  # 2021
    )
}

# Raw names of the input features used in BRFSS. Useful to
# subset before preprocessing, since some features contain near-duplicate
# versions (i.e. calculated and not-calculated versions, differing only by a
# precending underscore).
_BRFSS_INPUT_FEATURES = list(
    set(BRFSS_GLOBAL_FEATURES +
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

    # IYEAR is poorly coded, as e.g. "b'2015'"; here we parse it back to int.
    df["IYEAR"] = df["IYEAR"].apply(
        lambda x: re.search("\d+", x).group()).astype(int)

    # CHOLCHK values are coded inconsistently in 2015 vs. post-2015 surveys;
    # we align them here.
    # (See https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf)
    def _align_chol_chk(row):
        """Utility function to code 2015 BRFSS CHOLCHK to match post-2015."""
        if row["year"] == 2015 and row["CHOL_CHK_PAST_5_YEARS"] in (1, 2, 3, 4):
            return row["CHOL_CHK_PAST_5_YEARS"] + 1
        else:
            return row["CHOL_CHK_PAST_5_YEARS"]

    df["CHOL_CHK_PAST_5_YEARS"] = df.apply(_align_chol_chk)
    return df


def brfss_shared_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Shared preprocessing function for BRFSS data tasks."""
    df = df[_BRFSS_INPUT_FEATURES]

    # Sensitive columns
    # df["_PRACE1"] = (df["_PRACE1"] == 1).astype(int)
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
    df["TOLDHI"] = df["TOLDHI"].fillna("NOTASKED_MISSING").astype(str)

    # Remove leading underscores from column names
    renames = {c: re.sub("^_", "", c) for c in df.columns if c.startswith("_")}
    df.rename(columns=renames, inplace=True)

    return df


def preprocess_brfss_diabetes(df: pd.DataFrame):
    df = brfss_shared_preprocessing(df)

    df["DIABETES"].replace({2: 0, 3: 0, 4: 0}, inplace=True)

    # Reset the index after preprocessing to ensure splitting happens
    # correctly (splitting assumes sequential indexing).
    return df.reset_index(drop=True)


def preprocess_brfss_blood_pressure(df: pd.DataFrame) -> pd.DataFrame:
    df = brfss_shared_preprocessing(df)

    df["HIGH_BLOOD_PRESS"] = df["HIGH_BLOOD_PRESS"].replace(9, np.nan) - 1
    df.dropna(subset=["HIGH_BLOOD_PRESS"], inplace=True)

    # Retain samples only 50+ years of age (to focus on highest-risk groups
    # for high BP; see:
    # * Vasan RS, Beiser A, Seshadri S, Larson MG, Kannel WB, D’ Agostino RB,
    # et al. Residual lifetime risk for developing hypertension in middle-aged
    # women and men: the Framingham Heart Studyexternal icon. JAMA. 2002;287(
    # 8):1003–1010. ("the risk for developing hypertension increases markedly
    # during and after the sixth decade of life") and also:
    # * Dannenberg AL,  Garrison RJ, Kannel WB. Incidence of hypertension in the
    # Framingham Study. Am J Public Health.1988;78:676-679.

    df = df[df["AGEG5YR"] >= 7]

    # Create a binary indicator for poverty. This is based on the 2021 US poverty income guideline for a family of 4,
    # which was $26,500. https://aspe.hhs.gov/topics/poverty-economic-mobility/poverty-guidelines/prior-hhs-poverty
    # -guidelines-federal-register-references/2021-poverty-guidelines#threshholds Note that we actually use a
    # slightly lower threshold of 25,000 due to the response coding in BRFSS.

    # Drop unknown/not responded income levels; otherwise comparison with nan values always returns False.
    idxs = idx_where_not_in(df["INCOME"], (77, 99))
    df = df.iloc[idxs]
    df["POVERTY"] = (df["INCOME"] <= 4).astype(int)
    df.drop(columns=["INCOME"], inplace=True)

    # Reset the index after preprocessing to ensure splitting happens
    # correctly (splitting assumes sequential indexing).
    return df.reset_index(drop=True)


def preprocess_brfss(df, task: str):
    assert task in ("diabetes", "blood_pressure")
    if task == "diabetes":
        return preprocess_brfss_diabetes(df)
    elif task == "blood_pressure":
        return preprocess_brfss_blood_pressure(df)
    else:
        raise NotImplementedError
