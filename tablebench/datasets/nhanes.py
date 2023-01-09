"""
NHANES-related tools. See also the documentation at the link below:
https://www.cdc.gov/Nchs/Nhanes/about_nhanes.htm
"""
from collections import defaultdict
import json
import os

import numpy as np
import pandas as pd

from tablebench.core.features import Feature, FeatureList, cat_dtype

NHANES_YEARS = [1999, 2001, 2003, 2005, 2007, 2009, 2011, 2013, 2015, 2017]

# Dictionary mapping years to data sources. Because NHANES uses the same
# name for each file, we need to manually track the year associated with
# each dataset.
NHANES_DATA_SOURCES = os.path.join(os.path.dirname(__file__),
                                   "nhanes_data_sources.json")

# Mapping of NHANES component types to names of data sources to use.
# See nhanes_data_sources.json. This ensures that only needed files
# are downloaded/read from disk, because NHANES contains a huge number of sources per year.

NHANES_CHOLESTEROL_DATA_SOURCES_TO_USE = {
    "Demographics": [
        "Demographic Variables & Sample Weights",  # 1999 - 2003
        "Demographic Variables and Sample Weights"],  # 2005- 2017
    "Questionnaire": ["Blood Pressure & Cholesterol",  # All years
                      "Cardiovascular Health",  # All years
                      "Diabetes",  # All years
                      "Kidney Conditions",  # 1999
                      "Kidney Conditions - Urology",  # 2001 - 2017
                      "Medical Conditions",  # All years
                      "Osteoporosis",  # Not preset in 2011, 2015
                      ],
    "Laboratory": ["Cholesterol - LDL & Triglycerides",  # 1999 - 2003, 2007 - 2013
                   "Cholesterol - LDL, Triglyceride & Apoliprotein (ApoB)",  # 2005
                   "Cholesterol - Low - Density Lipoprotein (LDL) & Triglycerides",  # 2015
                   "Cholesterol - Low-Density Lipoproteins (LDL) & Triglycerides"  # 2017
                   ],
}

NHANES_LEAD_DATA_SOURCES_TO_USE = {
    "Demographics": [
        "Demographic Variables & Sample Weights",  # 1999 - 2003
        "Demographic Variables and Sample Weights"],  # 2005- 2017
    # TODO(jpgard): fill in missing data sources for the below categories for lead.
    "Questionnaire": ["Diet Behavior & Nutrition",
                      "Income"  # 2007 - 2017; prior to 2017 income questions are in Demographics.
                      ],
    "Laboratory": [
        "Cadmium, Lead, Mercury, Cotinine & Nutritional Biochemistries",  # 1999
        "Cadmium, Lead, Total Mercury, Ferritin, Serum Folate, RBC Folate, "
        "Vitamin B12, Homocysteine, Methylmalonic "
        "acid, Cotinine - Blood, Second Exam",  # 2001
        "Cadmium, Lead, & Total Mercury - Blood",  # 2003
        "Cadmium, Lead, & Total Mercury - Blood",  # 2005
        "Cadmium, Lead, & Total Mercury - Blood",  # 2007
        "Cadmium, Lead, & Total Mercury - Blood",  # 2009
        "Cadmium, Lead, Total Mercury, Selenium, & Manganese - Blood",  # 2011
        "Lead, Cadmium, Total Mercury, Selenium, and Manganese - Blood",  # 2013
        "Lead, Cadmium, Total Mercury, Selenium & Manganese - Blood",  # 2015
        "Lead, Cadmium, Total Mercury, Selenium, & Manganese - Blood"]  # 2017
}


def get_nhanes_data_sources(task: str, years=None):
    """Fetch a mapping of {year: list of urls} for NHANES."""
    years = [int(x) for x in years]
    if task == "cholesterol":
        data_sources_to_use = NHANES_CHOLESTEROL_DATA_SOURCES_TO_USE
    elif task == "lead":
        data_sources_to_use = NHANES_LEAD_DATA_SOURCES_TO_USE
    else:
        raise ValueError

    output = defaultdict(list)
    with open(NHANES_DATA_SOURCES, "r") as f:
        data_sources = json.load(f)
    for year, components in data_sources.items():
        if (years is not None) and (int(year) in years):
            for component, sources in components.items():
                for source_name, source_url in sources.items():
                    if source_name in data_sources_to_use[component]:
                        output[year].append(source_url)
    return output


NHANES_SHARED_FEATURES = FeatureList(features=[
    # Derived feature for survey year
    Feature("nhanes_year", int, "Derived feature for year."),

    Feature('DMDBORN4', cat_dtype, """In what country {were you/was SP} born? 1	Born in 50 US states or Washington, 
    DC 2 Others""", na_values=(77, 99, ".")),

    # What is the highest grade or level of school {you have/SP has} completed
    # or the highest degree {you have/s/he has} received?
    Feature('DMDEDUC2', cat_dtype),

    # Age in years of the participant at the time of screening. Individuals
    # 80 and over are topcoded at 80 years of age.
    Feature('RIDAGEYR', float),

    # Gender of the participant.
    Feature('RIAGENDR', cat_dtype),

    # Marital status
    Feature('DMDMARTL', cat_dtype),

    Feature('RIDRETH_merged', int),

], documentation="https://wwwn.cdc.gov/Nchs/Nhanes/")

NHANES_CHOLESTEROL_FEATURES = FeatureList(features=[

    Feature('LBDLDL', float, is_target=True, description='Direct LDL-Cholesterol (mg/dL)'),

    # Below we use the additional set of risk factors listed in the above report
    # (Table 6) **which can be asked in a questionnaire** (i.e. those which
    # do not require laboratory testing).

    ####### Risk Factor: Family history of ASCVD

    # No questions on this topic.

    ####### Risk Factor: Metabolic syndrome (increased waist circumference,
    # elevated triglycerides [>175 mg/dL], elevated blood pressure,
    # elevated glucose, and low HDL-C [<40 mg/dL in men; <50 in women
    # mg/dL] are factors; tally of 3 makes the diagnosis)

    # {Have you/Has SP} ever been told by a doctor or other health professional
    # that {you/s/he} had hypertension, also called high blood pressure?
    Feature('BPQ020', cat_dtype),

    # {Have you/Has SP} ever been told by a doctor or other health professional
    # that {you have/SP has} any of the following: prediabetes, impaired
    # fasting glucose, impaired glucose tolerance, borderline diabetes or
    # that {your/her/his} blood sugar is higher than normal but not high enough
    # to be called diabetes or sugar diabetes?
    Feature('DIQ160', cat_dtype),

    # The next questions are about specific medical conditions.
    # {Other than during pregnancy, {have you/has SP}/{Have you/Has SP}}
    # ever been told by a doctor or health professional that
    # {you have/{he/she/SP} has} diabetes or sugar diabetes?
    Feature('DIQ010', cat_dtype),

    ####### Risk Factor: Chronic kidney disease
    # In the past 12 months, {have you/has SP} received dialysis
    # (either hemodialysis or peritoneal dialysis)?
    Feature('KIQ025', cat_dtype),

    # {Have you/Has SP} ever been told by a doctor or other health professional
    # that {you/s/he} had weak or failing kidneys? Do not include kidney
    # stones, bladder infections, or incontinence.
    Feature('KIQ022', cat_dtype),

    ####### Risk Factor: Chronic inflammatory conditions such as
    # psoriasis, RA, or HIV/AIDS

    Feature('MCQ070', cat_dtype,
            description="{Have you/Has SP} ever been told by a doctor or other health care "
                        "professional that {you/s/he} had psoriasis (sore-eye-asis)?"
                        "(note: not present after 2013)"),

    # Has a doctor or other health professional ever told {you/SP} that
    # {you/s/he} . . .had arthritis (ar-thry-tis)?
    Feature('MCQ160B', cat_dtype),

    # Note: no questions about HIV/AIDS.

    #######  Risk Factor: History of premature menopause (before age 40 y)
    # and history of pregnancy-associated conditions that increase later
    #  ASCVD risk such as preeclampsia

    # Note: no questions on these.

    # #######  Risk Factor: High-risk race/ethnicities (eg, South Asian ancestry)
    # Covered in shared 'RIDRETH' feature
], documentation="https://wwwn.cdc.gov/Nchs/Nhanes/")


def _postprocess_nhanes(df: pd.DataFrame, features) -> pd.DataFrame:
    # Fill categorical missing values with "missing".
    for feature in features:
        name = feature.name
        if name not in df.columns:
            print(f"[WARNING] feature {feature.name} missing; filling with indicator;"
                  f"this can happen when data is subset by years since some questions "
                  f"are not asked in all survey years.")
            df[name] = pd.Series(["MISSING"] * len(df))

        elif name != features.target and feature.kind == cat_dtype:
            print(f"[DEBUG] filling and casting categorical feature {name}")
            df[name] = df[name].fillna("MISSING").apply(str).astype("category")

    df.reset_index(drop=True, inplace=True)
    return df


def _merge_ridreth_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create a single race/ethnicity feature by using 'RIDRETH3'
    where available, else 'RIDRETH1'. """
    if ('RIDRETH3' in df.columns) and ('RIDRETH1' in df.columns):
        race_col = np.where(~np.isnan(df['RIDRETH3']), df['RIDRETH3'], df['RIDRETH1'])
        df.drop(columns=['RIDRETH3', 'RIDRETH1'], inplace=True)
    elif 'RIDRETH3' in df.columns:
        race_col = df['RIDRETH3']
        df.drop(columns=['RIDRETH3'], inplace=True)
    else:
        race_col = df['RIDRETH1']
        df.drop(columns=['RIDRETH1'], inplace=True)

    df['RIDRETH_merged'] = race_col
    return df


def preprocess_nhanes_cholesterol(df: pd.DataFrame, threshold=160.):
    features = NHANES_CHOLESTEROL_FEATURES + NHANES_SHARED_FEATURES
    try:
        assert "LBXBPB" not in features.names
        assert 'INDFMPIRBelowCutoff' not in features.names
    except AssertionError as ae:
        print(ae)
        import ipdb;ipdb.set_trace()
    df = _merge_ridreth_features(df)

    # Drop observations with missing target or missing domain split variable
    df.dropna(subset=[features.target, 'RIDRETH_merged'], inplace=True)

    # Binarize the target
    df[features.target] = (df[features.target] >= threshold).astype(float)

    df = _postprocess_nhanes(df, features=features)
    return df


NHANES_LEAD_FEATURES = FeatureList(features=[

    # A ratio of family income to poverty guidelines.
    Feature('INDFMPIRBelowCutoff', float,
            'Binary indicator for whether family PIR (poverty-income ratio)'
            'is <= 1.3. The threshold of 1.3 is selected based on the categorization '
            'in NHANES, where PIR <= 1.3 is the lowest level (see INDFMMPC feature).'),

    Feature("LBXBPB", float, "Blood lead (ug/dL)", is_target=True,
            na_values=(".",)),
], documentation="https://wwwn.cdc.gov/Nchs/Nhanes/")


def preprocess_nhanes_lead(df: pd.DataFrame, threshold: float = 3.5):
    """Preprocess the NHANES lead prediction dataset.

    The value of 3.5 µg/dl is based on the CDC Blood Lead Reference Value
    (BLRF) https://www.cdc.gov/nceh/lead/prevention/blood-lead-levels.htm
    """
    features = NHANES_LEAD_FEATURES + NHANES_SHARED_FEATURES
    target = NHANES_LEAD_FEATURES.target
    df = _merge_ridreth_features(df)

    # Drop observations with missing target and missing domain split
    df = df.dropna(subset=[target, 'INDFMPIR', 'RIDAGEYR'])

    # Keep only children
    df = df[df['RIDAGEYR'] <= 18.]

    # Create the domain split variable for poverty-income ratio
    df['INDFMPIRBelowCutoff'] = (df['INDFMPIR'] <= 1.3).astype(int)
    df.drop(columns=['INDFMPIR'])

    # Binarize the target
    df[target] = (df[target] >= threshold).astype(float)

    df = _postprocess_nhanes(df, features=features)
    return df
