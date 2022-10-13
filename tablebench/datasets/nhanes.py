from collections import defaultdict
import json
import os

import pandas as pd

from tablebench.core.features import Feature, FeatureList, cat_dtype

# Dictionary mapping years to data sources. Because NHANES uses the same
# name for each file, we need to manually track the year associated with
# each dataset.
NHANES_DATA_SOURCES = os.path.join(os.path.dirname(__file__),
                                   "nhanes_data_sources.json")

# Mapping of NHANES component types to names of data sources to use.
# See nhanes_data_sources.json. This ensures that only needed files
# are downloaded/read from disk, because NHANES contains a huge number of sources per year.
NHANES_DATA_SOURCES_TO_USE = {
    "Demographics": ["Demographic Variables and Sample Weights"],
    "Laboratory": ["Cholesterol - LDL & Triglycerides"],
    "Questionnaire": ["Blood Pressure & Cholesterol",
                      "Cardiovascular Health",
                      "Diabetes",
                      "Kidney Conditions - Urology",
                      "Medical Conditions",
                      "Osteoporosis", ]
}


def get_nhanes_data_sources():
    """Fetch a mapping of {year: list of urls} for NHANES."""
    output = defaultdict(list)
    with open(NHANES_DATA_SOURCES, "r") as f:
        data_sources = json.load(f)
    for year, components in data_sources.items():
        for component, sources in components.items():
            for source_name, source_url in sources.items():
                if source_name in NHANES_DATA_SOURCES_TO_USE[component]:
                    output[year].append(source_url)
    return output


NHANES_DEMOG_FEATURES = FeatureList(features=[
    # In what country {were you/was SP} born?
    Feature('DMDBORN4', cat_dtype),

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

    # Total family income (reported as a range value in dollars)
    Feature('INDFMIN2', cat_dtype),

    # A ratio of family income to poverty guidelines.
    Feature('INDFMPIR', float),

    # Recode of reported race and Hispanic origin information,
    # with Non-Hispanic Asian Category
    Feature('RIDRETH3', cat_dtype),
])

NHANES_CHOLESTEROL_FEATURES = FeatureList(features=[
    # Target: Direct LDL-Cholesterol (mg/dL). We use a threshold of 160mg/DL,
    # based on the definition of Primary hypercholestemia in
    # Blood Cholesterol: Executive Summary: A Report of the American College
    # of Cardiology/American Heart Association Task Force on Clinical
    # Practice Guidelines, DOI: 10.1161/CIR.0000000000000624 (cf. Table 6):
    # "Primary hypercholesterolemia (LDL-C, 160–189 mg/dL)".
    # We use the LBDLDL - LDL-Cholesterol, Friedewald (mg/dL) measurement,
    # since the other measurement
    # (LBDLDLM - LDL-Cholesterol, Martin-Hopkins (mg/dL))
    # is not available for all years and the two have very strong correlation
    # i.e. see this study: https://pubmed.ncbi.nlm.nih.gov/34881700/
    Feature('Target', float),  # Raw feature for this is 'LBDLDL'.

    # Below we use the additional set of risk factors listed in the above report
    # (Table 6) **which can be asked in a questionnaire** (i.e. those which
    # do not require laboratory testing).

    ####### Risk Factor: Family history of ASCVD

    # {Have you/Has SP} ever been told by a doctor or other health professional
    # that {you have/s/he has} health conditions or a medical or family history
    # that increases {your/his/her} risk for diabetes?
    # (Note: no other questions about family health history, so we use this
    # one, even though it is about diabetes).)
    Feature('DIQ170', cat_dtype),

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

    # {Have you/Has SP} ever been told by a doctor or other health care
    # professional that {you/s/he} had psoriasis (sore-eye-asis)?
    Feature('MCQ070', cat_dtype),

    # Has a doctor or other health professional ever told {you/SP} that
    # {you/s/he} . . .had arthritis (ar-thry-tis)?
    Feature('MCQ160B', cat_dtype),

    # Note: no questions about HIV/AIDS.

    #######  Risk Factor: History of premature menopause (before age 40 y)
    # and history of pregnancy-associated conditions that increase later
    #  ASCVD risk such as preeclampsia

    # Note: no questions on these.

    #######  Risk Factor: High-risk race/ethnicities (eg, South Asian ancestry)
    # Recode of reported race and Hispanic origin information,
    # with Non-Hispanic Asian Category
    Feature('RIDRETH3', cat_dtype),
])


def preprocess_nhanes_cholesterol(df: pd.DataFrame, threshold=160.):
    input_features = list(set(NHANES_CHOLESTEROL_FEATURES.names +
                              NHANES_DEMOG_FEATURES.names))
    input_features.remove("Target")
    input_features.insert(0, "LBDLDL")
    for f in input_features:
        assert f in df.columns, f"data is missing column {f}"
    df = df.loc[:, input_features]

    # Drop observations with missing target
    df = df.dropna(subset=["LBDLDL"])
    # Binarize the target
    df["Target"] = (df["LBDLDL"] >= threshold).astype(float)
    df.drop(columns="LBDLDL", inplace=True)

    # All features are categorical; we can fill missing values with "missing".
    output_feature_list = NHANES_CHOLESTEROL_FEATURES + NHANES_DEMOG_FEATURES
    for feature in output_feature_list:
        name = feature.name
        if name != "Target" and feature.kind == cat_dtype:
            df[name] = df[name].fillna("MISSING").apply(str).astype("category")

    df.reset_index(drop=True, inplace=True)
    return df
