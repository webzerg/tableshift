from tablebench.core.features import Feature, FeatureList, cat_dtype

DIABETES_READMISSION_RESOURCES = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00296"
    "/dataset_diabetes.zip"
]

DIABETES_READMISSION_FEATURES = FeatureList(features=[
    Feature('race', cat_dtype, """Nominal. Values: Caucasian, Asian, African 
    American, Hispanic, and other"""),
    Feature('gender', cat_dtype, """Nominal. Values: male, female, and 
    unknown/invalid."""),
    Feature('age', cat_dtype, """Nominal. Grouped in 10-year intervals: [0, 
    10), [10, 20), . . ., [90, 100)"""),
    Feature('weight', float, "Weight in pounds."),
    Feature('admission_type_id', float, """Integer identifier corresponding 
    to 9 distinct values, for example, emergency, urgent, elective, newborn, 
    and not available"""),
    Feature('discharge_disposition_id', float, """Integer identifier 
    corresponding to 29 distinct values, for example, discharged to home, 
    expired, and not available"""),
    Feature('admission_source_id', int, """Integer identifier corresponding to 
    21 distinct values, for example, physician referral, emergency room, 
    and transfer from a hospital"""),
    Feature('time_in_hospital', float),
    Feature('payer_code', cat_dtype),
    Feature('medical_specialty', cat_dtype),
    Feature('num_lab_procedures', float),
    Feature('num_procedures', float),
    Feature('num_medications', float),
    Feature('number_outpatient', float),
    Feature('number_emergency', float),
    Feature('number_inpatient', float),
    Feature('diag_1', cat_dtype),
    Feature('diag_2', cat_dtype),
    Feature('diag_3', cat_dtype),
    Feature('number_diagnoses', float),
    Feature('max_glu_serum', cat_dtype),
    Feature('A1Cresult', cat_dtype),
    Feature('metformin', cat_dtype),
    Feature('repaglinide', cat_dtype),
    Feature('nateglinide', cat_dtype),
    Feature('chlorpropamide', cat_dtype),
    Feature('glimepiride', cat_dtype),
    Feature('acetohexamide', cat_dtype),
    Feature('glipizide', cat_dtype),
    Feature('glyburide', cat_dtype),
    Feature('tolbutamide', cat_dtype),
    Feature('pioglitazone', cat_dtype),
    Feature('rosiglitazone', cat_dtype),
    Feature('acarbose', cat_dtype),
    Feature('miglitol', cat_dtype),
    Feature('troglitazone', cat_dtype),
    Feature('tolazamide', cat_dtype),
    Feature('examide', cat_dtype),
    Feature('citoglipton', cat_dtype),
    Feature('insulin', cat_dtype),
    Feature('glyburide-metformin', cat_dtype),
    Feature('glipizide-metformin', cat_dtype),
    Feature('glimepiride-pioglitazone', cat_dtype),
    Feature('metformin-rosiglitazone', cat_dtype),
    Feature('metformin-pioglitazone', cat_dtype),
    Feature('change', cat_dtype),
    Feature('diabetesMed', cat_dtype),
    # Converted to binary (readmit vs. no readmit).
    Feature('readmitted', float, is_target=True),
])


def preprocess_diabetes_readmission(df):
    df = df.loc[:, DIABETES_READMISSION_FEATURES.names]
    tgt_col = DIABETES_READMISSION_FEATURES.target
    df[tgt_col] = (df[tgt_col] != "NO").astype(float)
    df.rename(columns={DIABETES_READMISSION_FEATURES.target: "Target"},
              inplace=True)
    return df
