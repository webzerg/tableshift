import pandas as pd
from tablebench.core.features import Feature, FeatureList, cat_dtype

DIABETES_READMISSION_RESOURCES = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00296"
    "/dataset_diabetes.zip"
]

# Common feature description for the 24 features for medications
DIABETES_MEDICATION_FEAT_DESCRIPTION = """Indicates if there was any diabetic 
    medication prescribed. Values: 'yes' and 'no' For the generic names: 
    metformin, repaglinide, nateglinide, chlorpropamide, glimepiride, 
    acetohexamide, glipizide, glyburide, tolbutamide, pioglitazone, 
    rosiglitazone, acarbose, miglitol, troglitazone, tolazamide, examide, 
    sitagliptin, insulin, glyburide-metformin, glipizide-metformin, 
    glimepiride-pioglitazone, metformin-rosiglitazone, 
    and metformin-pioglitazone, the feature indicates whether the drug was 
    prescribed or there was a change in the dosage. Values: 'up' if the 
    dosage was increased during the encounter, 'down' if the dosage was 
    decreased, 'steady' if the dosage did not change, and 'no' if the drug 
    was not prescribed."""

# Note: the UCI version of this dataset does *not* exactly correspond to the
# version documented in the linked paper. For example, in the paper, 'weight'
# is described as a numeric feature, but is discretized into bins in UCI;
# similarly, many fields are marked as having much higher missing value counts
# in the paper than are present in the UCI data. The cause of this discrepancy
# is not clear.
DIABETES_READMISSION_FEATURES = FeatureList(features=[
    Feature('race', cat_dtype, """Nominal. Values: Caucasian, Asian, African 
    American, Hispanic, and other"""),
    Feature('gender', cat_dtype, """Nominal. Values: male, female, and 
    unknown/invalid."""),
    Feature('age', cat_dtype, """Nominal. Grouped in 10-year intervals: [0, 
    10), [10, 20), . . ., [90, 100)"""),
    Feature('weight', cat_dtype, "Weight in pounds. Grouped in 25-pound "
                                 "intervals."),
    Feature('admission_type_id', float, """Integer identifier corresponding 
    to 9 distinct values, for example, emergency, urgent, elective, newborn, 
    and not available"""),
    Feature('discharge_disposition_id', float, """Integer identifier 
    corresponding to 29 distinct values, for example, discharged to home, 
    expired, and not available"""),
    Feature('admission_source_id', int, """Integer identifier corresponding to 
    21 distinct values, for example, physician referral, emergency room, 
    and transfer from a hospital"""),
    Feature('time_in_hospital', float, "Integer number of days between "
                                       "admission and discharge"),
    Feature('payer_code', cat_dtype, "Integer identifier corresponding to 23 "
                                     "distinct values, for example, "
                                     "Blue Cross\Blue Shield, Medicare, "
                                     "and self-pay"),
    Feature('medical_specialty', cat_dtype, "Integer identifier of a "
                                            "specialty of the admitting "
                                            "physician, corresponding to 84 "
                                            "distinct values, for example, "
                                            "cardiology, internal medicine, "
                                            "family\general practice, "
                                            "and surgeon"),
    Feature('num_lab_procedures', float, "Number of lab tests performed "
                                         "during the encounter"),
    Feature('num_procedures', float, "Number of procedures (other than lab "
                                     "tests) performed during the encounter"),
    Feature('num_medications', float, "Number of distinct generic names "
                                      "administered during the encounter"),
    Feature('number_outpatient', float, "Number of outpatient visits of the "
                                        "patient in the year preceding the "
                                        "encounter"),
    Feature('number_emergency', float, "Number of emergency visits of the "
                                       "patient in the year preceding the "
                                       "encounter"),
    Feature('number_inpatient', float, "Number of inpatient visits of the "
                                       "patient in the year preceding the "
                                       "encounter"),
    Feature('diag_1', cat_dtype, "The primary diagnosis (coded as first three "
                                 "digits of ICD9); 848 distinct values"),
    Feature('diag_2', cat_dtype, "Secondary diagnosis (coded as first three "
                                 "digits of ICD9); 923 distinct values"),
    Feature('diag_3', cat_dtype, "Additional secondary diagnosis (coded as "
                                 "first three digits of ICD9); 954 distinct "
                                 "values"),
    Feature('number_diagnoses', float, "Number of diagnoses entered to the "
                                       "system"),
    Feature('max_glu_serum', cat_dtype, "Indicates the range of the result or "
                                        "if the test was not taken. Values: "
                                        "'>200,' '>300,' 'normal,' and 'none' "
                                        "if not measured"),
    Feature('A1Cresult', cat_dtype, "Indicates the range of the result or if "
                                    "the test was not taken. Values: '>8' if "
                                    "the result was greater than 8%, '>7' if "
                                    "the result was greater than 7% but less "
                                    "than 8%, 'normal' if the result was less "
                                    "than 7%, and 'none' if not measured."),
    Feature('metformin', cat_dtype, "Indicates if there was a change in "
                                    "diabetic medications (either dosage or "
                                    "generic name). Values: 'change' and 'no "
                                    "change'"),
    Feature('repaglinide', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION),
    Feature('nateglinide', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION),
    Feature('chlorpropamide', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION),
    Feature('glimepiride', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION),
    Feature('acetohexamide', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION),
    Feature('glipizide', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION),
    Feature('glyburide', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION),
    Feature('tolbutamide', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION),
    Feature('pioglitazone', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION),
    Feature('rosiglitazone', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION),
    Feature('acarbose', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION),
    Feature('miglitol', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION),
    Feature('troglitazone', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION),
    Feature('tolazamide', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION),
    Feature('examide', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION),
    Feature('citoglipton', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION),
    Feature('insulin', cat_dtype, DIABETES_MEDICATION_FEAT_DESCRIPTION),
    Feature('glyburide-metformin', cat_dtype,
            DIABETES_MEDICATION_FEAT_DESCRIPTION),
    Feature('glipizide-metformin', cat_dtype,
            DIABETES_MEDICATION_FEAT_DESCRIPTION),
    Feature('glimepiride-pioglitazone', cat_dtype,
            DIABETES_MEDICATION_FEAT_DESCRIPTION),
    Feature('metformin-rosiglitazone', cat_dtype,
            DIABETES_MEDICATION_FEAT_DESCRIPTION),
    Feature('metformin-pioglitazone', cat_dtype,
            DIABETES_MEDICATION_FEAT_DESCRIPTION),
    Feature('change', cat_dtype, "Indicates if there was a change in diabetic "
                                 "medications (either dosage or generic "
                                 "name). Values: 'change' and 'no change'"),
    Feature('diabetesMed', cat_dtype, "Indicates if there was any diabetic "
                                      "medication prescribed. Values: 'yes' "
                                      "and 'no'"),
    # Converted to binary (readmit vs. no readmit).
    Feature('readmitted', float, "30 days, '>30' if the patient was "
                                 "readmitted in more than 30 days, and 'No' "
                                 "for no record of readmission.",
            is_target=True),
], documentation="http://www.hindawi.com/journals/bmri/2014/781670/")


def preprocess_diabetes_readmission(df: pd.DataFrame):
    # Drop 2273 obs with missing race (2.2336% of total data)
    df.dropna(subset=["race"], inplace=True)

    tgt_col = DIABETES_READMISSION_FEATURES.target
    df[tgt_col] = (df[tgt_col] != "NO").astype(float)

    # Some columns contain a small fraction of missing values (~1%); fill them.
    df.fillna("MISSING")
    return df
