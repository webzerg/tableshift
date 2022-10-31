import numpy as np
import pandas as pd

from tablebench.core.features import Feature, FeatureList, cat_dtype

CANDC_RESOURCES = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/communities/communities.data"]

# Column names of the raw input features from UCI.
CANDC_INPUT_FEATURES = [
    'state', 'county', 'community', 'communityname', 'fold', 'population',
    'householdsize', 'racepctblack', 'racePctWhite', 'racePctAsian',
    'racePctHisp', 'agePct12t21', 'agePct12t29', 'agePct16t24', 'agePct65up',
    'numbUrban', 'pctUrban', 'medIncome', 'pctWWage', 'pctWFarmSelf',
    'pctWInvInc', 'pctWSocSec', 'pctWPubAsst', 'pctWRetire', 'medFamInc',
    'perCapInc', 'whitePerCap', 'blackPerCap', 'indianPerCap', 'AsianPerCap',
    'OtherPerCap', 'HispPerCap', 'NumUnderPov', 'PctPopUnderPov',
    'PctLess9thGrade', 'PctNotHSGrad', 'PctBSorMore', 'PctUnemployed',
    'PctEmploy', 'PctEmplManu', 'PctEmplProfServ', 'PctOccupManu',
    'PctOccupMgmtProf', 'MalePctDivorce', 'MalePctNevMarr', 'FemalePctDiv',
    'TotalPctDiv', 'PersPerFam', 'PctFam2Par', 'PctKids2Par',
    'PctYoungKids2Par', 'PctTeen2Par', 'PctWorkMomYoungKids', 'PctWorkMom',
    'NumIlleg', 'PctIlleg', 'NumImmig', 'PctImmigRecent', 'PctImmigRec5',
    'PctImmigRec8', 'PctImmigRec10', 'PctRecentImmig', 'PctRecImmig5',
    'PctRecImmig8', 'PctRecImmig10', 'PctSpeakEnglOnly', 'PctNotSpeakEnglWell',
    'PctLargHouseFam', 'PctLargHouseOccup', 'PersPerOccupHous',
    'PersPerOwnOccHous', 'PersPerRentOccHous', 'PctPersOwnOccup',
    'PctPersDenseHous', 'PctHousLess3BR', 'MedNumBR', 'HousVacant',
    'PctHousOccup', 'PctHousOwnOcc', 'PctVacantBoarded', 'PctVacMore6Mos',
    'MedYrHousBuilt', 'PctHousNoPhone', 'PctWOFullPlumb', 'OwnOccLowQuart',
    'OwnOccMedVal', 'OwnOccHiQuart', 'RentLowQ', 'RentMedian', 'RentHighQ',
    'MedRent', 'MedRentPctHousInc', 'MedOwnCostPctInc',
    'MedOwnCostPctIncNoMtg', 'NumInShelters', 'NumStreet', 'PctForeignBorn',
    'PctBornSameState', 'PctSameHouse85', 'PctSameCity85', 'PctSameState85',
    'LemasSwornFT', 'LemasSwFTPerPop', 'LemasSwFTFieldOps',
    'LemasSwFTFieldPerPop', 'LemasTotalReq', 'LemasTotReqPerPop',
    'PolicReqPerOffic', 'PolicPerPop', 'RacialMatchCommPol', 'PctPolicWhite',
    'PctPolicBlack', 'PctPolicHisp', 'PctPolicAsian', 'PctPolicMinor',
    'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz', 'PolicAveOTWorked', 'LandArea',
    'PopDens', 'PctUsePubTrans', 'PolicCars', 'PolicOperBudg',
    'LemasPctPolicOnPatr', 'LemasGangUnitDeploy', 'LemasPctOfficDrugUn',
    'PolicBudgPerPop', 'ViolentCrimesPerPop'
]

CANDC_FEATURES = FeatureList([
    Feature('state', cat_dtype),
    Feature('communityname', cat_dtype),
    Feature('population', float),
    Feature('householdsize', float),
    Feature('agePct12t21', float),
    Feature('agePct12t29', float),
    Feature('agePct16t24', float),
    Feature('agePct65up', float),
    Feature('numbUrban', float),
    Feature('pctUrban', float),
    Feature('pctWWage', float),
    Feature('pctWFarmSelf', float),
    Feature('pctWInvInc', float),
    Feature('pctWSocSec', float),
    Feature('pctWPubAsst', float),
    Feature('pctWRetire', float),
    Feature('medFamInc', float),
    Feature('perCapInc', float),
    Feature('NumUnderPov', float),
    Feature('PctPopUnderPov', float),
    Feature('PctLess9thGrade', float),
    Feature('PctNotHSGrad', float),
    Feature('PctBSorMore', float),
    Feature('PctUnemployed', float),
    Feature('PctEmploy', float),
    Feature('PctEmplManu', float),
    Feature('PctEmplProfServ', float),
    Feature('PctOccupManu', float),
    Feature('PctOccupMgmtProf', float),
    Feature('MalePctDivorce', float),
    Feature('MalePctNevMarr', float),
    Feature('FemalePctDiv', float),
    Feature('TotalPctDiv', float),
    Feature('PersPerFam', float),
    Feature('PctFam2Par', float),
    Feature('PctKids2Par', float),
    Feature('PctYoungKids2Par', float),
    Feature('PctTeen2Par', float),
    Feature('PctWorkMomYoungKids', float),
    Feature('PctWorkMom', float),
    Feature('NumIlleg', float),
    Feature('PctIlleg', float),
    Feature('NumImmig', float),
    Feature('PctImmigRecent', float),
    Feature('PctImmigRec5', float),
    Feature('PctImmigRec8', float),
    Feature('PctImmigRec10', float),
    Feature('PctRecentImmig', float),
    Feature('PctRecImmig5', float),
    Feature('PctRecImmig8', float),
    Feature('PctRecImmig10', float),
    Feature('PctSpeakEnglOnly', float),
    Feature('PctNotSpeakEnglWell', float),
    Feature('PctLargHouseFam', float),
    Feature('PctLargHouseOccup', float),
    Feature('PersPerOccupHous', float),
    Feature('PersPerOwnOccHous', float),
    Feature('PersPerRentOccHous', float),
    Feature('PctPersOwnOccup', float),
    Feature('PctPersDenseHous', float),
    Feature('PctHousLess3BR', float),
    Feature('MedNumBR', float),
    Feature('HousVacant', float),
    Feature('PctHousOccup', float),
    Feature('PctHousOwnOcc', float),
    Feature('PctVacantBoarded', float),
    Feature('PctVacMore6Mos', float),
    Feature('MedYrHousBuilt', float),
    Feature('PctHousNoPhone', float),
    Feature('PctWOFullPlumb', float),
    Feature('OwnOccLowQuart', float),
    Feature('OwnOccMedVal', float),
    Feature('OwnOccHiQuart', float),
    Feature('RentLowQ', float),
    Feature('RentMedian', float),
    Feature('RentHighQ', float),
    Feature('MedRent', float),
    Feature('MedRentPctHousInc', float),
    Feature('MedOwnCostPctInc', float),
    Feature('MedOwnCostPctIncNoMtg', float),
    Feature('NumInShelters', float),
    Feature('NumStreet', float),
    Feature('PctForeignBorn', float),
    Feature('PctBornSameState', float),
    Feature('PctSameHouse85', float),
    Feature('PctSameCity85', float),
    Feature('PctSameState85', float),
    Feature('LandArea', float),
    Feature('PopDens', float),
    Feature('PctUsePubTrans', float),
    Feature('LemasPctOfficDrugUn', float),
    Feature('Target', int, is_target=True),
    Feature('Race', int),
    Feature('income_level_above_median', int)
])


def preprocess_candc(df: pd.DataFrame,
                     target_threshold: float = 0.08) -> pd.DataFrame:
    # remove non predicitive features
    # for c in ['state', 'county', 'community', 'communityname', 'fold']:
    #     if c != self.domain_split_colname:
    #         data.drop(columns=c, axis=1, inplace=True)

    df = df.replace('?', np.nan).dropna(axis=1)
    df["Race"] = df['racePctWhite'].apply(
        lambda x: 1 if x >= 0.85 else 0)
    income_thresh = df["medIncome"].median()
    df["income_level_above_median"] = df["medIncome"].apply(
        lambda x: 1 if x > income_thresh else 0)

    df = df.drop(columns=['racePctAsian', 'racePctHisp',
                          'racepctblack', 'whitePerCap',
                          'blackPerCap', 'indianPerCap',
                          'AsianPerCap',  # 'OtherPerCap',
                          'HispPerCap',
                          'racePctWhite', 'medIncome',
                          'fold',
                          ], axis=1).rename(
        columns={'ViolentCrimesPerPop': "Target"})

    # The label of a community is 1 if that community is among the
    # 70% of communities with the highest crime rate and 0 otherwise,
    # following Khani et al. and Kearns et al. 2018.
    df["Target"] = (df["Target"] >= target_threshold).astype(int)
    df["state"] = df["state"].apply(str).astype("category")
    return df.loc[:, CANDC_FEATURES.names]
