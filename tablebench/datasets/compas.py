import pandas as pd

from tablebench.core.features import Feature, FeatureList

COMPAS_RESOURCES = [
    "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"]

COMPAS_FEATURES = FeatureList(features=[
    Feature('juv_fel_count', int),
    Feature('juv_misd_count', int),
    Feature('juv_other_count', int),
    Feature('priors_count', int),
    Feature('age', int),
    Feature('c_charge_degree', int),
    Feature('sex', str),
    Feature('race', str),
    Feature('Target', str),
])


def preprocess_compas(df: pd.DataFrame):
    """Preprocess COMPAS dataset.

    See https://github.com/RuntianZ/doro/blob/master/compas.py .
    """

    columns = ['juv_fel_count', 'juv_misd_count', 'juv_other_count',
               'priors_count',
               'age',
               'c_charge_degree',
               'sex', 'race', 'is_recid', 'compas_screening_date']

    df = df[['id'] + columns].drop_duplicates()
    df = df[columns]

    # race_dict = {'African-American': 1, 'Caucasian': 0}
    # df['race'] = df.apply(
    #     lambda x: race_dict[x['race']] if x['race'] in race_dict.keys() else 2,
    #     axis=1).astype(
    #     'category')

    # # Screening dates are either in year 2013, or 2014.
    # df['screening_year_is_2013'] = df['compas_screening_date'].apply(
    #     lambda x: int(datetime.strptime(x, "%Y-%m-%d").year == 2013))
    # df.drop(columns=['compas_screening_date'], inplace=True)

    # sex_map = {'Female': 0, 'Male': 1}
    # df['sex'] = df['sex'].map(sex_map)
    #
    # c_charge_degree_map = {'F': 0, 'M': 1}
    # df['c_charge_degree'] = df['c_charge_degree'].map(c_charge_degree_map)
    df.rename(columns={"is_recid": "Target"}, inplace=True)
    return df
