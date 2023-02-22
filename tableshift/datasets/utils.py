import pandas as pd


def convert_numeric_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Utility function for automatically casting int-valued columns to float."""
    for c in df.columns:
        df[c] = df[c].convert_dtypes(convert_string=False, convert_boolean=False)
    return df


def complete_cases(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=0, how='any')
