"""Data sources for TableBench."""
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
import glob
import logging
import os
import re
from typing import Sequence, Optional, Callable
import zipfile

import folktables
import pandas as pd

from tableshift.core import utils
from tableshift.datasets.acs import ACS_STATE_LIST, preprocess_acs, \
    get_acs_data_source, ACS_TASK_CONFIGS, acs_data_to_df
from tableshift.datasets.acs_feature_mappings import get_feature_mapping
from tableshift.datasets.adult import ADULT_RESOURCES, ADULT_FEATURE_NAMES, \
    preprocess_adult
from tableshift.datasets.anes import preprocess_anes
from tableshift.datasets.brfss import preprocess_brfss, align_brfss_features
from tableshift.datasets.communities_and_crime import CANDC_RESOURCES, \
    preprocess_candc, CANDC_INPUT_FEATURES
from tableshift.datasets.compas import COMPAS_RESOURCES, preprocess_compas
from tableshift.datasets.diabetes_readmission import \
    DIABETES_READMISSION_RESOURCES, preprocess_diabetes_readmission
from tableshift.datasets.german import GERMAN_RESOURCES, preprocess_german
from tableshift.datasets.heloc import preprocess_heloc
from tableshift.datasets.mimic_extract import preprocess_mimic_extract, \
    MIMIC_EXTRACT_STATIC_FEATURES
from tableshift.datasets.mooc import preprocess_mooc
from tableshift.datasets.nhanes import preprocess_nhanes_cholesterol, \
    get_nhanes_data_sources, preprocess_nhanes_lead, NHANES_YEARS
from tableshift.datasets.physionet import preprocess_physionet


class DataSource(ABC):
    """Abstract class to represent a generic data source."""

    def __init__(self, cache_dir: str,
                 preprocess_fn: Callable[[pd.DataFrame], pd.DataFrame],
                 resources: Sequence[str] = None,
                 download: bool = True,
                 ):
        self.cache_dir = cache_dir
        self.download = download

        self.preprocess_fn = preprocess_fn
        self.resources = resources
        self._initialize_cache_dir()

    def _initialize_cache_dir(self):
        """Create cache_dir if it does not exist."""
        utils.initialize_dir(self.cache_dir)

    def get_data(self) -> pd.DataFrame:
        """Fetch data from local cache or download if necessary."""
        self._download_if_not_cached()
        raw_data = self._load_data()
        return self.preprocess_fn(raw_data)

    def _download_if_not_cached(self):
        """Download files if they are not already cached."""
        for url in self.resources:
            utils.download_file(url, self.cache_dir)

    @abstractmethod
    def _load_data(self) -> pd.DataFrame:
        """Load the raw data from disk and return it.

        Any preprocessing should be performed in preprocess_fn, not here."""
        raise

    @property
    def is_cached(self) -> bool:
        """Check whether all resources exist in cache dir."""
        for url in self.resources:
            basename = utils.basename_from_url(url)
            fp = os.path.join(self.cache_dir, basename)
            if not os.path.exists(fp):
                return False
        return True


class OfflineDataSource(DataSource):

    def get_data(self) -> pd.DataFrame:
        raw_data = self._load_data()
        return self.preprocess_fn(raw_data)

    def _load_data(self) -> pd.DataFrame:
        raise


class ANESDataSource(OfflineDataSource):
    def __init__(
            self,
            years: Optional[Sequence] = None,
            preprocess_fn=preprocess_anes,
            resources=("anes_timeseries_cdf_csv_20220916/"
                       "anes_timeseries_cdf_csv_20220916.csv",),
            **kwargs):
        if years is not None:
            assert isinstance(years, list) or isinstance(years, tuple), \
                f"years must be a list or tuple, not type {type(years)}."
        self.years = years
        super().__init__(resources=resources,
                         preprocess_fn=preprocess_fn,
                         **kwargs)

    def _load_data(self) -> pd.DataFrame:
        fp = os.path.join(self.cache_dir, self.resources[0])
        df = pd.read_csv(fp, low_memory=False, na_values=(' '))
        if self.years:
            df = df[df["VCF0004"].isin(self.years)]
        return df


class MOOCDataSource(OfflineDataSource):
    def __init__(
            self,
            preprocess_fn=preprocess_mooc,
            resources=(os.path.join("dataverse_files",
                                    "HXPC13_DI_v3_11-13-2019.csv"),),
            **kwargs):
        super().__init__(resources=resources,
                         preprocess_fn=preprocess_fn,
                         **kwargs)

    def _load_data(self) -> pd.DataFrame:
        fp = os.path.join(self.cache_dir, self.resources[0])
        if not os.path.exists(fp):
            raise RuntimeError(
                f"""Data files does not exist at {fp}. This dataset must be 
                manually downloaded. Visit https://doi.org/10.7910/DVN/26147,
                click 'Access Dataset' > 'Original Format ZIP', download the ZIP
                file to the cache directory at {self.cache_dir}, and 
                unzip it.""")
        df = pd.read_csv(fp)
        return df


class KaggleDataSource(DataSource):
    def __init__(
            self,
            kaggle_dataset_name: str,
            kaggle_creds_dir="~/.kaggle",
            **kwargs):
        self.kaggle_creds_dir = kaggle_creds_dir
        self.kaggle_dataset_name = kaggle_dataset_name
        super().__init__(**kwargs)

    @property
    def kaggle_creds(self):
        return os.path.expanduser(
            os.path.join(self.kaggle_creds_dir, "kaggle.json"))

    @property
    def zip_file_name(self):
        """Name of the zip file downloaded by Kaggle API."""
        return os.path.basename(self.kaggle_dataset_name) + ".zip"

    @abstractmethod
    def _load_data(self) -> pd.DataFrame:
        # Should be implemented by subclasses.
        raise

    def _download_kaggle_data(self):
        """Download the data from Kaggle."""
        # Check Kaggle authentication.
        assert os.path.exists(self.kaggle_creds), \
            f"No kaggle credentials found at {self.kaggle_creds}."
        "Create an access token at https://www.kaggle.com/YOUR_USERNAME/account"
        f"and store it at {self.kaggle_creds}."

        # Download the dataset using Kaggle CLI.
        cmd = "kaggle datasets download " \
              f"-d {self.kaggle_dataset_name} " \
              f"-p {self.cache_dir}"
        utils.run_in_subproces(cmd)
        return

    def _download_if_not_cached(self):
        self._download_kaggle_data()
        # location of the local zip file
        zip_fp = os.path.join(self.cache_dir, self.zip_file_name)
        # where to unzip the file to
        unzip_dest = os.path.join(self.cache_dir, self.kaggle_dataset_name)
        with zipfile.ZipFile(zip_fp, 'r') as zf:
            zf.extractall(unzip_dest)


class BRFSSDataSource(DataSource):
    """BRFSS data source.

    Note that the BRFSS is composed of three components: 'fixed core' questions,
    asked every year, 'rotating core', asked every other year, and 'emerging
    core'. Since some of our features come from the rotating core, we only
    use every-other-year data sources; some features would be empty for the
    intervening years.

    See also https://www.cdc.gov/brfss/about/brfss_faq.htm , "What are the
    components of the BRFSS questionnaire?"
    """

    def __init__(self, task: str, preprocess_fn=preprocess_brfss,
                 years=(2021,), **kwargs):
        self.years = years
        resources = tuple([
            f"https://www.cdc.gov/brfss/annual_data/{y}/files/LLCP{y}XPT.zip"
            for y in self.years])
        _preprocess_fn = partial(preprocess_fn, task=task)
        super().__init__(preprocess_fn=_preprocess_fn, resources=resources,
                         **kwargs)

    def _load_data(self) -> pd.DataFrame:
        dfs = {}
        for url in self.resources:
            zip_fname = utils.basename_from_url(url)
            xpt_fname = zip_fname.replace("XPT.zip", ".XPT")
            xpt_fp = os.path.join(self.cache_dir, xpt_fname)
            # Unzip the file if needed
            if not os.path.exists(xpt_fp):
                zip_fp = os.path.join(self.cache_dir, zip_fname)
                logging.debug(f"unzipping {zip_fp}")
                with zipfile.ZipFile(zip_fp, 'r') as zf:
                    zf.extractall(self.cache_dir)
                # BRFSS data files have an awful space at the end; remove it.
                os.rename(xpt_fp + " ", xpt_fp)
            # Read the XPT data
            logging.debug(f"reading {xpt_fp}")
            df = utils.read_xpt(xpt_fp)
            df = align_brfss_features(df)
            dfs[url] = df

        return pd.concat(dfs.values(), axis=0)


class NHANESDataSource(DataSource):
    def __init__(
            self,
            nhanes_task: str,
            years: Sequence[int] = NHANES_YEARS,
            **kwargs):
        self.nhanes_task = nhanes_task
        self.years = years

        if self.nhanes_task == "cholesterol":
            preprocess_fn = preprocess_nhanes_cholesterol
        elif self.nhanes_task == "lead":
            preprocess_fn = preprocess_nhanes_lead
        else:
            raise ValueError

        super().__init__(preprocess_fn=preprocess_fn,
                         **kwargs)

    def _download_if_not_cached(self):

        def _add_suffix_to_fname_from_url(url: str, suffix: str):
            """Helper function to add unique names to files by year."""
            fname = utils.basename_from_url(url)
            f, extension = fname.rsplit(".")
            new_fp = f + suffix + "." + extension
            return new_fp

        sources = get_nhanes_data_sources(self.nhanes_task, self.years)
        for year, urls in sources.items():
            for url in urls:
                destfile = _add_suffix_to_fname_from_url(url, str(year))
                utils.download_file(url, self.cache_dir,
                                    dest_file_name=destfile)

    def _load_data(self) -> pd.DataFrame:
        files = glob.glob(os.path.join(self.cache_dir, "*.XPT"))

        # Initialize a dict where keys are years, and values are lists
        # containing the list of dataframes of data for that year; these
        # can be joined on their index (not sure whether the index is
        # unique across years).
        year_dfs = defaultdict(list)

        for f in files:
            year = int(re.search(".*([0-9]{4})\\.XPT", f).group(1))
            if year in self.years:
                logging.debug(f"reading {f}")
                df = utils.read_xpt(f)
                try:
                    df.set_index("SEQN", inplace=True)
                except KeyError:
                    # Some LLCP files only contain 'SEQNO' feature.
                    df.set_index("SEQNO", inplace=True)
                year_dfs[year].append(df)

        df_list = []
        for year in year_dfs.keys():
            # Join the first dataframe with all others.
            dfs = year_dfs[year]
            src_df = dfs[0]
            try:
                logging.info(f"joining {len(dfs)} dataframes for {year}")
                df = src_df.join(dfs[1:], how="outer")
                df["nhanes_year"] = int(year)
                logging.info("finished joins")
                df_list.append(df)
            except Exception as e:
                logging.error(e)

        if len(df_list) > 1:
            df = pd.concat(df_list, axis=0)
        else:
            df = df_list[0]

        return df


class ACSDataSource(DataSource):
    def __init__(self,
                 acs_task: str,
                 preprocess_fn=preprocess_acs,
                 years: Sequence[int] = (2018,),
                 states=ACS_STATE_LIST,
                 feature_mapping="coarse",
                 **kwargs):
        self.acs_task = acs_task.lower().replace("acs", "")
        self.feature_mapping = get_feature_mapping(feature_mapping)
        self.states = states
        self.years = years
        super().__init__(preprocess_fn=preprocess_fn, **kwargs)

    def _get_acs_data(self):
        year_dfs = []

        for year in self.years:
            logging.info(f"fetching ACS data for year {year}...")
            data_source = get_acs_data_source(year, self.cache_dir)
            year_data = data_source.get_data(states=self.states,
                                             join_household=True,
                                             download=True)
            year_data["ACS_YEAR"] = year
            year_dfs.append(year_data)
        logging.info("fetching ACS data complete.")
        return pd.concat(year_dfs, axis=0)

    def _download_if_not_cached(self):
        """No-op for ACS data; folktables already downloads or uses cache as
        needed at _load_data(). """
        return

    def _load_data(self) -> pd.DataFrame:
        acs_data = self._get_acs_data()
        task_config = ACS_TASK_CONFIGS[self.acs_task]
        target_transform = partial(task_config.target_transform,
                                   threshold=task_config.threshold)
        ACSProblem = folktables.BasicProblem(
            features=task_config.features_to_use.predictors,
            target=task_config.target,
            target_transform=target_transform,
            preprocess=task_config.preprocess,
            postprocess=task_config.postprocess,
        )
        X, y, _ = ACSProblem.df_to_numpy(acs_data)
        df = acs_data_to_df(X, y, task_config.features_to_use,
                            feature_mapping=self.feature_mapping)
        return df


class AdultDataSource(DataSource):
    """Data source for the Adult dataset."""

    def __init__(self, resources=ADULT_RESOURCES,
                 preprocess_fn=preprocess_adult, **kwargs):
        super().__init__(resources=resources,
                         preprocess_fn=preprocess_fn, **kwargs)

    def _load_data(self) -> pd.DataFrame:
        train_fp = os.path.join(self.cache_dir, "adult.data")
        train = pd.read_csv(
            train_fp,
            names=ADULT_FEATURE_NAMES,
            sep=r'\s*,\s*',
            engine='python', na_values="?")
        train["Split"] = "train"

        test_fp = os.path.join(self.cache_dir, "adult.test")

        test = pd.read_csv(
            test_fp,
            names=ADULT_FEATURE_NAMES,
            sep=r'\s*,\s*',
            engine='python', na_values="?", skiprows=1)
        test["Split"] = "test"

        return pd.concat((train, test))


class COMPASDataSource(DataSource):
    def __init__(self, resources=COMPAS_RESOURCES,
                 preprocess_fn=preprocess_compas, **kwargs):
        super().__init__(resources=resources,
                         preprocess_fn=preprocess_fn, **kwargs)

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(self.cache_dir, "compas-scores-two-years.csv"))
        return df


class GermanDataSource(DataSource):
    def __init__(self, resources=GERMAN_RESOURCES,
                 preprocess_fn=preprocess_german, **kwargs):
        super().__init__(resources=resources, preprocess_fn=preprocess_fn,
                         **kwargs)

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.cache_dir, "german.data"),
                         sep=" ", header=None)
        return df


class DiabetesReadmissionDataSource(DataSource):
    def __init__(self, resources=DIABETES_READMISSION_RESOURCES,
                 preprocess_fn=preprocess_diabetes_readmission, **kwargs):
        super().__init__(resources=resources, preprocess_fn=preprocess_fn,
                         **kwargs)

    def _load_data(self) -> pd.DataFrame:
        # unzip the file
        zip_fp = os.path.join(self.cache_dir, "dataset_diabetes.zip")
        with zipfile.ZipFile(zip_fp, 'r') as zf:
            zf.extractall(self.cache_dir)
        # read the dataframe
        df = pd.read_csv(os.path.join(self.cache_dir, "dataset_diabetes",
                                      "diabetic_data.csv"),
                         na_values="?")
        return df


class CommunitiesAndCrimeDataSource(DataSource):
    def __init__(self, resources=CANDC_RESOURCES,
                 preprocess_fn=preprocess_candc, **kwargs):
        super().__init__(resources=resources, preprocess_fn=preprocess_fn,
                         **kwargs)

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.cache_dir, 'communities.data'),
                         names=CANDC_INPUT_FEATURES)
        return df


class PhysioNetDataSource(DataSource):
    def __init__(self, preprocess_fn=preprocess_physionet, **kwargs):
        super().__init__(preprocess_fn=preprocess_fn, **kwargs)

    def _download_if_not_cached(self):
        # check if correct number of training files exist in cache dir
        root = os.path.join(self.cache_dir, "physionet.org", "files",
                            "challenge-2019", "1.0.0", "training")
        n_train_a = len(glob.glob(os.path.join(root, "training_setA", "*.psv")))
        n_train_b = len(glob.glob(os.path.join(root, "training_setB", "*.psv")))

        if (not n_train_a == 20336) or (not n_train_b == 20000):
            logging.info("downloading physionet training data. This could "
                         "take several minutes.")
            # download the training data
            cmd = "wget -r -N -c -np https://physionet.org/files/challenge" \
                  f"-2019/1.0.0/training/ -P={self.cache_dir}"
            utils.run_in_subproces(cmd)
        return

    def _load_data(self) -> pd.DataFrame:
        root = os.path.join(self.cache_dir, "physionet.org", "files",
                            "challenge-2019", "1.0.0", "training")
        logging.info("reading physionet data files.")
        train_a_files = glob.glob(os.path.join(root, "training_setA", "*.psv"))
        df_a = pd.concat(pd.read_csv(x, delimiter="|") for x in train_a_files)
        train_b_files = glob.glob(os.path.join(root, "training_setB", "*.psv"))
        df_b = pd.concat(pd.read_csv(x, delimiter="|") for x in train_b_files)
        logging.info("done reading physionet data files.")
        df_a["set"] = "a"
        df_b["set"] = "b"
        df = pd.concat((df_a, df_b))
        df.reset_index(drop=True, inplace=True)
        return df


class MIMICExtractDataSource(OfflineDataSource):

    def __init__(self, task: str = "los_3",
                 preprocess_fn=preprocess_mimic_extract,
                 static_features=MIMIC_EXTRACT_STATIC_FEATURES.names, **kwargs):
        # Note: mean label values in overall dataset:
        # mort_hosp 0.106123
        # mort_icu 0.071709
        # los_3 0.430296
        # los_7 0.077055

        if task not in ('mort_hosp', 'mort_icu', 'los_3', 'los_7'):
            raise NotImplementedError(f"task {task} is not implemented.")
        self.task = task
        self.static_features = static_features
        _preprocess_fn = partial(preprocess_fn, task=task,
                                 static_features=self.static_features)
        super().__init__(**kwargs, preprocess_fn=_preprocess_fn)

    def _load_data(self, gap_time_hrs=6, window_size_hrs=24) -> pd.DataFrame:
        """Load the data and apply any shared MIMIC-extract preprocessing
        with default parameters."""

        filename = os.path.join(self.cache_dir, 'all_hourly_data.h5')
        assert os.path.exists(
            filename), \
            f"""file {filename} does not exist; see the TableShift 
            instructions for  accessing/placing the MIMIC-extract dataset at 
            the expected location. The data file can be accessed at 
            https://storage.googleapis.com/mimic_extract/all_hourly_data.h5 
            after  obtaining access as described at 
            https://github.com/MLforHealth/MIMIC_Extract"""
        data_full_lvl2 = pd.read_hdf(filename, 'vitals_labs')
        statics = pd.read_hdf(filename, 'patients')

        # Extract/compute labels, retaining only the labels for the current task.
        Ys = statics[statics.max_hours > window_size_hrs + gap_time_hrs][
            ['mort_hosp', 'mort_icu', 'los_icu']]
        Ys['los_3'] = Ys['los_icu'] > 3
        Ys['los_7'] = Ys['los_icu'] > 7
        Ys = Ys[[self.task]]
        Ys = Ys.astype(int)

        # MIMIC-default filtering: keep only those observations where labels are known; keep
        # only those observations within the window size.
        lvl2 = data_full_lvl2[
            (data_full_lvl2.index.get_level_values('icustay_id').isin(
                set(Ys.index.get_level_values('icustay_id')))) &
            (data_full_lvl2.index.get_level_values(
                'hours_in') < window_size_hrs)]

        # Join data with labels and static features.
        df_out = lvl2.join(Ys, how="inner")
        df_out = df_out.join(statics[MIMIC_EXTRACT_STATIC_FEATURES.names])
        assert len(df_out) == len(lvl2), "Sanity check of labels join."
        return df_out


class HELOCDataSource(OfflineDataSource):
    """FICO Home Equity Line of Credit data source.

    To obtain data access, visit
    https://community.fico.com/s/explainable-machine-learning-challenge
    """

    def __init__(self, preprocess_fn=preprocess_heloc, **kwargs):
        super().__init__(preprocess_fn=preprocess_fn, **kwargs)

    def _load_data(self) -> pd.DataFrame:
        filename = os.path.join(self.cache_dir, "heloc_dataset_v1.csv")
        assert os.path.exists(
            filename), \
            f"""file {filename} does not exist; see the TableShift 
            instructions for  accessing/placing the FICO HELOC dataset at the 
            expected location. The data file can be accessed by filling out the
            data access agreement at
            https://community.fico.com/s/explainable-machine-learning-challenge
            """
        return pd.read_csv(filename)


class MetaMIMICDataSource(OfflineDataSource):
    """MetaMIMIC data source.

    The dataset must be manually derived from MIMIC using the scripts
    provided in https://github.com/ModelOriented/metaMIMIC .
    """

    def __init__(self, preprocess_fn=lambda x: x, **kwargs):
        super().__init__(preprocess_fn=preprocess_fn, **kwargs)

    def _load_data(self) -> pd.DataFrame:
        filename = os.path.join(self.cache_dir, "metaMIMIC.csv")
        assert os.path.exists(filename), \
            f"file {filename} does not exist. Ensure you have constructed the " \
            f"metaMIMIC dataset as described at " \
            f"https://github.com/ModelOriented/metaMIMIC and placed the " \
            f"resulting file at {filename} ."
        return pd.read_csv(filename)
