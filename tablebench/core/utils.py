import collections
from itertools import islice
import os
import requests
import subprocess
import urllib.parse

import pandas as pd
import xport.v56

from .splitter import Splitter, DomainSplitter


def initialize_dir(dir: str):
    """Create a directory if it does not exist."""
    if not os.path.exists(dir):
        os.makedirs(dir)


def basename_from_url(url: str) -> str:
    parsed_url = urllib.parse.urlparse(url)
    fname = os.path.basename(parsed_url.path)
    return fname


def download_file(url: str, dirpath: str, if_not_exist=True,
                  dest_file_name=None):
    """Download file to the specified directory."""
    if dest_file_name:
        fname = dest_file_name
    else:
        fname = basename_from_url(url)
    fpath = os.path.join(dirpath, fname)

    if os.path.exists(fpath) and if_not_exist:
        # Case: file already exists; skip it.
        print(f"[DEBUG] not downloading {url}; exists at {fpath}")

    else:
        initialize_dir(dirpath)
        print(f"[DEBUG] downloading {url} to {fpath}")
        with open(fpath, "wb") as f:
            f.write(requests.get(url).content)

    return fpath


def read_xpt(fp) -> pd.DataFrame:
    assert os.path.exists(fp), "file does not exist %s" % fp
    with open(fp, "rb") as f:
        obj = xport.v56.load(f)
    # index into SAS structure, assuming there is only one dataframe
    assert len(tuple(obj._members.keys())) == 1
    key = tuple(obj._members.keys())[0]
    ds = obj[key]
    # convert xport.Dataset to pd.DataFrame
    columns = [ds[c] for c in ds.columns]
    df = pd.DataFrame(columns).T
    del ds
    return df


def run_in_subproces(cmd):
    print(f"[INFO] running {cmd}")
    res = subprocess.run(cmd, shell=True)
    print(f"[DEBUG] {cmd} returned {res}")
    return


def sliding_window(iterable, n):
    """ From https://docs.python.org/3/library/itertools.html"""
    # sliding_window('ABCDEFG', 4) --> ABCD BCDE CDEF DEFG
    it = iter(iterable)
    window = collections.deque(islice(it, n), maxlen=n)
    if len(window) == n:
        yield tuple(window)
    for x in it:
        window.append(x)
        yield tuple(window)


def make_uid(name: str, splitter: Splitter, replace_chars="*/:'$!") -> str:
    """Make a unique identifier for an experiment."""
    uid = name
    if isinstance(splitter, DomainSplitter):
        attrs = {'domain_split_varname': splitter.domain_split_varname,
                 'domain_split_ood_value': ''.join(str(x) for x in splitter.domain_split_ood_values)}
        if splitter.domain_split_id_values:
            attrs['domain_split_id_values'] = ''.join(str(x) for x in splitter.domain_split_id_values)
        uid += ''.join(f'{k}_{v}' for k, v in attrs.items())
    # if any slashes exist, replace with periods.
    for char in replace_chars:
        uid = uid.replace(char, '.')
    return uid
