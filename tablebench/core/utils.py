import os
import requests
import subprocess
import urllib.parse

import pandas as pd

import xport.v56


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


def read_xpt(fp):
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
