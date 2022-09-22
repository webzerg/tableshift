import os
import requests
import urllib.parse


def initialize_dir(dir: str):
    """Create a directory if it does not exist."""
    if not os.path.exists(dir):
        os.makedirs(dir)


def basename_from_url(url: str) -> str:
    parsed_url = urllib.parse.urlparse(url)
    fname = os.path.basename(parsed_url.path)
    return fname


def download_file(url: str, dirpath: str, if_not_exist=True):
    """Download file to the specified directory."""
    fname = basename_from_url(url)
    fpath = os.path.join(dirpath, fname)

    if os.path.exists(fpath) and if_not_exist:
        # Case: file already exists; skip it.
        return

    initialize_dir(dirpath)
    with open(fpath, "wb") as f:
        f.write(requests.get(url).content)
