"""
Downloads the REAL StudentLife dataset directly - no R, no approval needed.
Source (official, public): https://studentlife.cs.dartmouth.edu/dataset/dataset.tar.bz2
48 Dartmouth students, 10-week term, phone sensing + EMA (interruptibility-relevant) labels.

pip install requests tqdm
"""

import os
import tarfile
import requests
from tqdm import tqdm

URL = "https://studentlife.cs.dartmouth.edu/dataset/dataset.tar.bz2"
OUT_DIR = os.path.join(os.path.dirname(__file__), "studentlife_raw")
ARCHIVE_PATH = os.path.join(os.path.dirname(__file__), "dataset.tar.bz2")


def download():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Downloading from {URL} (large file, be patient)...")
    resp = requests.get(URL, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(ARCHIVE_PATH, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print("Extracting...")
    with tarfile.open(ARCHIVE_PATH, "r:bz2") as tar:
        tar.extractall(OUT_DIR)
    print(f"Done. Data is in {OUT_DIR}")
    print("Relevant folders: EMA/ (interruptibility labels), sensing/ (activity/phonelock context)")


if __name__ == "__main__":
    download()
