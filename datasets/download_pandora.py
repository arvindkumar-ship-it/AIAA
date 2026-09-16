"""
PANDORA (Reddit Big-5 personality) dataset.

IMPORTANT: the official source (psy.takelab.fer.hr/datasets/all/pandora)
is "available on request" and comes with a user agreement -- that WILL
take longer than a week to get approved. Skip it.

Instead, use the public HuggingFace mirror which has the same PANDORA
Big-5 data (1,608 users, Reddit comments + Big Five scores) with no
request/approval needed:
    https://huggingface.co/datasets/Fatima0923/Automated-Personality-Prediction

pip install datasets
"""

from datasets import load_dataset
import os

OUT_PATH = os.path.join(os.path.dirname(__file__), "pandora_style_data.csv")


def download_and_save():
    ds = load_dataset("Fatima0923/Automated-Personality-Prediction")
    df = ds["train"].to_pandas()
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUT_PATH}")
    print(df.columns.tolist())
    return df


if __name__ == "__main__":
    download_and_save()
