from __future__ import annotations
import os
import pandas as pd
from pathlib import Path
from typing import Optional

data_dir = Path(os.getenv("ML_DATA_DIR", Path(__file__).parents[1] / "data"))

def load_merged(path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        path = data_dir / "merged_results.json"
    return pd.read_json(path)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # example: convert year to int, fill missing ratings with NaN
    df = df.copy()
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    return df
