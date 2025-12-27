import streamlit as st
import pandas as pd
from pathlib import Path

st.title("ML starter — merged results overview")

# attempt to load merged_results.json from repo root
data_path = Path(__file__).parents[1] / "merged_results.json"
if data_path.exists():
    try:
        df = pd.read_json(data_path)
        st.write(f"Loaded {len(df)} records")
        st.dataframe(df.head(50))
    except Exception as e:
        st.error(f"Error loading data: {e}")
else:
    st.warning(f"Data file not found: {data_path}")

st.info("Add preprocessing and model inference here.")
