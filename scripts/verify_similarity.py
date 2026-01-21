import sys
import math
from pathlib import Path
import importlib.util

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPERS_PATH = REPO_ROOT / "streamlit_app" / "helpers.py"

spec = importlib.util.spec_from_file_location("helpers", str(HELPERS_PATH))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

# Load reference dataframe
ref_df = helpers._load_reference_df(helpers.REF_DATA_PATH)
if ref_df is None or getattr(ref_df, "empty", True):
    print("ERROR: Reference dataset is empty or missing.")
    sys.exit(1)

# Use first row as a realistic user input
row = ref_df.iloc[0]
user_values = dict(row)

results = helpers._similar_movies_with_explanations(ref_df, user_values, [], [], top_n=3)

if not results:
    print("ERROR: No similar movies returned.")
    sys.exit(2)

print("title\tsimilarity_pct\tposter_present\tposter_source\turl")
for r in results:
    title = r.get("title") or ""
    sim = r.get("similarity_pct")
    sim_str = "" if sim is None else f"{sim}"
    poster_url = r.get("poster_url") or ""
    poster_present = bool(poster_url)
    if poster_url:
        if "image.tmdb.org" in poster_url:
            poster_source = "TMDB"
        elif "letterboxd" in poster_url or "film-poster" in poster_url:
            poster_source = "Letterboxd"
        else:
            poster_source = "Other"
    else:
        poster_source = "None"
    url = r.get("url") or ""
    # Check for NaN-like similarity
    bad_sim = (sim is None) or (not isinstance(sim, (int, float))) or (not math.isfinite(float(sim)))
    # Print row
    print(f"{title}\t{sim_str}\t{poster_present}\t{poster_source}\t{url}")

# Summaries
nan_count = sum(1 for r in results if (r.get("similarity_pct") is None) or (not isinstance(r.get("similarity_pct"), (int, float))) or (not math.isfinite(float(r.get("similarity_pct") or 0.0))))
no_poster = sum(1 for r in results if not r.get("poster_url"))
print(f"SUMMARY\tNaN_similarities={nan_count}\tmissing_posters={no_poster}")
