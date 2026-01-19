from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_CANDIDATES = [
    REPO_ROOT / "ml" / "data" / "final_results_28.json",
    REPO_ROOT / "ml" / "data" / "merged_results.json",
    REPO_ROOT / "merged_results.json",
]


def _similarity_weight(col_name: str) -> float:
    c = str(col_name).lower().strip()
    if c in {"genres", "genre", "themes", "theme"}:
        return 2.0
    if c == "rating":
        return 1.8
    if c in {"director", "directors", "actors", "casting", "cast"}:
        return 1.5
    if c in {"budget", "revenue"}:
        return 1.2
    return 1.0


def _clean_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for x in values:
        if x is None:
            continue
        s = str(x).strip()
        if not s:
            continue
        out.append(s)
    return out


def _identity_similarity(col: str, uval: object, rval: object) -> float | None:
    u = _clean_list(uval)
    r = _clean_list(rval)

    # Match app behavior: only compare the first 5 actors for cast-like columns.
    if col.lower().strip() in {"casting", "actors", "cast"}:
        u = u[:5]
        r = r[:5]

    if not u or not r:
        return None

    us = {s.lower() for s in u if s.strip()}
    rs = {s.lower() for s in r if s.strip()}
    if not us or not rs:
        return None

    inter = us & rs
    max_count = max(len(us), len(rs))
    if max_count <= 0:
        return None
    return float(len(inter) / max_count)


def _numeric_spans(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    spans: dict[str, float] = {}
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        s = s[np.isfinite(s)]
        if s.empty:
            continue
        span = float(s.max() - s.min())
        spans[c] = span if span > 0 else 1.0
    return spans


def _numeric_similarity(spans: dict[str, float], col: str, uval: object, rval: object) -> float | None:
    try:
        u = float(uval)  # type: ignore[arg-type]
        r = float(rval)  # type: ignore[arg-type]
    except Exception:
        return None
    if not (np.isfinite(u) and np.isfinite(r)):
        return None
    span = spans.get(col)
    if not span:
        return None
    d = abs(u - r)
    return float(1.0 - min(d / float(span), 1.0))


def _pick_row(df: pd.DataFrame, title: str, year: int | None) -> tuple[int | None, dict[str, Any] | None]:
    if df.empty:
        return None, None

    def _norm(s: str) -> str:
        s = (s or "").replace("\u00a0", " ").strip()
        s = " ".join(s.split())
        return s

    t = _norm(title or "")
    if not t:
        return None, None

    titles = df["title"].astype(str).map(_norm)

    # Exact match first
    m = titles.eq(t)
    if year is not None and "year" in df.columns:
        y = pd.to_numeric(df["year"], errors="coerce")
        m = m & (y == float(year))
    exact = df[m]
    if not exact.empty:
        idx = int(exact.index[0])
        return idx, {str(k): v for k, v in df.loc[idx].to_dict().items()}

    # Fuzzy contains
    m = titles.str.contains(t, case=False, na=False)
    if year is not None and "year" in df.columns:
        y = pd.to_numeric(df["year"], errors="coerce")
        m = m & (y == float(year))
    fuzzy = df[m]
    if not fuzzy.empty:
        idx = int(fuzzy.index[0])
        return idx, {str(k): v for k, v in df.loc[idx].to_dict().items()}

    return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description="Explain similarity computation between two dataset films.")
    ap.add_argument("--a-title", required=True)
    ap.add_argument("--a-year", type=int, default=None)
    ap.add_argument("--b-title", required=True)
    ap.add_argument("--b-year", type=int, default=None)
    ap.add_argument("--data", type=str, default=None, help="Path to reference json; defaults to ml/data/final_results_28.json then fallbacks")
    args = ap.parse_args()

    data_path = Path(args.data) if args.data else None
    candidates = [data_path] if data_path else []
    candidates.extend(DEFAULT_DATA_CANDIDATES)

    df = pd.DataFrame()
    used_path = None
    for p in candidates:
        if not p:
            continue
        if p.exists():
            df = pd.read_json(p)
            used_path = p
            break

    if df.empty:
        raise SystemExit("No dataset found or dataset is empty")

    a_idx, a = _pick_row(df, args.a_title, args.a_year)
    b_idx, b = _pick_row(df, args.b_title, args.b_year)

    print(f"Dataset: {used_path}")
    print(f"A: {args.a_title} ({args.a_year}) -> idx={a_idx}")
    print(f"B: {args.b_title} ({args.b_year}) -> idx={b_idx}")

    if a is None:
        print("\nCould not find film A. Closest title matches:")
        print(df[df['title'].astype(str).str.contains(args.a_title, case=False, na=False)][['title','year','url']].head(15).to_string(index=True))
        return 2
    if b is None:
        print("\nCould not find film B. Closest title matches:")
        print(df[df['title'].astype(str).str.contains(args.b_title.split(':')[0], case=False, na=False)][['title','year','url']].head(25).to_string(index=True))
        return 2

    # Focus on the fields you mentioned + the main people/list fields.
    numeric_cols = [
        "fans_favoris",
        "budget",
        "revenue",
        "nbr_watched",
        "nbr_likes",
        "nbr_appearence",
        "duration",
        "year",
    ]
    identity_cols = [
        "genres",
        "themes",
        "directors",
        "casting",
        "producers",
        "writers",
        "composer",
        "studio",
        "languages",
    ]

    spans = _numeric_spans(df, numeric_cols)

    rows: list[dict[str, Any]] = []
    total_w = 0.0
    total_ws = 0.0

    def _add(col: str, typ: str, sim: float | None, details: dict[str, Any]) -> None:
        nonlocal total_w, total_ws
        w = _similarity_weight(col)
        used = sim is not None
        if used:
            total_w += w
            total_ws += float(sim) * w
        rows.append(
            {
                "col": col,
                "type": typ,
                "weight": w,
                "used": used,
                "sim_pct": None if sim is None else round(100.0 * float(sim), 1),
                **details,
            }
        )

    for col in numeric_cols:
        sim = _numeric_similarity(spans, col, a.get(col), b.get(col))
        _add(
            col,
            "numeric",
            sim,
            {
                "a": a.get(col),
                "b": b.get(col),
                "span": spans.get(col),
                "note": "ignored (missing/unusable)" if sim is None else "",
            },
        )

    for col in identity_cols:
        sim = _identity_similarity(col, a.get(col), b.get(col))
        a_list = _clean_list(a.get(col))
        b_list = _clean_list(b.get(col))
        if col.lower().strip() in {"casting", "actors", "cast"}:
            a_list = a_list[:5]
            b_list = b_list[:5]
        inter = sorted({s.lower() for s in a_list} & {s.lower() for s in b_list})
        _add(
            col,
            "identity",
            sim,
            {
                "a_n": len(set(s.lower() for s in a_list)),
                "b_n": len(set(s.lower() for s in b_list)),
                "matched": inter[:10],
                "note": "ignored (missing/empty)" if sim is None else "",
            },
        )

    overall = (total_ws / total_w) if total_w > 0 else 0.0

    print("\nPer-feature similarity (same formula as app):")
    show = pd.DataFrame(rows)
    cols = [
        "col",
        "type",
        "weight",
        "used",
        "sim_pct",
        "a",
        "b",
        "span",
        "a_n",
        "b_n",
        "matched",
        "note",
    ]
    cols = [c for c in cols if c in show.columns]
    with pd.option_context("display.max_colwidth", 120, "display.width", 200):
        print(show[cols].to_string(index=False))

    print("\nOverall similarity:")
    print(f"  weighted_mean = {overall:.4f} -> {overall*100:.1f}%")
    print(f"  total_weight_used = {total_w:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
