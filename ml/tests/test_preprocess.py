import json
import sys
from pathlib import Path

import pandas as pd

from ml.src import preprocess


def test_preprocess_main_reads_and_reports(tmp_path, monkeypatch, capsys):
    data = [
        {"title": "A", "rating": 4.0, "year": 2020},
        {"title": "B", "rating": None, "year": 2021},
    ]
    src = tmp_path / "data.json"
    src.write_text(json.dumps(data), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["preprocess", "--input", str(src)])
    rc = preprocess.main()
    assert rc == 0

    out = capsys.readouterr().out
    assert "Missing values" in out
    assert "rating" in out


def test_preprocess_heatmap_writes_file(tmp_path, monkeypatch):
    data = [{"title": "A", "rating": 4.0, "year": 2020}]
    src = tmp_path / "data.json"
    src.write_text(json.dumps(data), encoding="utf-8")
    out_png = tmp_path / "heatmap.png"

    monkeypatch.setattr(
        sys,
        "argv",
        ["preprocess", "--input", str(src), "--save-heatmap", str(out_png)],
    )
    rc = preprocess.main()
    assert rc == 0
    assert out_png.exists()
