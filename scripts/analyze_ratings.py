"""Résume les notes d'un fichier de résultats JSON.

Lit `results_parallel.json` (sinon `results_parallel_20.json`), calcule les stats
de base (moyenne/médiane/min/max/écart-type), un histogramme par pas de 0,5 et
affiche les 10 meilleurs/pire. Imprime le résumé au format JSON sur stdout.

Usage (PowerShell) :
    $env:PYTHONPATH='src'
    .\.venv\Scripts\python .\scripts\analyze_ratings.py
"""
from pathlib import Path
import json
import statistics
import math


def main(path: Path):
    """Calcule les statistiques des notes et les affiche en JSON."""
    data = json.loads(path.read_text(encoding='utf-8'))
    ratings = []
    records = []
    for rec in data:
        title = rec.get('title') or rec.get('url')
        year = rec.get('year')
        r = rec.get('rating')
        records.append({'title': title, 'year': year, 'rating': r, 'url': rec.get('url')})
        if r is None or (isinstance(r, str) and r.strip() == ''):
            continue
        try:
            ratings.append(float(r))
        except Exception:
            continue

    total = len(data)
    present = len(ratings)
    missing = total - present

    out: dict = {
        'total_records': total,
        'ratings_present': present,
        'ratings_missing': missing,
    }

    if present:
        out.update({
            'mean': statistics.mean(ratings),
            'median': statistics.median(ratings),
            'min': min(ratings),
            'max': max(ratings),
        })
        if present > 1:
            out['stdev_population'] = statistics.pstdev(ratings)

        # histogram with 0.5 bins
        bins = [i * 0.5 for i in range(0, 11)]
        labels = [f"{bins[i]:.1f}-{(bins[i+1]-0.01):.2f}" for i in range(len(bins)-1)]
        hist = {lab: 0 for lab in labels}
        for v in ratings:
            placed = False
            for i in range(len(bins) - 1):
                if v >= bins[i] and v < bins[i + 1]:
                    hist[labels[i]] += 1
                    placed = True
                    break
            if not placed and math.isclose(v, bins[-1]):
                hist[labels[-1]] += 1
        out['histogram_0.5_bins'] = hist

        # top / bottom
        sorted_by = sorted(records, key=lambda r: (r['rating'] if r['rating'] is not None else -9999), reverse=True)
        out['top10'] = sorted_by[:10]
        sorted_low = sorted(records, key=lambda r: (r['rating'] if r['rating'] is not None else 9999))
        out['bottom10'] = sorted_low[:10]

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = Path('results_31_40.json')
    if not p.exists():
        # fallback to 20-pages file if present
        p2 = Path('results_21_30.json')
        if p2.exists():
            p = p2
        else:
            raise SystemExit('results_31_40.json not found')
    main(p)
