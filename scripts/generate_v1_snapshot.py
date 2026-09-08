"""生成等级分 v2 上线前的 v1 静态快照。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rating_system import calculate_ratings  # noqa: E402


def main() -> None:
    source = pd.read_csv(ROOT / "data.csv")
    source["Date"] = pd.to_datetime(source["Date"], errors="coerce")
    result = calculate_ratings(source, version="v1")

    history = result.history
    stats = history.groupby("Name").agg(
        total_games=("Result", "size"),
        wins=("Result", lambda values: int((values == "Win").sum())),
    )
    rows = []
    for name, rating in result.ratings.items():
        total_games = int(stats.at[name, "total_games"])
        wins = int(stats.at[name, "wins"])
        rows.append(
            {
                "name": name,
                "rating_exact": rating,
                "rating_display": round(rating),
                "total_games": total_games,
                "wins": wins,
                "win_rate": wins / total_games if total_games else 0,
                "last_active": result.last_active.get(name),
            }
        )

    snapshot = pd.DataFrame(rows).sort_values(
        ["rating_exact", "name"], ascending=[False, True]
    )
    snapshot.insert(0, "rank", range(1, len(snapshot) + 1))
    output_dir = ROOT / "snapshots"
    output_dir.mkdir(exist_ok=True)
    snapshot.to_csv(output_dir / "rating_v1_2026-09-08.csv", index=False)


if __name__ == "__main__":
    main()
