"""Generate synthetic telemetry with the same shape as the public pipeline expects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(13)
    rows = []
    start = pd.Timestamp("2024-01-01T00:00:00Z")

    for animal_idx in range(18):
        label = int(animal_idx >= 12)
        base_lat = 44.0 + rng.normal(0, 0.05)
        base_lon = -106.0 + rng.normal(0, 0.05)
        lat = base_lat
        lon = base_lon
        for step in range(90):
            timestamp = start + pd.Timedelta(hours=8 * step)
            movement_scale = 0.0025 if label == 0 else 0.004
            lat += rng.normal(0, movement_scale)
            lon += rng.normal(0, movement_scale)
            rows.append(
                {
                    "animal_id": f"synthetic_{animal_idx:02d}",
                    "timestamp": timestamp.isoformat(),
                    "latitude": lat,
                    "longitude": lon,
                    "species": "white_tailed_deer" if animal_idx % 2 else "mule_deer",
                    "sex": "F" if animal_idx % 3 else "M",
                    "genotype": ["GG", "GS", "SS"][animal_idx % 3],
                    "label": label,
                }
            )

    out = Path("data/sample/synthetic_telemetry.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
