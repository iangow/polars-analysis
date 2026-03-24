from pathlib import Path
import numpy as np
import polars as pl
from datetime import date, timedelta

def write_synth_parquet(
    path: str | Path,
    n_firms: int = 10_000,
    n_days: int = 2_500,
    seed: int = 0,
):
    path = Path(path)

    rng = np.random.default_rng(seed)

    start = date(2000, 1, 1)
    dates = np.array(
        [start + timedelta(days=i) for i in range(n_days)],
        dtype="datetime64[D]",
    )

    df = pl.DataFrame(
        {
            "firm_id": np.repeat(np.arange(n_firms, dtype=np.int32), n_days),
            "date": np.tile(dates, n_firms),
            "ret_excess": rng.normal(0.0, 0.02, n_firms * n_days),
            "mkt_excess": rng.normal(0.0, 0.02, n_firms * n_days),
        }
    ).sort("firm_id", "date")

    df.write_parquet(path)
    return path

path = write_synth_parquet("synthetic.parquet")