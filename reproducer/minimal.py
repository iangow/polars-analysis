# /// script
# requires-python = ">=3.11"
# dependencies = ["polars==2.1.0"]
# ///
from datetime import date
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import polars as pl

mode = sys.argv[1] if len(sys.argv) > 1 else "cov"
if mode not in {"cov", "cov-alt", "var", "self-cov", "corr", "mean"}:
    raise SystemExit("Usage: minimal.py [cov|cov-alt|var|self-cov|corr|mean]")

firms, days = 1000, 2500
row = pl.col("row")
df = pl.DataFrame({"row": pl.int_range(firms * days, eager=True)}).select(
    firm_id=(row // days).cast(pl.Int32),
    date=(pl.lit(date(2000, 1, 1)) + pl.duration(days=row % days)).cast(pl.Date),
    x=(row.cast(pl.Float64) * 0.1).sin() * 0.02,
    y=(row.cast(pl.Float64) * 0.07).cos() * 0.03,
)
print(f"Input: {df.height:,} rows; {df.estimated_size('mb'):.2f} MiB uncompressed")
aggs = {"cov": {"value": pl.cov("x", "y")},
        "cov-alt": {"value": ((pl.col("x") - pl.col("x").mean()) *
                               (pl.col("y") - pl.col("y").mean())).sum() / (pl.len() - 1)},
        "self-cov": {"value": pl.cov("x", "x")},
        "corr": {"value": pl.corr("x", "y")},
        "mean": {"mean_x": pl.col("x").mean(), "mean_y": pl.col("y").mean()},
        "var": {"var_x": pl.col("x").var(), "var_y": pl.col("y").var()}}[mode]
with TemporaryDirectory() as directory:
    path = Path(directory) / "synthetic.parquet"
    df.write_parquet(path)
    del df
    query = (
        pl.scan_parquet(path)
        .group_by_dynamic(
            "date", every="1mo", period="60mo", offset="-60mo",
            group_by="firm_id", closed="left", label="right",
        )
        .agg(**aggs, n=pl.len())
        .filter(pl.col("n") >= 1000)
        .sort("firm_id", "date")
    )
    result = query.collect(engine="streaming")
    print(result)
