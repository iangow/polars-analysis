# /// script
# requires-python = ">=3.11"
# dependencies = ["polars==2.1.0", "duckdb==1.5.1", "pyarrow==23.0.1"]
# ///
"""Self-contained reproduction of high memory use in grouped dynamic covariance.

Run with: uv run --no-project repro.py --output repro-results
Default: 1,000 firms x 2,500 days, three fresh processes per query.
Supports macOS and Linux; no external data or NumPy is needed.
"""

import argparse
import contextlib
import ctypes
from datetime import date, timedelta
import io
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

SCALAR_CASES = ["cov", "self-cov", "corr", "var", "std", "mean", "sum", "min",
                "max", "median", "quantile", "n_unique", "skew", "kurtosis"]
BOOK_CASES = ["cov-alt", "corr-spearman", "beta-alpha", "weighted-mean",
              "conditional-mean", "first", "last", "count", "any", "custom-dot"]
WORKER_CASES = ["polars-cov-streaming", "polars-cov-in-memory", "polars-two-vars", "duckdb-cov"] + ["polars-" + x for x in SCALAR_CASES if x != "cov"] + ["polars-" + x for x in BOOK_CASES]

def memory_footprint():
    """macOS rusage_info_v4: current footprint and lifetime peak, in bytes."""
    if sys.platform != "darwin":
        return None
    # Layout from sys/resource.h: 16-byte UUID followed by 35 uint64_t fields.
    # ri_phys_footprint is field 7; ri_lifetime_max_phys_footprint is field 28.
    class Usage(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16), ("values", ctypes.c_uint64 * 35)]
    usage = Usage()
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
    libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    if libproc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(usage)):
        return None
    return {"current_bytes": usage.values[7], "peak_bytes": usage.values[28]}


def generate(path, firms, days):
    import polars as pl
    row = pl.col("row")
    # Sorted by (firm_id, date) by construction. Both variables have nonzero
    # within-window variance; the data is entirely deterministic.
    df = (
        pl.DataFrame({"row": pl.int_range(firms * days, eager=True)})
        .select(
            firm_id=(row // days).cast(pl.Int32),
            date=(pl.lit(date(2000, 1, 1)) + pl.duration(days=row % days)).cast(pl.Date),
            x=(row.cast(pl.Float64) * 0.1).sin() * 0.02,
            y=(row.cast(pl.Float64) * 0.07).cos() * 0.03,
        )
    )
    df.write_parquet(path)
    return {"rows": df.height, "columns": df.width,
            "uncompressed_bytes": df.estimated_size(),
            "parquet_bytes": path.stat().st_size,
            "firms": firms, "days_per_firm": days}


def polars_query(path, aggregation):
    import polars as pl
    x = pl.col("x")
    y = pl.col("y")
    expressions = {"cov": pl.cov("x", "y"), "self-cov": pl.cov("x", "x"),
                   "corr": pl.corr("x", "y"), "var": x.var(), "std": x.std(),
                   "mean": x.mean(), "sum": x.sum(), "min": x.min(), "max": x.max(),
                   "median": x.median(), "quantile": x.quantile(0.9),
                   "n_unique": x.n_unique(), "skew": x.skew(), "kurtosis": x.kurtosis(),
                   "cov-alt": ((x-x.mean())*(y-y.mean())).sum()/(pl.len()-1),
                   "corr-spearman": pl.corr("x", "y", method="spearman"),
                   "weighted-mean": (x*y.abs()).sum()/y.abs().sum(),
                   "conditional-mean": pl.when(y>0).then(x).mean(),
                   "first": x.first(), "last": x.last(), "count": x.count(),
                   "any": (x>0).any(),
                   "custom-dot": pl.map_groups(["x", "y"], lambda s: float((s[0]*s[1]).sum()),
                                                return_dtype=pl.Float64, returns_scalar=True)}
    if aggregation == "beta-alpha":
        beta = pl.cov("x", "y")/x.var()
        aggs = {"beta": beta, "alpha": y.mean()-beta*x.mean()}
    elif aggregation == "two-vars":
        aggs = {"var_x": x.var(), "var_y": y.var()}
    else:
        aggs = {"value": expressions[aggregation]}
    return (
        pl.scan_parquet(path)
        .group_by_dynamic(
            "date", every="1mo", period="60mo", offset="-60mo",
            group_by="firm_id", closed="left", label="right",
        )
        .agg(**aggs, n=pl.len())
        .filter(pl.col("n") >= 1000)
        .sort("firm_id", "date")
    )


def duckdb_query(path):
    import duckdb
    con = duckdb.connect()
    escaped = str(path).replace("'", "''")
    return con.sql(f"""
        WITH observations AS (
            SELECT firm_id, date, x, y FROM '{escaped}'
        ), bounds AS (
            SELECT firm_id,
                   date_trunc('month', min(date)) + INTERVAL 1 MONTH AS first_label,
                   date_trunc('month', max(date)) + INTERVAL 60 MONTH AS last_label
            FROM observations GROUP BY firm_id
        ), evaluation_points AS (
            SELECT firm_id,
                   unnest(generate_series(first_label, last_label, INTERVAL 1 MONTH))::DATE AS date
            FROM bounds
        ), window_input AS (
            SELECT firm_id, date, x, y, 1 AS observation, false AS is_evaluation
            FROM observations
            UNION ALL
            SELECT firm_id, date, NULL::DOUBLE, NULL::DOUBLE, NULL::INTEGER, true
            FROM evaluation_points
        )
        SELECT firm_id, date, covar_samp(x, y) OVER w AS value,
               count(observation) OVER w AS n
        FROM window_input
        WINDOW w AS (
            PARTITION BY firm_id ORDER BY date
            RANGE BETWEEN INTERVAL 60 MONTH PRECEDING
                      AND INTERVAL 1 DAY PRECEDING
        )
        QUALIFY is_evaluation AND count(observation) OVER w >= 1000
        ORDER BY firm_id, date
    """).pl()


def worker(args):
    import duckdb
    import polars as pl
    start = time.perf_counter()
    if args.worker == "duckdb-cov":
        result = duckdb_query(args.data)
    else:
        agg = "cov" if args.worker in {"polars-cov-streaming", "polars-cov-in-memory"} else args.worker.removeprefix("polars-")
        engine = "in-memory" if args.worker.endswith("in-memory") else "streaming"
        result = polars_query(args.data, agg).collect(engine=engine)
    elapsed = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    footprint = memory_footprint()
    print(json.dumps({"case": args.worker, "elapsed_seconds": elapsed,
                      "peak_rss_bytes": peak_bytes,
                      "peak_footprint_bytes": footprint["peak_bytes"] if footprint else None,
                      "rows": result.height, "polars": pl.__version__,
                      "duckdb": duckdb.__version__, "python": platform.python_version(),
                      "platform": platform.platform(), "polars_threads": pl.thread_pool_size(),
                      "ooc_memory_budget_mb": os.environ.get("POLARS_OOC_MEMORY_BUDGET_MB")}), flush=True)
    # Serialization, plan generation and version inspection are excluded from
    # both elapsed time and reported memory peaks.
    if args.save_result:
        result.write_parquet(args.save_result)
        if args.worker.startswith("polars-"):
            plan = polars_query(args.data, agg).show_graph(
                show=False, raw_output=True, engine="streaming", plan_stage="physical")
            args.save_result.with_suffix(".dot").write_text(plan)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--firms", type=int, default=1000)
    p.add_argument("--days", type=int, default=2500)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, default=Path("repro-results"))
    p.add_argument("--memory-budget-mb", type=int)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--sweep", action="store_true", help="Also measure related scalar aggregates")
    p.add_argument("--book-cases", action="store_true", help="Also measure examples and controls motivated by the ERA book")
    p.add_argument("--worker", choices=WORKER_CASES)
    p.add_argument("--data", type=Path)
    p.add_argument("--save-result", type=Path)
    args = p.parse_args()
    if args.worker:
        worker(args)
        return
    if args.firms < 1 or args.days < 1100 or args.repeats < 1:
        p.error("Use firms >= 1, days >= 1100 and repeats >= 1")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = output / "synthetic.parquet"
    metadata = generate(data, args.firms, args.days)
    (output / "input.json").write_text(json.dumps(metadata, indent=2) + "\n")
    env = os.environ.copy()
    if args.memory_budget_mb is not None:
        env["POLARS_OOC_MEMORY_BUDGET_MB"] = str(args.memory_budget_mb)
    if args.verbose:
        env["POLARS_VERBOSE"] = "1"
    measurements = []
    cases = ["polars-cov-streaming", "polars-cov-in-memory", "polars-var", "polars-two-vars", "duckdb-cov"]
    if args.sweep:
        cases += ["polars-" + x for x in SCALAR_CASES if x not in {"cov", "var"}]
    if args.book_cases:
        cases += ["polars-" + x for x in BOOK_CASES]
    for rep in range(1, args.repeats + 1):
        shift = (rep - 1) % len(cases)
        for case in cases[shift:] + cases[:shift]:
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", case,
                       "--data", str(data), "--days", str(args.days)]
            if rep == 1:
                command += ["--save-result", str(output / (case + ".parquet"))]
            proc = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
            (output / f"{case}-{rep}.stderr").write_text(proc.stderr)
            row = json.loads(proc.stdout.strip().splitlines()[-1])
            row["repetition"] = rep
            measurements.append(row)
            print(json.dumps(row), flush=True)
            (output / "measurements.json").write_text(json.dumps(measurements, indent=2) + "\n")
    import polars as pl
    from polars.testing import assert_frame_equal
    left = pl.read_parquet(output / "polars-cov-streaming.parquet")
    right = pl.read_parquet(output / "duckdb-cov.parquet")
    memory = pl.read_parquet(output / "polars-cov-in-memory.parquet")
    right = right.cast({"n": pl.UInt32})
    assert_frame_equal(left, right, check_exact=False, rel_tol=1e-10, abs_tol=1e-14)
    assert_frame_equal(left, memory, check_exact=True)
    validation = {"polars_rows": left.height, "duckdb_rows": right.height,
                  "identical_keys": left.select("firm_id", "date").equals(right.select("firm_id", "date")),
                  "identical_counts": left["n"].equals(right["n"]),
                  "null_counts": left.null_count().to_dicts()[0],
                  "max_absolute_covariance_difference": (left["value"]-right["value"]).abs().max()}
    if args.sweep:
        self_cov = pl.read_parquet(output / "polars-self-cov.parquet")
        variance = pl.read_parquet(output / "polars-var.parquet")
        assert_frame_equal(self_cov, variance, check_exact=False, rel_tol=1e-10, abs_tol=1e-14)
        validation["max_absolute_self_covariance_vs_variance_difference"] = (self_cov["value"]-variance["value"]).abs().max()
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    summary = []
    for case in cases:
        group = [x for x in measurements if x["case"] == case]
        summary.append({"case": case, "runs": len(group),
                        "median_seconds": statistics.median(x["elapsed_seconds"] for x in group),
                        "median_peak_rss_gib": statistics.median(x["peak_rss_bytes"]/1024**3 for x in group),
                        "median_peak_footprint_gib": statistics.median(x["peak_footprint_bytes"]/1024**3 for x in group) if sys.platform == "darwin" else None})
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    versions = io.StringIO()
    with contextlib.redirect_stdout(versions):
        pl.show_versions()
    (output / "versions.txt").write_text(versions.getvalue())
    print(json.dumps({"input": metadata, "summary": summary, "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
