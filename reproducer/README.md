# Polars overlapping-window memory reproducer

All data is synthetic. Defaults: 1,000 firms × 2,500 days = 2,500,000 rows, 57.22 MiB uncompressed input. Released covariance/correlation uses about 2.57 GiB; the locally compiled patch reduces that to about 0.30 GiB.

## Released-wheel reproduction

```sh
uv run --no-project minimal.py cov
uv run --no-project minimal.py cov-alt
uv run --no-project minimal.py var
uv run --no-project minimal.py self-cov
uv run --no-project minimal.py corr
uv run --no-project minimal.py mean
uv run --no-project repro.py --output repro-results --sweep --book-cases --verbose
```

The runner pins Polars 2.1.0, DuckDB 1.5.1 and PyArrow 23.0.1. It supports macOS and Linux, writes synthetic input once, runs each case in three fresh processes, and records runtime, peak RSS and (on macOS) peak total footprint. Timing excludes imports, generation, serialization and validation. Memory is the fresh worker process's peak, including the interpreter and imported libraries, measured before serialization and validation; generation runs separately. No input downloads, credentials or NumPy are required.

`minimal.py var` and `minimal.py mean` aggregate both x and y, so they read the same columns as covariance. `self-cov` computes covariance of x with itself. External measurements can use `/usr/bin/time -l` on macOS or `/usr/bin/time -v` on Linux; those minimal-process measurements include generation.

## Matching Polars and DuckDB

DuckDB generates the same monthly window labels as Polars, including labels after the final input date. Null evaluation rows do not enter the covariance or observation count. Both versions return all 77,000 keys with the same counts. Polars is not filtered to the last input date.

## Patched build

See `BUILD.md` for build and test commands. Run the compiled version with its Python interpreter directly; the uv metadata selects the official release. `candidate-fix.patch` built successfully against 2.1.0, and all 26 Polars before/after tables were exactly equal. The patch also fixes the book's direct `cov_alt()` expression by evaluating and summing one overlapping window at a time.

`results/` contains released and patched summaries, raw measurements, CSV comparisons, numerical checks, native-module provenance and the upstream test log. `SOURCE_NOTES.md` traces the implementation. `--firms 10000` generates a 25-million-row input; the original full beta query results are recorded in the patch validation file.

## Separate candidate patches

Both candidate fixes change Polars' Rust execution engine. The Python `cov_alt()` formula is unchanged.

- [Multi-input apply patch](multi-input-apply.patch): lazy window views for covariance, correlation and other multi-input grouped functions.
- [Composed-sum patch](composed-window-sum.patch): evaluate and sum Float64 arithmetic over columns and column means one overlapping window at a time.
- [Combined patch](candidate-fix.patch): the exact source change used for the final build and numerical validation.

The separate patches are supplied for review. The final benchmark and regression results were recorded with both applied; the earlier multi-input-only build also provides the same-profile control.
