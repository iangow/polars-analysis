# Building and testing the patch

Source: Polars tag `py-2.1.0`, commit `7f902bd3ffd27ff82d14278a09a295ff2f7aed92`. The candidate patch built successfully without changes. The native runtime uses the full default feature set and 32-bit indices.

```sh
git clone --depth 1 --branch py-2.1.0 https://github.com/pola-rs/polars.git
cd polars
git apply /path/to/candidate-fix.patch
rustup toolchain install nightly-2026-09-01 --profile minimal
uv venv --python 3.14.2 .venv-patched
uv pip install --python .venv-patched/bin/python maturin duckdb==1.5.1 pyarrow==23.0.1
CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0 \
PYO3_PYTHON="$PWD/.venv-patched/bin/python" \
.venv-patched/bin/maturin build --locked --profile nodebug-release \
  --manifest-path py-polars/runtime/polars-runtime-32/Cargo.toml --out patched-wheels
uv pip install --python .venv-patched/bin/python --no-deps patched-wheels/*.whl polars==2.1.0
```

The initial full build on macOS ARM took 15 minutes 26 seconds; the extended patch was then rebuilt from the cached dependencies. The wheel is specific to that platform and supports CPython 3.11 and newer through abi3. The official wheel uses fat LTO; this local build uses thin LTO. Recorded runtime comparisons therefore include that profile difference.

Use the patched interpreter directly to run the reproducer:

```sh
/path/to/.venv-patched/bin/python repro.py --sweep --book-cases --output patched-results
```

The script's inline uv metadata pins the official release. `uv run repro.py` is the convenient **released-wheel** reproduction; use the explicit patched interpreter when testing your compiled wheel.

For the same-profile control, build `results/first-patch-for-control.patch` on a separate clean checkout of the same release commit with the same commands and compiler profile. It contains the earlier multi-input apply fix; the extended patch adds the composed-sum path.

For identical-input comparisons, generate once with the released environment, then call the worker under each interpreter using the same `--data` path, for example:

```sh
/path/to/released/bin/python repro.py --output released-results
/path/to/patched/bin/python repro.py --worker polars-cov-streaming \
  --data released-results/synthetic.parquet --save-result patched-cov.parquet
```

The validation here used three alternating fresh-process runs per case on that same Parquet file. All 26 Polars output tables, including `cov_alt()`, were bit-for-bit identical. The original full-size Polars result was also bit-for-bit identical. Independent DuckDB executions can differ by floating-point rounding.

Upstream test coverage includes dynamic windows, grouped apply, covariance/correlation, arithmetic and aggregation tests, plus a new centered-product regression test. The exact passed count is in `results/patch-validation.json`. The supplied `edgecases.py` covers **240** before/after scenarios, and `cov_alt_cases.py` adds **384** composed-expression scenarios. This is targeted regression coverage, not the complete Polars suite.

`results/build-provenance.json`, `results/patch-validation.json` and `results/upstream-tests.txt` contain the detailed evidence. The additional `cov_alt()` fix and its scope are described in `SOURCE_NOTES.md`.

To reproduce the composed-expression checks with both interpreters:

```sh
/path/to/released/bin/python cov_alt_cases.py --output released-cases
/path/to/patched/bin/python cov_alt_cases.py --output patched-cases
/path/to/patched/bin/python compare_results.py released-cases patched-cases
```

`compare_results.py` checks schemas, keys, nulls, values and non-NaN floating-point bits without NumPy. It reports NaN sign/payload differences separately. For upstream tests, install `pytest hypothesis pytest-xdist numpy pandas` in the patched environment, then run from `py-polars/`:

```sh
/path/to/patched/bin/python -m pytest -q -o addopts= --import-mode=importlib \
  tests/unit/operations/test_group_by_dynamic.py \
  tests/unit/operations/map/test_map_groups.py \
  tests/unit/functions/test_functions.py::test_cov \
  tests/unit/functions/test_functions.py::test_corr \
  tests/unit/functions/test_functions.py::test_null_handling_correlation \
  tests/unit/functions/test_functions.py::test_corr_spearman_float_dtype_26335 \
  tests/unit/operations/aggregation/test_aggregations.py \
  tests/unit/operations/arithmetic/test_arithmetic.py
```

## Separate candidate patches

Both candidate fixes change Polars' Rust execution engine. The Python `cov_alt()` formula is unchanged.

- [Multi-input apply patch](multi-input-apply.patch): lazy window views for covariance, correlation and other multi-input grouped functions.
- [Composed-sum patch](composed-window-sum.patch): evaluate and sum Float64 arithmetic over columns and column means one overlapping window at a time.
- [Combined patch](candidate-fix.patch): the exact source change used for the final build and numerical validation.

The separate patches are supplied for review. The final benchmark and regression results were recorded with both applied; the earlier multi-input-only build also provides the same-profile control.
