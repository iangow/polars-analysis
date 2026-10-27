# Implementation analysis and tested candidate patch

The Polars 2.1 source suggests that the memory expansion occurs before the numerical covariance kernel:

1. The multi-input group-aware apply path calls `iter_groups()` on each input. For an unaggregated input, that calls `aggregated()`, which calls `agg_list()`. See [multi-input apply](https://github.com/pola-rs/polars/blob/7f902bd3ffd27ff82d14278a09a295ff2f7aed92/crates/polars-expr/src/expressions/apply.rs#L343-L403), [group iteration](https://github.com/pola-rs/polars/blob/7f902bd3ffd27ff82d14278a09a295ff2f7aed92/crates/polars-expr/src/expressions/group_iter.rs#L62-L74) and [aggregation context](https://github.com/pola-rs/polars/blob/7f902bd3ffd27ff82d14278a09a295ff2f7aed92/crates/polars-expr/src/expressions/mod.rs#L451-L473).
2. For numeric slice groups, `agg_list()` appends each window's values to a new list-values buffer. Overlapping values are therefore copied repeatedly. See [numeric list aggregation](https://github.com/pola-rs/polars/blob/7f902bd3ffd27ff82d14278a09a295ff2f7aed92/crates/polars-core/src/frame/group_by/aggregations/agg_list.rs#L91-L111).
3. Mean and variance have specialized aggregation paths, including rolling kernels that operate on the original values and window offsets. See [mean](https://github.com/pola-rs/polars/blob/7f902bd3ffd27ff82d14278a09a295ff2f7aed92/crates/polars-core/src/frame/group_by/aggregations/mod.rs#L991-L1027) and [variance](https://github.com/pola-rs/polars/blob/7f902bd3ffd27ff82d14278a09a295ff2f7aed92/crates/polars-core/src/frame/group_by/aggregations/mod.rs#L1056-L1090).
4. The single-input apply path already has a lazy iteration branch for overlapping groups. The corresponding multi-input path does not use it. See [the existing safeguard](https://github.com/pola-rs/polars/blob/7f902bd3ffd27ff82d14278a09a295ff2f7aed92/crates/polars-expr/src/expressions/apply.rs#L139-L150). The low-level covariance function itself uses an accumulator state; it does not require all windows to be copied at once ([kernel](https://github.com/pola-rs/polars/blob/7f902bd3ffd27ff82d14278a09a295ff2f7aed92/crates/polars-ops/src/chunked_array/cov.rs#L7-L29)).

This is a static source trace, not an allocation-profiler trace. However, it also predicts the observed scale: these 2,500,000 input rows participate in 150,000,000 window memberships before the output filters. Copying two Float64 values per membership takes 2,400,000,000 bytes (2.24 GiB), before vector capacity, input columns, output arrays and other overhead. The measured footprint is about 2.57 GiB.




## Compiled patch results

The candidate patch has now been built against the exact Polars 2.1.0 release commit, using its pinned `nightly-2026-09-01` toolchain, default full runtime features, 32-bit indices and the optimized `nodebug-release` profile. Python was 3.14.2 throughout.

The official wheel uses `dist-release` (fat LTO); the local build used thin LTO. Timings therefore include a build-profile difference. The before/after memory reduction and exact numerical comparisons are the principal checks here. Build provenance records the commit, patch hash, wheel hash and both imported native-module hashes.

All 26 Polars benchmark outputs and the 240 original edge cases are bit-for-bit identical before and after the patch. The 384 additional composed-expression scenarios have identical values, null/NaN positions and non-NaN floating-point bits. Some NaN bit patterns differ between the official and local builds; the same differences were present in the earlier compiled patch. 1674 upstream tests passed. The aligned DuckDB result has the same full keys and observation counts and agrees within floating-point precision. These checks do not constitute the entire Polars test suite.

| Query | Released seconds | Patched seconds | Released footprint, GiB | Patched footprint, GiB |
|---|---:|---:|---:|---:|
| Covariance, streaming | 0.739 | 0.181 | 2.573 | 0.297 |
| Covariance, in-memory | 0.658 | 0.159 | 2.570 | 0.297 |
| Covariance of x with itself | 0.818 | 0.162 | 2.519 | 0.243 |
| Pearson correlation | 0.786 | 0.174 | 2.570 | 0.298 |
| Spearman correlation | 8.296 | 7.850 | 2.573 | 0.298 |
| FFJR beta and alpha | 1.922 | 0.182 | 4.875 | 0.298 |
| Custom two-input aggregate | 2.151 | 1.587 | 2.573 | 0.297 |
| Book cov_alt expression | 2.164 | 0.295 | 4.892 | 0.298 |
| Variance of x and y | 0.075 | 0.077 | 0.297 | 0.296 |
| Mean of x | 0.046 | 0.067 | 0.242 | 0.242 |

## Composed covariance expression

The book's `cov_alt()` expression uses a separate [binary group-aware path](https://github.com/pola-rs/polars/blob/7f902bd3ffd27ff82d14278a09a295ff2f7aed92/crates/polars-expr/src/expressions/binary.rs#L277-L305). That path materializes the centered inputs and their product for every overlapping window. The first `ApplyExpr` patch left its footprint at about 4.89 GiB.

The extended patch recognizes Float64 arithmetic over columns, column means and literals feeding a sum on the original overlapping slice groups. It computes the means through the existing aggregation kernels, then evaluates and sums one window at a time. It preserves the expression's independent means, null handling, denominator and arithmetic order; it does not replace the expression with pairwise-complete covariance. The measured footprint is now **0.298 GiB**, versus **4.892 GiB** on released 2.1, with bit-for-bit identical output. Other expression types and group layouts retain their existing evaluation paths. This is a scoped candidate, not a general solution for every composed aggregation.

A separate comparison keeps the build profile fixed: both local wheels use `nodebug-release` with thin LTO. Across three alternating fresh-process runs, `cov_alt()` changed from 1.791 seconds and 4.856 GiB with the first patch to 0.299 seconds and 0.297 GiB with the extended patch. The output is bit-for-bit identical.
