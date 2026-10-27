"""Compare values and non-NaN float bits, recording NaN bit differences separately."""
import argparse
import json
import math
from pathlib import Path
import struct

import polars as pl
from polars.testing import assert_frame_equal


def same_bits(left, right, dtype):
    if left is None or right is None:
        return left is right, left is right, 0
    if isinstance(dtype, pl.List):
        results = [
            same_bits(a, b, dtype.inner) for a, b in zip(left, right)
        ]
        return all(x[0] for x in results), all(x[1] for x in results), sum(x[2] for x in results)
    if dtype.is_float():
        fmt = '<f' if dtype == pl.Float32 else '<d'
        equal = struct.pack(fmt, left) == struct.pack(fmt, right)
        nan = math.isnan(left) and math.isnan(right)
        return equal, nan or equal, int(nan and not equal)
    return True, True, 0


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('before', type=Path)
p.add_argument('after', type=Path)
p.add_argument('--output', type=Path)
args = p.parse_args()
if args.before.is_dir() and args.after.is_dir():
    old = {f.name: f for f in args.before.glob('*.parquet')}
    new = {f.name: f for f in args.after.glob('*.parquet')}
    assert old and old.keys() == new.keys(), 'Parquet filenames differ or no files found'
    pairs = [(name, old[name], new[name]) for name in sorted(old)]
else:
    pairs = [(args.before.name, args.before, args.after)]
checks = []
for name, before, after in pairs:
    a, b = pl.read_parquet(before), pl.read_parquet(after)
    assert_frame_equal(a, b, check_exact=True)
    bits = {}
    non_nan_bits = {}
    nan_differences = {}
    for col, dtype in a.schema.items():
        if dtype.is_float() or isinstance(dtype, pl.List):
            results = [same_bits(x, y, dtype) for x, y in zip(a[col].to_list(), b[col].to_list())]
            bits[col] = all(x[0] for x in results)
            non_nan_bits[col] = all(x[1] for x in results)
            nan_differences[col] = sum(x[2] for x in results)
    assert all(non_nan_bits.values()), f'Non-NaN floating-point bits differ: {name}: {non_nan_bits}'
    checks.append(dict(file=name, rows=a.height, exact=True, floating_point_bits_equal=bits,
                       non_nan_float_bits_equal=non_nan_bits,nan_bit_pattern_differences=nan_differences))
report = dict(files=len(checks), rows=sum(c['rows'] for c in checks), all_exact=True,
              all_non_nan_float_bits_equal=True,
              all_float_bits_equal=all(all(c['floating_point_bits_equal'].values()) for c in checks),
              nan_bit_pattern_differences=sum(sum(c['nan_bit_pattern_differences'].values()) for c in checks),
              checks=checks)
if args.output:
    args.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'checks'}))
