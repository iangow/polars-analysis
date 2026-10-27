"""Capture composed-sum behavior for released/patched runtime comparison."""
import argparse
import json
import math
from pathlib import Path

import polars as pl

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
cases = []

for pattern in ['finite', 'nulls', 'all-null', 'nan', 'infinity', 'constant', 'large-offset', 'empty']:
    x = [float(i % 7 - 3) for i in range(32)]
    y = [math.sin(i * 0.5) for i in range(32)]
    if pattern == 'nulls':
        x = [None if i % 5 == 0 else v for i, v in enumerate(x)]
        y = [None if i % 7 == 0 else v for i, v in enumerate(y)]
    elif pattern == 'all-null':
        x = [None] * 32; y = [None] * 32
    elif pattern == 'nan':
        x[3] = float('nan'); y[20] = float('nan')
    elif pattern == 'infinity':
        x[3] = float('inf'); y[20] = float('-inf')
    elif pattern == 'constant':
        x = [2.0] * 32
    elif pattern == 'large-offset':
        x = [v + 1e12 for v in x]; y = [v + 1e8 for v in y]
    elif pattern == 'empty':
        x = []; y = []
    for dtype in [pl.Float64, pl.Float32]:
        n = len(x)
        df = pl.DataFrame({'g': pl.Series([i // 16 for i in range(n)], dtype=pl.Int64),
                           't': pl.Series([i % 16 for i in range(n)], dtype=pl.Int64),
                           'x': pl.Series(x, dtype=dtype), 'y': pl.Series(y, dtype=dtype)})
        # Keep multiple input chunks; the lazy window iterator must handle these too.
        df = pl.concat([df[:9], df[9:21], df[21:]], rechunk=False)
        a, b = pl.col('x'), pl.col('y')
        center_a, center_b = a - a.mean(), b - b.mean()
        expressions = dict(
            cov_alt=(center_a * center_b).sum() / (pl.len() - 1),
            centered_sum=center_a.sum(),
            centered_square=(center_a * center_a).sum(),
            literal=(center_a * (b - b.mean()) + 2.0).sum(),
            alias=((a - a.mean()).alias('renamed') * center_b).sum(),
            scalar_only=(a.mean() * b.mean()).sum(),
            raw_only=(a * b).sum(),
            mixed_division=(a / (b.mean() + 3.0)).sum(),
            filtered=((a.filter(a > 0) - a.filter(a > 0).mean()) * b.mean()).sum(),
            list_leaf=((a - a.mean()).sort() * center_b).sum(),
            length_scalar=(a - pl.len()).sum(),
        )
        for engine in ['streaming', 'in-memory']:
            for period in ['1i', '4i', '20i']:
                for closed in ['left', 'right', 'both', 'none']:
                    name = f'{pattern}-{dtype}-{engine}-{period}-{closed}'
                    result = (df.lazy().group_by_dynamic('t', every='2i', period=period,
                              offset='-4i', group_by='g', closed=closed, label='right')
                              .agg(**expressions, n=pl.len()).sort('g', 't').collect(engine=engine))
                    result.write_parquet(args.output / (name + '.parquet'))
                    cases.append({'case': name, 'rows': result.height})
(args.output / 'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
print(json.dumps({'cases': len(cases), 'rows': sum(c['rows'] for c in cases)}))
