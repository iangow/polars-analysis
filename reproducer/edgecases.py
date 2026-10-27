"""Capture numerical/shape behavior in independent released and patched runtimes."""
import argparse
import json
import math
from pathlib import Path

import polars as pl

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
frames=[]
cases=[]
for pattern in ['finite','nulls','nan','constant','all-null']:
    x=[float(i%7-3) for i in range(32)]
    y=[math.sin(i*0.5) for i in range(32)]
    if pattern=='nulls':
        x=[None if i%5==0 else v for i,v in enumerate(x)]
        y=[None if i%7==0 else v for i,v in enumerate(y)]
    elif pattern=='nan':
        x[3]=float('nan');y[20]=float('nan')
    elif pattern=='constant':
        x=[2.0]*32
    elif pattern=='all-null':
        x=[None]*32;y=[None]*32
    df=pl.DataFrame({'g':[0]*16+[1]*16,'t':list(range(16))*2,
                     'x':pl.Series(x,dtype=pl.Float64),'y':pl.Series(y,dtype=pl.Float64)})
    for engine in ['streaming','in-memory']:
        for closed in ['left','right','both','none']:
            for label in ['left','right']:
                for ddof in [0,1,2]:
                    case=f'{pattern}-{engine}-{closed}-{label}-{ddof}'
                    q=(df.lazy().group_by_dynamic('t',every='2i',period='4i',offset='-4i',
                                                  group_by='g',closed=closed,label=label)
                        .agg(n=pl.len(),
                             cov=pl.cov('x','y',ddof=ddof),
                             self_cov=pl.cov('x','x',ddof=ddof),
                             variance=pl.col('x').var(ddof=ddof),
                             pearson=pl.corr('x','y'),
                             spearman=pl.corr('x','y',method='spearman'),
                             constant_cov=pl.cov(pl.col('x'),pl.lit(1.0),ddof=ddof),
                             group_mean_cov=pl.cov(pl.col('x'),pl.col('y').mean(),ddof=ddof),
                             custom=pl.map_groups(['x','y'],lambda s: float((s[0]*s[1]).sum()),
                                                  return_dtype=pl.Float64,returns_scalar=True),
                             custom_list=pl.map_groups(['x','y'],lambda s:s[0]+s[1],
                                                       return_dtype=pl.Float64),
                             mixed=pl.map_groups([pl.col('x'),pl.col('y').mean()],
                                                 lambda s:float(s[0].sum()+s[1].sum()),
                                                 return_dtype=pl.Float64,returns_scalar=True),
                             names=pl.map_groups([pl.col('x').alias('left'),pl.col('y').alias('right')],
                                                 lambda s:len(s[0].name)*100+len(s[1].name),
                                                 return_dtype=pl.Int64,returns_scalar=True))
                        .sort('g','t'))
                    frame=q.collect(engine=engine).with_columns(case=pl.lit(case))
                    frames.append(frame)
                    cases.append({'case':case,'rows':frame.height})
pl.concat(frames).write_parquet(args.output/'edgecase-results.parquet')
(args.output/'edgecase-cases.json').write_text(json.dumps({'cases':cases,'total':len(cases),'polars':pl.__version__},indent=2)+'\n')
print(json.dumps({'cases':len(cases),'rows':sum(x['rows'] for x in cases),'polars':pl.__version__}))
