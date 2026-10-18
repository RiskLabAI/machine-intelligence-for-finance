"""Save only fixed bootstrap positions and Gaussian-mixture primitives for Chapter 22."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from input_common import REPORT, ROOT, PYTHON, digest, save_exact, toml_value

SUBJECT='22_position_sizing'
STEMS=['22_01_signal_calibration_position_sizing','22_03_drawdowns_strategy_risk_statistics']


def public_data():
    path=ROOT/'data'/'public'/'fred'/'SP500_2014-01-02_2025-12-31.csv'
    assert digest(path)=='2098b209ee3afdb3b2c1e783ac90a2bb7dd3865feb416d0fabedbb4a7a542754'
    frame=pd.read_csv(path,index_col=0,parse_dates=True)
    return pd.to_numeric(frame.SP500,errors='coerce').dropna().sort_index()


def primitives():
    rng=np.random.default_rng(20260928)
    draws=np.column_stack([((rng.integers(0,2300,size=25)[:,None]+np.arange(20)[None,:])%2300).ravel()+1 for _ in range(300)])
    yield 'drawdown_bootstrap.csv',pd.DataFrame(draws,columns=['sample_'+str(i) for i in range(1,301)])
    rng=np.random.RandomState(20260928)
    for name,weight in [('half',.5),('biased',.8)]:
        count=int(weight*1000)
        normals=np.concatenate([rng.standard_normal(count),rng.standard_normal(1000-count)])
        order=np.arange(1000)
        rng.shuffle(order)
        yield 'mixture_'+name+'.csv',pd.DataFrame(dict(normal=normals,permutation=order+1))


def main():
    destination=ROOT/'notebooks'/'julia'/'data'/'chapter22_random_inputs'
    destination.mkdir(parents=True,exist_ok=True)
    records=[]
    for name,frame in primitives():
        path=destination/name
        save_exact(path,frame.to_csv(index=False).encode())
        records.append(dict(path=str(path),name=name,sha256=digest(path),rows=len(frame),columns=len(frame.columns)))
    manifest=destination/'inputs.toml'
    save_exact(manifest,''.join('[files."'+r['name']+'"]\nsha256 = "'+r['sha256']+'"\nrows = '+str(r['rows'])+'\ncolumns = '+str(r['columns'])+'\n\n' for r in records).encode())
    result=dict(files=records,manifest=dict(path=str(manifest),sha256=digest(manifest)),seed=20260928,primitive='One-based circular bootstrap positions, standard-normal draws and one-based shuffle positions. No fitted values or derived statistics are supplied to native notebooks.')
    (REPORT/'mif_chapter22_random_input_preparation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
