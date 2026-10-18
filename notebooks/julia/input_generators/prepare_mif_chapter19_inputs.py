"""Preserve the original PBO chapter's random innovations and bootstrap rows."""

import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from input_common import REPORT, ROOT, PYTHON, digest, save_exact, toml_value

SUBJECT = '19_backtest_overfitting'


def moving_block_bootstrap_indices(count,block,generator):
    block=max(int(block),1)
    starts=generator.integers(0,count,size=int(np.ceil(count/block)))
    return (starts[:,None]+np.arange(block)[None,:]).ravel()[:count]%count


def public_data():
    manifest = tomllib.loads((ROOT/'notebooks'/'julia'/'data'/'inputs.toml').read_text())['datasets']['fred_sp500']
    path = ROOT/manifest['original']
    assert digest(path)==manifest['original_sha256']=='2098b209ee3afdb3b2c1e783ac90a2bb7dd3865feb416d0fabedbb4a7a542754'
    raw = pd.read_csv(path)
    raw['observation_date'] = pd.to_datetime(raw.observation_date)
    prices = raw.set_index('observation_date').SP500.dropna().astype(float).sort_index()
    returns = np.log(prices).diff()
    columns,records = {},[]
    for fast in [3,5,10,20]:
        for slow in [30,60,120,240]:
            fast_average = prices.ewm(span=fast,adjust=False,min_periods=fast).mean()
            slow_average = prices.ewm(span=slow,adjust=False,min_periods=slow).mean()
            position = np.sign(fast_average-slow_average).replace(0,np.nan).ffill().shift(1)
            for orientation,multiplier in [('Trend',1.),('Reversal',-1.)]:
                name = orientation+' '+str(fast)+'/'+str(slow)
                columns[name] = multiplier*position*returns
                records.append([len(records)+1,orientation,fast,slow,name])
    return prices,pd.DataFrame(columns).dropna(),pd.DataFrame(records,columns=['Trial','Rule','Fast span','Slow span','Name'])


def main():
    _,performance,_ = public_data()
    destination = ROOT/'notebooks'/'julia'/'data'/'chapter19_random_inputs'
    destination.mkdir(parents=True,exist_ok=True)
    records = []

    def save(name,values,primitive,seed):
        frame = pd.DataFrame(values,columns=['draw_'+str(index) for index in range(1,values.shape[1]+1)])
        path = destination/(name+'.csv')
        save_exact(path,frame.to_csv(index=False).encode('utf-8'))
        records.append(dict(path=str(path),name=path.name,sha256=digest(path),rows=values.shape[0],columns=values.shape[1],primitive=primitive,seed=int(seed)))

    save('preliminary_uniform',np.random.RandomState(42).rand(6,4),'Unmodified uniform draws before the two original deterministic row replacements.',42)
    save('null_one',np.random.default_rng(1).normal(0,1,(1000,50)),'Standard Gaussian innovations for the first null experiment.',1)
    save('null_two',np.random.RandomState(42).standard_normal((1000,50)),'Standard Gaussian innovations before the original mean 0.0001 and scale 0.01.',42)
    for condition in range(3):
        for observations in [120,240,480,960]:
            for repetition in range(40):
                seed = 100000*condition+1000*observations+repetition
                rng = np.random.default_rng(seed)
                volatility = rng.normal(0,.16,observations)
                common = rng.standard_t(7,observations)
                individual = rng.standard_t(7,(observations,16))
                save('simulation_'+str(condition)+'_'+str(observations)+'_'+str(repetition+1),np.column_stack([volatility,common,individual]),'Volatility innovations, common Student-t innovations and sixteen individual Student-t innovations, before all dynamic scaling and mean patterns.',seed)
        print(json.dumps(dict(condition=condition,files=len(records))),flush=True)
    for seed in range(12):
        save('noisy_'+str(seed),np.random.default_rng(seed).standard_normal((240,8)),'Gaussian innovations before the original scale and trial means.',seed)
    for name,count,bags,seed in [('noisy_bootstrap',240,25,0),('public_bootstrap',len(performance),40,17)]:
        rng = np.random.default_rng(seed)
        block = max(count//20,5)
        rows = np.column_stack([moving_block_bootstrap_indices(count,block,rng)+1 for _ in range(bags)])
        save(name,rows,'One-based moving-block bootstrap observation positions.',seed)
    manifest = destination/'inputs.toml'
    content = ''.join('[files."'+row['name']+'"]\nsha256 = "'+row['sha256']+'"\nrows = '+str(row['rows'])+'\ncolumns = '+str(row['columns'])+'\nseed = '+str(row['seed'])+'\n\n' for row in records)
    save_exact(manifest,content.encode('utf-8'))
    report = dict(status='PREPARED_PUBLIC_RANDOM_PRIMITIVES',files=records,manifest=dict(path=str(manifest),sha256=digest(manifest)),script_sha256=digest(Path(__file__)),observations=len(performance),trials=performance.shape[1],primitive='Only unmodified original random innovations and public moving-block bootstrap row positions. No simulated performance matrices, ranks, logits, PBO estimates or selected trials are supplied to Julia.')
    (REPORT/'mif_chapter19_random_input_preparation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(files=len(records),values=sum(row['rows']*row['columns'] for row in records),manifest_sha256=digest(manifest),public_observations=len(performance)),indent=2))


if __name__=='__main__':
    main()
