"""Preserve public observations, bootstrap positions and isolated fit-validation fixtures."""

import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from input_common import REPORT, ROOT, PYTHON, digest, save_exact, toml_value

SUBJECT = '21_tail_risk'


def public_data():
    record = tomllib.loads((ROOT/'notebooks'/'julia'/'data'/'inputs.toml').read_text())['datasets']['fred_nasdaq']
    path = ROOT/record['original']
    assert digest(path)==record['original_sha256']=='2e988948ffcb108da4902dfedaa5df756e7e102dc523a99a8916a0533291f2d7'
    data = pd.read_csv(path)
    data['Date'] = pd.to_datetime(data.observation_date)
    data['NASDAQ Composite'] = pd.to_numeric(data.NASDAQCOM,errors='coerce')
    data = data.dropna(subset=['NASDAQ Composite']).sort_values('Date').reset_index(drop=True)
    data['Daily log return (%)'] = 100*np.log(data['NASDAQ Composite']/data['NASDAQ Composite'].shift())
    data['Loss (%)'] = -data['Daily log return (%)']
    data['Simple return'] = np.expm1(data['Daily log return (%)']/100)
    returns = data.iloc[1:].copy()
    design,held = returns[returns.Date<='2014-12-31'],returns[returns.Date>='2015-01-01']
    assert len(design)==5287 and len(held)==2767
    return data,returns,design,held


def main():
    _,returns,design,held = public_data()
    destination = ROOT/'notebooks'/'julia'/'data'/'chapter21_random_inputs'
    destination.mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(2202)
    positions = np.column_stack([np.concatenate([start+np.arange(20) for start in rng.integers(0,len(design)-20+1,size=25)])+1 for _ in range(300)])
    path = destination/'drawdown_bootstrap.csv'
    save_exact(path,pd.DataFrame(positions,columns=['sample_'+str(index) for index in range(1,301)]).to_csv(index=False).encode())
    record = dict(path=str(path),name=path.name,sha256=digest(path),rows=500,columns=300,values=150000,seed=2202,primitive='One-based moving-block row positions. All public return selection, wealth paths and statistics are computed in Julia.')
    manifest = destination/'inputs.toml'
    save_exact(manifest,('[files."drawdown_bootstrap.csv"]\nsha256 = "'+record['sha256']+'"\nrows = 500\ncolumns = 300\n').encode())
    preparation = dict(files=[record],manifest=dict(path=str(manifest),sha256=digest(manifest)),script_sha256=digest(Path(__file__)))
    (REPORT/'mif_chapter21_random_input_preparation.json').write_text(json.dumps(preparation,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(primitive=record,manifest=preparation['manifest']),indent=2))


if __name__=='__main__':
    main()
