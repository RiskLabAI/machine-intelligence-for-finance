"""Freeze original NumPy draw streams for the native predictor-clustering chapter."""

import json
import io
from pathlib import Path

import numpy as np
import pandas as pd

from input_common import REPORT, ROOT, PYTHON, digest, save_exact, toml_value


def main():
    root = ROOT/'notebooks'/'julia'
    destination = root/'data'/'chapter15_random_inputs'
    destination.mkdir(exist_ok=True)
    files = []

    def save(name,values,columns=None):
        path = destination/(name+'.csv')
        content = pd.DataFrame(values,columns=columns).to_csv(index=False,float_format='%.17g',lineterminator='\n').encode('utf-8')
        if path.exists():
            assert path.read_bytes()==content
        else:
            path.write_bytes(content)
        files.append(dict(path=str(path),name=path.name,sha256=digest(path)))

    voters = [1,3,5,11,21,51,101]
    rng = np.random.default_rng(13001)
    controlled = []
    for rho in [0.,.05,.15,.35]:
        for n in voters:
            if rho==0:
                votes = rng.binomial(n,.6,80000)
            else:
                concentration = rho**-1-1
                probability = rng.beta(.6*concentration,.4*concentration,80000)
                votes = rng.binomial(n,probability)
            controlled.append(votes)
    save('controlled_correct_votes',np.array(controlled).T)
    source = root/'data'/'french_49_industries.csv'
    lines = source.read_text(encoding='latin1').splitlines()
    start = next(index for index,line in enumerate(lines) if line.startswith(',Agric'))
    end = next(index for index in range(start+1,len(lines)) if not lines[index].strip())
    raw = pd.read_csv(io.StringIO('\n'.join(lines[start:end])))
    dates = pd.to_datetime(raw.iloc[:,0].astype(str),format='%Y%m%d')
    returns = raw.iloc[:,1:].set_index(dates).replace([-99.99,-999.],np.nan).loc['2019-01-01':'2025-12-31'].dropna()/100
    target = (returns.mean(axis=1).iloc[5:]>0).astype(int).to_numpy()
    split = int(.7*len(target))
    bagging_seeds = np.random.RandomState(13011).randint(0,2147483647,101)
    plans = []
    for seed in bagging_seeds:
        generator = np.random.RandomState(seed)
        model_seed = generator.randint(0,2147483647)
        split_seed = np.random.RandomState(model_seed).randint(0,2147483647)
        samples = np.random.RandomState(seed).randint(0,split,500)
        plans.append(np.r_[model_seed,split_seed,np.bincount(samples,minlength=split)])
    save('public_bagging_plan',np.array(plans).T)
    rng = np.random.default_rng(13012)
    subsets = [np.r_[n,rep+1,rng.permutation(101)+1] for n in voters for rep in range(60)]
    save('tree_subsets',subsets,['voters','repetition']+[f'tree{i+1}' for i in range(101)])
    rng = np.random.RandomState(13013)
    save('test_bootstrap',np.array([rng.choice(527,527,replace=True)+1 for _ in range(500)]).T)
    for seed in [13021,13031]+list(range(13100,13124))+list(range(13400,13412)):
        seeds = np.random.RandomState(seed).randint(0,2147483647,220)
        draws = np.array([np.random.RandomState(value).random_sample(45) for value in seeds])
        save(f'kmeans_{seed}',np.column_stack([seeds,draws]),['seed']+[f'draw{i+1}' for i in range(45)])
    rng = np.random.default_rng(13032)
    positions = []
    for _ in range(24):
        starts = rng.integers(0,1760,88)
        positions.append(np.concatenate([(start+np.arange(20))%1760 for start in starts])[:1760]+1)
    save('stability_positions',np.array(positions).T)
    for rep in range(12):
        rng = np.random.RandomState(13200+rep)
        cuts = np.sort(rng.choice(range(1,5),4,replace=False))
        save(f'block_{rep+1}_cuts',cuts[:,None],['cut'])
        for block in range(5):
            common = rng.normal(size=100)
            noise = rng.normal(size=(100,6))
            save(f'block_{rep+1}_part_{block+1}',np.column_stack([common,noise]))
        rng.choice(range(1,1),0,replace=False)
        common = rng.normal(size=465)
        noise = rng.normal(size=(465,30))
        save(f'block_{rep+1}_market',np.column_stack([common,noise]))
        save(f'block_{rep+1}_permutation',(np.random.default_rng(13300+rep).permutation(30)+1)[:,None],['position'])
    lines = ['title = "Fixed public book random primitives for native Julia"','']
    for row in files:
        lines += ['[files.'+json.dumps(row['name'])+']','sha256 = '+json.dumps(row['sha256']),'']
    manifest = destination/'inputs.toml'
    content = ('\n'.join(lines)+'\n').encode('utf-8')
    if manifest.exists():
        assert manifest.read_bytes()==content
    else:
        manifest.write_bytes(content)
    result = dict(status='PREPARED_PUBLIC_RANDOM_PRIMITIVES',files=files,manifest=str(manifest),manifest_sha256=digest(manifest),script_sha256=digest(Path(__file__)))
    (REPORT/'mif_chapter15_random_input_preparation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],files=len(files),manifest_sha256=result['manifest_sha256']),indent=2),flush=True)


if __name__=='__main__':
    main()
