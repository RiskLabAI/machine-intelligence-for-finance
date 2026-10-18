"""Preserve public hedging normal draws and initial weights in the original Float64 design."""

import json
import sys
from pathlib import Path

import pandas as pd
import torch

from input_common import REPORT, ROOT, PYTHON, digest, save_exact, toml_value

sys.path.insert(0,str(PYTHON.parent))
import RiskLabAI.backtest.deep_hedging as public_hedging
assert Path(public_hedging.__file__).resolve().is_relative_to(PYTHON)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
SUBJECT='31_deep_hedging'
STEMS=['31_01_hedging_cashflows_entropic_risk','31_02_adapted_deep_hedge_training','31_03_comparators_sensitivity']


def main():
    destination=ROOT/'notebooks'/'julia'/'data'/'chapter31_random_inputs'
    destination.mkdir(parents=True,exist_ok=True)
    inputs=[(33001,8192,20),(33999,8192,20),(33011,8192,20),(33011,2048,20),(33021,2048,20),(33031,2048,20)]
    inputs += [(seed,count,steps) for seed,count in [(33701,384),(33703,2048)] for steps in [10,20,40]]
    records=[]
    for seed,count,steps in inputs:
        generator=torch.Generator(device='cpu').manual_seed(seed)
        draws=torch.randn(count,steps,generator=generator,dtype=torch.float64).numpy()
        name=f'normal_{seed}_{count}_{steps}.csv'
        path=destination/name
        save_exact(path,pd.DataFrame(draws,columns=[f'z{i+1}' for i in range(steps)]).to_csv(index=False).encode())
        records.append(dict(path=str(path),name=name,sha256=digest(path),rows=count,columns=steps,seed=seed,kind='normal'))
    for seed in [33012,33022,33032,33702]:
        model=public_hedging.DeepHedge(16,seed=seed)
        content='\n'.join(name+' = '+toml_value(value.detach().numpy().tolist()) for name,value in zip(['weight1','bias1','weight2','bias2','weight3','bias3'],model.parameters()))+'\n'
        path=destination/f'initial_{seed}.toml'
        save_exact(path,content.encode())
        records.append(dict(path=str(path),name=path.name,sha256=digest(path),seed=seed,kind='initial_parameters'))
    manifest=destination/'inputs.toml'
    save_exact(manifest,''.join('[files."'+r['name']+'"]\nsha256 = "'+r['sha256']+'"\nseed = '+str(r['seed'])+'\nkind = "'+r['kind']+'"\n\n' for r in records).encode())
    result=dict(files=records,manifest=dict(path=str(manifest),sha256=digest(manifest)),python_source=dict(path=str(Path(public_hedging.__file__)),sha256=digest(Path(public_hedging.__file__))),primitive='Original Float64 PyTorch normal draws and unfitted parameters from the public notebook seeds. Native Julia computes generated paths, positions, gradients, fitted weights, losses and independent evaluation; no Python fitted outputs are native inputs.')
    (REPORT/'mif_chapter31_random_input_preparation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(files=len(records),normal_values=sum(r.get('rows',0)*r.get('columns',0) for r in records),manifest=result['manifest']),indent=2))


if __name__=='__main__':
    main()
