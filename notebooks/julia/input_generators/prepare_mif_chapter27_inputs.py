"""Preserve original random comparison points, training batches, and bootstrap block starts."""

import json

import numpy as np
import pandas as pd
import torch

from input_common import REPORT, ROOT, PYTHON, digest, save_exact, toml_value

SUBJECT='27_decision_focused_learning'
STEMS=['27_01_decision_loss_spo_plus','27_02_public_decision_focused_learning','27_03_robust_portfolio_choice']


def main():
    destination=ROOT/'notebooks'/'julia'/'data'/'chapter27_random_inputs'
    destination.mkdir(parents=True,exist_ok=True)
    generator=torch.Generator().manual_seed(2802)
    batches=np.array([torch.randint(0,697,(32,),generator=generator).numpy()+1 for _ in range(80)])
    rng=np.random.default_rng(2803)
    blocks=np.array([rng.integers(0,1258-20+1,size=63)+1 for _ in range(300)])
    records=[]
    for name,values in [('comparison_points.csv',np.random.default_rng(2801).normal(size=(100,2))),('training_batches.csv',batches),('bootstrap_blocks.csv',blocks)]:
        path=destination/name
        save_exact(path,pd.DataFrame(values,columns=[f'x{i+1}' for i in range(values.shape[1])]).to_csv(index=False).encode())
        records.append(dict(path=str(path),name=name,sha256=digest(path),rows=values.shape[0],columns=values.shape[1]))
    manifest=destination/'inputs.toml'
    save_exact(manifest,''.join('[files."'+r['name']+'"]\nsha256 = "'+r['sha256']+'"\nrows = '+str(r['rows'])+'\ncolumns = '+str(r['columns'])+'\n\n' for r in records).encode())
    result=dict(files=records,manifest=dict(path=str(manifest),sha256=digest(manifest)),primitive='Original NumPy comparison points, one-based PyTorch training indices from seed 2802, and one-based NumPy bootstrap block starts from seed 2803. All features, predictions, gradients, model updates, sampled observations, estimates and portfolios are computed natively.')
    (REPORT/'mif_chapter27_random_input_preparation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
