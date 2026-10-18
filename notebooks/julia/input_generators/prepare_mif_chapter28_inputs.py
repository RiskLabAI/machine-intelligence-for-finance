"""Preserve the public network experiments' normal, uniform, and node-index draws."""

import json

import numpy as np
import pandas as pd

from input_common import REPORT, ROOT, PYTHON, digest, save_exact, toml_value

SUBJECT='28_network_clearing'
STEMS=['28_01_proportional_clearing','28_02_network_structure_shocks','28_03_liability_concentration_systemic_loss']


def main():
    destination=ROOT/'notebooks'/'julia'/'data'/'chapter28_random_inputs'
    destination.mkdir(parents=True,exist_ok=True)
    repeated_normals,repeated_uniforms,nodes,concentration_normals,concentration_uniforms=[],[],[],[],[]
    for draw in range(200):
        rng=np.random.default_rng(290200+draw)
        repeated_normals.append(rng.standard_normal(12))
        repeated_uniforms.append(rng.random(12))
        nodes.append([int(rng.integers(0,12))+1])
        rng=np.random.default_rng(290300+draw)
        concentration_normals.append(rng.standard_normal(12))
        concentration_uniforms.append(rng.random(12))
    records=[]
    values=[('baseline_normals.csv',[np.random.default_rng(2902).standard_normal(12)]),('repeated_normals.csv',repeated_normals),('repeated_uniforms.csv',repeated_uniforms),('repeated_nodes.csv',nodes),('concentration_normals.csv',concentration_normals),('concentration_uniforms.csv',concentration_uniforms)]
    for name,rows in values:
        array=np.asarray(rows)
        path=destination/name
        save_exact(path,pd.DataFrame(array,columns=[f'x{i+1}' for i in range(array.shape[1])]).to_csv(index=False).encode())
        records.append(dict(path=str(path),name=name,sha256=digest(path),rows=array.shape[0],columns=array.shape[1]))
    manifest=destination/'inputs.toml'
    save_exact(manifest,''.join('[files."'+r['name']+'"]\nsha256 = "'+r['sha256']+'"\nrows = '+str(r['rows'])+'\ncolumns = '+str(r['columns'])+'\n\n' for r in records).encode())
    result=dict(files=records,manifest=dict(path=str(manifest),sha256=digest(manifest)),primitive='Original NumPy standard normals and unit uniforms from seeds 2902, 290200:290399 and 290300:290499, plus one-based shocked-node indices. Native Julia computes all transformed observations, balance sheets, clearing payments, default decisions and summaries.')
    (REPORT/'mif_chapter28_random_input_preparation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(files=len(records),values=sum(r['rows']*r['columns'] for r in records),manifest=result['manifest']),indent=2))


if __name__=='__main__':
    main()
