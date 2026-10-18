"""Preserve the original public NCO innovations and K-means initialization draws."""

import json

import numpy as np
import pandas as pd

from input_common import REPORT, ROOT, PYTHON, digest, save_exact, toml_value

SUBJECT='26_nested_cluster_optimization'
STEMS=['26_01_within_between_cluster_optimization','26_02_nco_estimation_error','26_03_hedge_exposure_diagnostics']


def block_covariance(blocks,size,within,between=.03):
    n=blocks*size
    correlation=np.full((n,n),between)
    for block in range(blocks):
        indices=slice(block*size,(block+1)*size)
        correlation[indices,indices]=within
    np.fill_diagonal(correlation,1.)
    scales=np.linspace(.10,.22,n)
    return correlation*np.outer(scales,scales),correlation


def primitives():
    for seed in [2201,2210,2220,2230,2250,2260]:
        seeds=np.random.RandomState(seed).randint(0,2147483647,30)
        draws=np.array([np.random.RandomState(value).random_sample(10) for value in seeds])
        yield f'kmeans_{seed}.csv',np.column_stack([seeds,draws])
    yield 'demonstration_normals.csv',np.random.default_rng(2202).standard_normal((180,24))
    for objective in range(2):
        for repetition in range(40):
            seed=222000+1000*objective+repetition
            yield f'repeated_{objective}_{repetition+1}.csv',np.random.default_rng(seed).standard_normal((180,24))
    for within in [.30,.60]:
        for training in [80,160,320]:
            for repetition in range(10):
                seed=223000+int(within*1000)+training+repetition
                yield f'sensitivity_{int(within*100)}_{training}_{repetition+1}.csv',np.random.default_rng(seed).standard_normal((training,24))
    yield 'timing_normals.csv',np.column_stack([np.random.default_rng(224000+job).standard_normal(2000) for job in range(20)])


def main():
    destination=ROOT/'notebooks'/'julia'/'data'/'chapter26_random_inputs'
    destination.mkdir(parents=True,exist_ok=True)
    records=[]
    for name,values in primitives():
        path=destination/name
        save_exact(path,pd.DataFrame(values,columns=[f'x{i+1}' for i in range(values.shape[1])]).to_csv(index=False).encode())
        records.append(dict(path=str(path),name=name,sha256=digest(path),rows=values.shape[0],columns=values.shape[1]))
    conventions=[]
    for within in [.30,.55,.60]:
        covariance,_=block_covariance(4,6,within,.04)
        u,s,v=np.linalg.svd(covariance)
        signs=np.sign(u[np.argmax(abs(u),axis=0),np.arange(24)]).astype(int)
        conventions.append(f'[families."{int(within*100)}"]\norientation = {signs.tolist()}\n')
    convention=destination/'coordinate_convention.toml'
    save_exact(convention,('\n'.join(conventions)+'\n').encode())
    records.append(dict(path=str(convention),name=convention.name,sha256=digest(convention),kind='known_covariance_coordinate_convention',rows=72,columns=1))
    manifest=destination/'inputs.toml'
    save_exact(manifest,''.join('[files."'+r['name']+'"]\nsha256 = "'+r['sha256']+'"\nrows = '+str(r['rows'])+'\ncolumns = '+str(r['columns'])+'\n\n' for r in records).encode())
    result=dict(files=records,manifest=dict(path=str(manifest),sha256=digest(manifest)),primitive='Original public standard-normal innovations, independent K-means initialization seeds and uniforms, and SVD coordinate signs for three declared known covariance designs. All samples, factors, trees, clusters, estimates and outcomes are computed natively.')
    (REPORT/'mif_chapter26_random_input_preparation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(files=len(records),primitive_values=sum(r['rows']*r['columns'] for r in records if r['name'].endswith('.csv')),manifest=result['manifest']),indent=2))


if __name__=='__main__':
    main()
