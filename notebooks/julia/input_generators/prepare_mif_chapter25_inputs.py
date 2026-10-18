"""Preserve the declared Generator innovations and known-covariance coordinate signs."""

import json

import numpy as np
import pandas as pd

from input_common import REPORT, ROOT, PYTHON, digest, save_exact, toml_value

SUBJECT='25_hierarchical_risk_parity'
STEMS=['25_01_distances_linkage_quasi_diagonalization','25_02_recursive_bisection_cluster_risk','25_03_monte_carlo_inverse_covariance_comparison']


def clustered_covariance(n,within):
    correlation=np.full((n,n),.05)
    for block in range(4):
        indices=slice(block*(n//4),(block+1)*(n//4))
        correlation[indices,indices]=within
    np.fill_diagonal(correlation,1.)
    volatility=np.linspace(.12,.28,n)
    return correlation*np.outer(volatility,volatility),correlation


def primitives():
    yield 'repeated_normals.csv',np.random.default_rng(2103).standard_normal((60*126,24))
    rng=np.random.default_rng(2113)
    for n in [12,24]:
        for training in [63,126,252]:
            for within in [.3,.6]:
                yield f'sensitivity_{n}_{training}_{int(within*100)}.csv',rng.standard_normal((20*training,n))
    yield 'turnover_normals.csv',np.random.default_rng(2123).standard_normal((1000,24))


def main():
    destination=ROOT/'notebooks'/'julia'/'data'/'chapter25_random_inputs'
    destination.mkdir(parents=True,exist_ok=True)
    records=[]
    for name,values in primitives():
        path=destination/name
        frame=pd.DataFrame(values,columns=['x'+str(i) for i in range(1,values.shape[1]+1)])
        save_exact(path,frame.to_csv(index=False).encode())
        records.append(dict(path=str(path),name=name,sha256=digest(path),rows=len(frame),columns=len(frame.columns)))
    conventions=[]
    for n in [12,24]:
        for within in [.3,.6]:
            covariance,_=clustered_covariance(n,within)
            u,s,v=np.linalg.svd(covariance)
            signs=np.sign(u[np.argmax(abs(u),axis=0),np.arange(n)]).astype(int)
            z=np.random.default_rng(2103).standard_normal((126,n))
            reference=np.random.default_rng(2103).multivariate_normal(np.zeros(n),covariance,size=126)
            np.testing.assert_array_equal(reference,z@(u*np.sqrt(s)).T)
            conventions.append(f'[families."{n}_{int(within*100)}"]\norientation = {signs.tolist()}\n')
    convention=destination/'coordinate_convention.toml'
    save_exact(convention,('\n'.join(conventions)+'\n').encode())
    records.append(dict(path=str(convention),name=convention.name,sha256=digest(convention),kind='known_covariance_coordinate_convention',rows=72,columns=1))
    manifest=destination/'inputs.toml'
    save_exact(manifest,''.join('[files."'+r['name']+'"]\nsha256 = "'+r['sha256']+'"\nrows = '+str(r['rows'])+'\ncolumns = '+str(r['columns'])+'\n\n' for r in records).encode())
    result=dict(files=records,manifest=dict(path=str(manifest),sha256=digest(manifest)),seeds=[2103,2113,2123],primitive='Unmodified standard-normal innovations from the declared public Generator streams. Signs specify the SVD coordinate orientation of four known covariance designs only. Native Julia constructs all covariance factors, samples, trees, fits and reported results.')
    (REPORT/'mif_chapter25_random_input_preparation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(files=len(records),primitive_values=sum(r['rows']*r['columns'] for r in records if r['name'].endswith('.csv')),manifest=result['manifest']),indent=2))


if __name__=='__main__':
    main()
