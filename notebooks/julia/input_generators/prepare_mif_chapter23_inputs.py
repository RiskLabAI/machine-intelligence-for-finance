"""Preserve only the public random inputs for the portfolio-geometry experiments."""

import io
import json
import tomllib
import zipfile

import numpy as np
import pandas as pd

from input_common import REPORT, ROOT, PYTHON, digest, save_exact, toml_value

SUBJECT='23_portfolio_geometry'
STEMS=['23_01_mean_variance_geometry','23_02_covariance_mean_sensitivity','23_03_benchmark_stress_diagnostics']


def public_data():
    specification=tomllib.loads((ROOT/'notebooks'/'julia'/'data'/'inputs.toml').read_text(encoding='utf-8'))['datasets']['french_49_industries']
    path=ROOT/specification['original']
    assert digest(path)==specification['original_sha256']=='8f394fe34bea54d41b9aafed410425ee8f8e252ede3c71a7c1cd20bab83040de'
    with zipfile.ZipFile(path) as archive:
        lines=archive.read(archive.namelist()[0]).decode('latin1').splitlines()
    start=next(i for i,line in enumerate(lines) if line.startswith(',Agric'))
    end=next(i for i in range(start+1,len(lines)) if not lines[i].strip())
    frame=pd.read_csv(io.StringIO('\n'.join(lines[start:end])))
    frame=frame.rename(columns={frame.columns[0]:'Date'})
    frame.Date=pd.to_datetime(frame.Date.astype(str),format='%Y%m%d')
    return (frame.set_index('Date').replace([-99.99,-999.],np.nan)/100).loc['2019-01-01':'2025-12-31'].dropna()


def primitives():
    rng=np.random.RandomState(42)
    order=np.arange(30)
    rng.shuffle(order)
    deviations=rng.uniform(.05,.2,30)
    normals=rng.standard_normal(30)
    yield 'true_matrix_draws.csv',pd.DataFrame(dict(permutation=order+1,standard_deviation=deviations,normal=normals))
    yield 'sample_normals.csv',pd.DataFrame(rng.standard_normal((250,30)),columns=['x'+str(i) for i in range(1,31)])
    for index in range(40):
        values=np.random.RandomState(2000+index).standard_normal((250,30))
        yield 'repeated_normals_'+str(index+1)+'.csv',pd.DataFrame(values,columns=['x'+str(i) for i in range(1,31)])


def main():
    destination=ROOT/'notebooks'/'julia'/'data'/'chapter23_random_inputs'
    destination.mkdir(parents=True,exist_ok=True)
    records=[]
    for name,frame in primitives():
        path=destination/name
        save_exact(path,frame.to_csv(index=False).encode())
        records.append(dict(path=str(path),name=name,sha256=digest(path),rows=len(frame),columns=len(frame.columns)))
    manifest=destination/'inputs.toml'
    save_exact(manifest,''.join('[files."'+r['name']+'"]\nsha256 = "'+r['sha256']+'"\nrows = '+str(r['rows'])+'\ncolumns = '+str(r['columns'])+'\n\n' for r in records).encode())
    result=dict(files=records,manifest=dict(path=str(manifest),sha256=digest(manifest)),seeds=[42,*range(2000,2040)],primitive='Public one-based shuffle positions, uniform volatility draws, and standard-normal innovations. Native code constructs all means, covariances, samples, fits and portfolio results.')
    (REPORT/'mif_chapter23_random_input_preparation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(files=len(records),primitive_values=sum(r['rows']*r['columns'] for r in records),manifest=result['manifest']),indent=2))


if __name__=='__main__':
    main()
