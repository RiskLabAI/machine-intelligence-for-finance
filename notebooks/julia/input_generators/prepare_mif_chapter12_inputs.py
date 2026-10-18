"""Preserve the public chapter's seeded random primitives for native replay."""

import io
import json
from pathlib import Path

import numpy as np

from input_common import REPORT, ROOT, PYTHON, digest, save_exact, save_metadata, toml_value


def main():
    destination=ROOT/'notebooks'/'julia'/'data'/'chapter12_random_inputs'
    destination.mkdir(parents=True,exist_ok=True)
    records=[]

    def save(name,values,columns,description,integer=False):
        stream=io.StringIO(newline='')
        np.savetxt(stream,values,delimiter=',',header=','.join(columns),comments='',fmt='%d' if integer else '%.17g')
        content=stream.getvalue().encode('utf-8')
        path=destination/(name+'.csv')
        if path.exists():
            assert path.read_bytes()==content
        else:
            path.write_bytes(content)
        records.append(dict(path=str(path),name=path.name,sha256=digest(path),shape=list(values.shape),description=description))

    for name,size,count,seed in [('psy_public_normals',384,499,2718),('psy_controlled_normals',300,399,2718),('psy_variance_normals',130,599,31415)]:
        draws=np.vstack([np.random.default_rng(seed+simulation).standard_normal(size) for simulation in range(count)])
        save(name,draws,[f'z{i+1}' for i in range(size)],f'Public PSY calibration: {count} paths of length {size}; NumPy default_rng({seed} + zero-based simulation).')
    rng=np.random.default_rng(20260927)
    signs=np.vstack([rng.integers(0,2,383)*2-1 for _ in range(399)])
    save('public_bootstrap_signs',signs,[f'sign{i+1}' for i in range(383)],'Public wild bootstrap: 399 sign vectors from NumPy default_rng(20260927).',True)
    variance_normals=np.vstack([np.random.default_rng(8000+repetition).standard_normal(130) for repetition in range(32)])
    save('variance_normals',variance_normals,[f'z{i+1}' for i in range(130)],'Public no-bubble paths: NumPy default_rng(8000 + repetition), shared across the two volatility scenarios.')
    sign_rows=[]
    for repetition in range(32):
        rng=np.random.default_rng(9000+repetition)
        for bootstrap in range(79):
            sign_rows.append(np.r_[repetition,bootstrap,rng.integers(0,2,129)*2-1])
    save('variance_bootstrap_signs',np.array(sign_rows),['repetition','bootstrap']+[f'sign{i+1}' for i in range(129)],'Public volatility-adjusted calibration: 79 sign vectors per repetition from NumPy default_rng(9000 + repetition).',True)
    controlled=np.random.default_rng(20260927).standard_normal(300)
    save('controlled_shocks',controlled[:,None],['innovation'],'Public two-episode path: 300 standard-normal innovations from NumPy default_rng(20260927).')
    pelt=np.vstack([np.random.default_rng(12000+repetition).standard_normal(240) for repetition in range(24)])
    save('pelt_normals',pelt,[f'z{i+1}' for i in range(240)],'Public PELT experiments: NumPy default_rng(12000 + repetition); segment means and scales are applied in Julia.')
    lines=['title = "Chapter 12 public seeded random primitives"','']
    for row in records:
        lines+=['[files.'+json.dumps(row['name'])+']','sha256 = '+json.dumps(row['sha256']),'rows = '+str(row['shape'][0]),'columns = '+str(row['shape'][1]),'description = '+json.dumps(row['description']),'']
    manifest=destination/'inputs.toml'
    content=('\n'.join(lines)+'\n').encode('utf-8')
    save_metadata(manifest,content)
    result=dict(status='PREPARED_FIXED_PUBLIC_RANDOM_INPUTS',files=records,manifest=dict(path=str(manifest),sha256=digest(manifest)),script_sha256=digest(Path(__file__)),numpy_version=np.__version__)
    (REPORT/'mif_chapter12_random_input_preparation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(files=len(records),normal_innovations=499*384+399*300+599*130+32*130+300+24*240,signs=399*383+32*79*129),indent=2))


if __name__=='__main__':
    main()
