"""Preserve original Sharpe-inference innovations and circular bootstrap positions."""

import json
from pathlib import Path

import numpy as np

from input_common import REPORT, ROOT, PYTHON, digest, save_exact, toml_value
from prepare_mif_chapter19_inputs import public_data

SUBJECT = '20_sharpe_multiple_testing'


def circular_block_indices(count,block,generator):
    starts=generator.integers(0,count,size=int(np.ceil(count/block)))
    return ((starts[:,None]+np.arange(block)[None,:])%count).reshape(-1)[:count]


def main():
    _,performance,_ = public_data()
    destination = ROOT/'notebooks'/'julia'/'data'/'chapter20_random_inputs'
    destination.mkdir(parents=True,exist_ok=True)
    records = []

    def save(name,values,seed_description,primitive):
        values = np.asarray(values,dtype='<f8')
        path = destination/(name+'.bin')
        save_exact(path,values.tobytes(order='F'))
        np.testing.assert_array_equal(np.frombuffer(path.read_bytes(),dtype='<f8').reshape(values.shape,order='F'),values)
        records.append(dict(path=str(path),name=path.name,sha256=digest(path),shape=list(values.shape),values=int(values.size),bytes=path.stat().st_size,dtype='float64',byte_order='little',array_order='column_major',seed=seed_description,primitive=primitive))

    rng = np.random.default_rng(20260928)
    for count in [1,5,20,100]:
        save('maximum_'+str(count),rng.standard_normal((5000,count)),'20260928, sequential trial-count order 1, 5, 20, 100','Standard normal draws before multiplication by 0.5 and row maxima.')
    save('preliminary_maximum',np.random.default_rng(20260928).standard_normal((1000,100)),'20260928, fresh generator','Standard normal draws before multiplication by 0.5 and row maxima.')
    rng = np.random.default_rng(3)
    save('preliminary_planted',np.column_stack([rng.standard_normal(120) for _ in range(20)]),'3, sequential trial order','Standard normal draws before the original scale and three mean shifts.')
    save('ar1_preliminary',np.column_stack([np.random.default_rng(seed).standard_normal(500) for seed in range(300)]),'Column j uses seed j-1','Standard normal innovations before AR(1) recursion.')
    save('ar1_assertion',np.random.default_rng(999).standard_normal((500,1)),'999','Standard normal innovations for the original influence-function assertion.')
    for persistence in [0,.2,.4,.6]:
        code = int(1000*persistence)
        save('ar1_'+str(code),np.column_stack([np.random.default_rng(100000+code+repetition).standard_normal(500) for repetition in range(200)]),'100000+'+str(code)+'+zero-based repetition','Standard normal innovations before AR(1) recursion.')
    for persistence in [0,.4]:
        code = int(1000*persistence)
        save('paired_'+str(code),np.stack([np.random.default_rng(300000+code+repetition).standard_t(7,(240,2)) for repetition in range(60)],axis=2),'300000+'+str(code)+'+zero-based repetition','Student-t innovations with seven degrees of freedom before scale and bivariate AR(1) recursion.')
    for repetition in range(60):
        rng = np.random.default_rng(400000+repetition)
        count,block = 240,int(np.ceil(240**(1/3)))
        rows = np.column_stack([circular_block_indices(count,block,rng)+1 for _ in range(99)])
        save('paired_bootstrap_'+str(repetition+1),rows,str(400000+repetition),'One-based circular block positions reused for both original persistence conditions.')
    for pair in range(1,4):
        rng = np.random.default_rng(1800+pair)
        count = len(performance)
        block = int(np.ceil(count**(1/3)))
        rows = np.column_stack([circular_block_indices(count,block,rng)+1 for _ in range(399)])
        save('public_bootstrap_'+str(pair),rows,str(1800+pair),'One-based public paired-comparison circular block positions.')
    for count in [120,500]:
        for trials in [5,20,50]:
            draws = np.stack([np.random.default_rng(500000+10000*count+100*trials+repetition).standard_normal((count,trials)) for repetition in range(200)],axis=2)
            save('family_'+str(count)+'_'+str(trials),draws,'500000+10000*'+str(count)+'+100*'+str(trials)+'+zero-based repetition','Standard normal innovations before scale and the first three trial mean shifts.')
    manifest = destination/'inputs.toml'
    contents = 'format = "Little-endian Float64 in column-major order; integer row positions are exactly represented."\n\n'
    for row in records:
        contents += '[files."'+row['name']+'"]\nsha256 = "'+row['sha256']+'"\nshape = '+json.dumps(row['shape'])+'\nvalues = '+str(row['values'])+'\n\n'
    save_exact(manifest,contents.encode('utf-8'))
    report = dict(status='PREPARED_PUBLIC_RANDOM_PRIMITIVES',files=records,manifest=dict(path=str(manifest),sha256=digest(manifest)),script_sha256=digest(Path(__file__)),primitive='Unmodified original random innovations and circular bootstrap positions only. Every Sharpe estimate, p-value, rejection, maximum and summary remains a native calculation.')
    (REPORT/'mif_chapter20_random_input_preparation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(files=len(records),values=sum(row['values'] for row in records),bytes=sum(row['bytes'] for row in records),manifest_sha256=digest(manifest)),indent=2))


if __name__=='__main__':
    main()
