"""Preserve the chapter's public random primitives for entirely native replay."""

import io
import json
from pathlib import Path

import numpy as np

from input_common import REPORT, ROOT, PYTHON, digest, save_exact, save_metadata, toml_value


def shuffled_folds(count,splits,seed):
    indices=np.arange(count)
    np.random.RandomState(seed).shuffle(indices)
    sizes=np.full(splits,count//splits,dtype=int)
    sizes[:count%splits]+=1
    start=0
    for size in sizes:
        stop=start+size
        yield indices[start:stop]
        start=stop


def main():
    destination = ROOT/'notebooks'/'julia'/'data'/'chapter14_random_inputs'
    destination.mkdir(parents=True, exist_ok=True)
    files = []

    def save(name, values, description, integer=False, columns=None):
        stream = io.StringIO(newline='')
        np.savetxt(stream, values, delimiter=',', header=','.join(columns or [f'x{i+1}' for i in range(values.shape[1])]), comments='', fmt='%d' if integer else '%.17g')
        path = destination/(name+'.csv')
        content = stream.getvalue().encode('utf-8')
        if path.exists():
            assert path.read_bytes() == content
        else:
            path.write_bytes(content)
        files.append(dict(path=str(path), name=path.name, sha256=digest(path), rows=values.shape[0], columns=values.shape[1], description=description))

    rng = np.random.default_rng(12001)
    controlled = np.column_stack([rng.normal(size=1200), rng.normal(size=1200), rng.normal(size=1200), rng.normal(size=1200), rng.uniform(-3,3,1200), rng.integers(0,5,1200), rng.normal(size=1200)])
    save('controlled_draws', controlled, 'Original default_rng(12001) innovations, uniform noise, and five-level noise in source draw order.', columns=['signal_one','signal_two','copy_one_noise','copy_two_noise','continuous_noise','five_level_noise','target_noise'])
    rng = np.random.default_rng(13000)
    duplicates = np.column_stack([rng.normal(size=1400), rng.normal(size=1400), *[rng.normal(size=1400) for _ in range(4)]])
    save('duplicate_draws', duplicates, 'Original default_rng(13000) shared signal, target innovation, and four independent noises.', columns=['latent_signal','target_noise','noise_one','noise_two','noise_three','noise_four'])
    for seed in range(12030,12042):
        rng = np.random.default_rng(seed)
        values = np.column_stack([rng.integers(0,2,400), rng.random(400), rng.integers(0,2,400), rng.integers(0,4,400), rng.integers(0,20,400), rng.normal(size=400), rng.normal(size=400)])
        save(f'bias_{seed}', values, f'Original default_rng({seed}) binary signal, target uniform, and five noises.', columns=['signal','target_uniform','binary_noise','four_level_noise','twenty_level_noise','normal_a','normal_b'])
    for seed in (14001,15001):
        rng = np.random.default_rng(seed)
        values = np.column_stack([rng.normal(size=1200) for _ in range(5)])
        save(f'cpi_{seed}', values, f'Original default_rng({seed}) direct signal, surrogate innovation, second signal, independent noise, and target innovation.', columns=['direct_signal','surrogate_noise','second_signal','independent_noise','target_noise'])

    plans = [(1201,1755,False,120),(1202,1404,True,120),(1204,1404,True,120),(1221,1200,False,160),(1222,960,True,160),(1224,960,True,160),(1231,1755,False,120),(1232,1404,True,120),(1241,1755,False,120)]
    for count in (1,2,4,8):
        plans += [(13010+count,1400,False,120),(13020+count,1050,True,120),(13040+count,1400,False,120),(13050+count,1050,True,120)]
    plans += [(seed,400,False,120) for seed in range(12030,12042)]
    plans += [(seed,900,False,80) for seed in (14001,15001)]
    for seed,count,weighted,trees in plans:
        tree_seeds = np.random.RandomState(seed).randint(0,2147483647,trees)
        split_seeds = np.array([np.random.RandomState(current).randint(0,2147483647) for current in tree_seeds])
        weights = []
        for current in tree_seeds:
            rng = np.random.RandomState(current)
            positions = rng.choice(count,count,replace=True,p=np.ones(count)/count) if weighted else rng.randint(0,count,count)
            weights.append(np.bincount(positions,minlength=count))
        values = np.vstack([tree_seeds,split_seeds,np.array(weights).T])
        save(f'forest_{seed}_{count}_{int(weighted)}', values, f'Public forest seed {seed}; row 1 estimator seeds, row 2 split seeds, remaining rows bootstrap multiplicities; explicit all-one sample weights={weighted}.', True)

    fold_specs = [(1203,1755,5,12),(1223,1200,5,6),(1233,1755,5,4)]
    fold_specs += [(13030+count,1400,4,count+4) for count in (1,2,4,8)]
    fold_specs += [(13060+count,1400,4,2) for count in (1,2,4,8)]
    for seed,count,splits,groups in fold_specs:
        membership = np.zeros(count,dtype=int)
        rows = []
        for fold,test in enumerate(shuffled_folds(count,splits,seed)):
            membership[test] = fold+1
            rng = np.random.default_rng(seed+fold)
            permutations = []
            for group in range(groups):
                positions = np.arange(len(test))
                rng.shuffle(positions)
                permutations.append(positions+1)
            rows.append(np.column_stack([np.full(len(test),fold+1),np.array(permutations).T]))
        save(f'folds_{seed}', membership[:,None], f'Original shuffled KFold({splits},random_state={seed}); one-based held-out fold per original row.', True, ['fold'])
        save(f'permutations_{seed}', np.vstack(rows), f'Original sequential default_rng({seed}+zero-based fold) row permutations; first column fold, other columns one-based row positions.', True)
    for seed in (14001,15001):
        membership = np.zeros(1200,dtype=int)
        rows = []
        for fold,test in enumerate(shuffled_folds(1200,4,seed)):
            membership[test] = fold+1
            rng = np.random.default_rng(seed+100+fold)
            rows.append(np.column_stack([np.full(len(test),fold+1),*[rng.standard_normal(len(test)) for _ in range(4)]]))
        save(f'folds_{seed}', membership[:,None], f'Original shuffled KFold(4,random_state={seed}); one-based held-out fold per original row.', True, ['fold'])
        save(f'gaussians_{seed}', np.vstack(rows), f'Original CPI default_rng({seed}+100+zero-based fold) conditional Gaussian innovations in feature-major draw order.')

    lines = ['title = "Chapter 14 public forest and experiment primitives"', '']
    for row in files:
        lines += ['[files.'+json.dumps(row['name'])+']','sha256 = '+json.dumps(row['sha256']),'rows = '+str(row['rows']),'columns = '+str(row['columns']),'description = '+json.dumps(row['description']),'']
    manifest = destination/'inputs.toml'
    content = ('\n'.join(lines)+'\n').encode('utf-8')
    save_metadata(manifest,content)
    result = dict(status='PREPARED_FIXED_PUBLIC_RANDOM_INPUTS', files=files, forest_plans=len(plans), manifest=dict(path=str(manifest),sha256=digest(manifest)), script_sha256=digest(Path(__file__)))
    (REPORT/'mif_chapter14_random_input_preparation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(files=len(files),forest_plans=len(plans),sha256=digest(manifest)),indent=2))


if __name__ == '__main__':
    main()
