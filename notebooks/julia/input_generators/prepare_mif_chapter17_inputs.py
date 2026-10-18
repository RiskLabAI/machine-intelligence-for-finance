"""Preserve the public CPCV bagging row draws and deterministic estimator seeds."""

import ast
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from input_common import REPORT, ROOT, PYTHON, digest, save_exact, toml_value
from prepare_mif_chapter16_inputs import FEATURES, public_data

SUBJECT = '17_combinatorial_purged_cross_validation'


class DrawRecorder(ClassifierMixin,BaseEstimator):
    """Record the public bagging fitting weights without estimating a model."""

    def __init__(self,random_state=15):
        self.random_state = random_state

    def fit(self,data,labels,sample_weight=None):
        self.classes_ = np.unique(labels)
        self.weights_ = np.asarray(sample_weight).copy()
        return self

    def predict(self,data):
        raise RuntimeError('The draw recorder cannot produce model predictions.')


def scaled_estimator_class():
    path = ROOT/'notebooks'/SUBJECT/'17_03_public_path_comparison.ipynb'
    source = ''.join(json.loads(path.read_text(encoding='utf-8'))['cells'][2]['source'])
    nodes = [node for node in ast.parse(source).body if isinstance(node,ast.ClassDef) and node.name=='ScaledLogisticRegression']
    assert len(nodes)==1
    namespace = dict(BaseEstimator=BaseEstimator,ClassifierMixin=ClassifierMixin,StandardScaler=StandardScaler,LogisticRegression=LogisticRegression)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['ScaledLogisticRegression']


def save_exact(path,content):
    if path.exists():
        assert path.read_bytes()==content
    else:
        path.write_bytes(content)


def main():
    sys.path.insert(0,str(PYTHON.parent))
    module = importlib.import_module('RiskLabAI.backtest.validation.combinatorial_purged')
    assert Path(module.__file__).resolve().is_relative_to(PYTHON)
    _,_,frame = public_data()
    x,y,times = frame[FEATURES],frame.Direction.astype(int),frame['End time']
    validator = module.CombinatorialPurged(n_splits=6,n_test_groups=2,times=times,embargo=.01)
    destination = ROOT/'notebooks'/'julia'/'data'/'chapter17_random_inputs'
    destination.mkdir(parents=True,exist_ok=True)
    records,seeds = [],[]
    for split,(train,_) in enumerate(validator.split(x,y),start=1):
        bag = BaggingClassifier(estimator=DrawRecorder(),n_estimators=12,random_state=15,n_jobs=1)
        bag.fit(x.iloc[train],y.iloc[train],sample_weight=np.ones(len(train)))
        columns = {}
        for member,(fitted,draws,features) in enumerate(zip(bag.estimators_,bag.estimators_samples_,bag.estimators_features_),start=1):
            np.testing.assert_array_equal(features,np.arange(6))
            np.testing.assert_array_equal(fitted.weights_,np.bincount(draws,minlength=len(train)))
            assert len(draws)==len(train)
            columns['member_'+str(member)] = draws+1
            seeds.append(dict(split=split,member=member,seed=int(fitted.random_state)))
        path = destination/('split_'+str(split)+'.csv')
        save_exact(path,pd.DataFrame(columns).to_csv(index=False).encode('utf-8'))
        records.append(dict(path=str(path),name=path.name,sha256=digest(path),rows=len(train),columns=12))
    path = destination/'member_seeds.csv'
    save_exact(path,pd.DataFrame(seeds).to_csv(index=False).encode('utf-8'))
    records.append(dict(path=str(path),name=path.name,sha256=digest(path),rows=len(seeds),columns=3))
    manifest = destination/'inputs.toml'
    content = ''.join('[files."'+row['name']+'"]\nsha256 = "'+row['sha256']+'"\nrows = '+str(row['rows'])+'\ncolumns = '+str(row['columns'])+'\n\n' for row in records)
    save_exact(manifest,content.encode('utf-8'))
    record = dict(status='PREPARED_PUBLIC_RANDOM_PRIMITIVES',files=records,manifest=dict(path=str(manifest),sha256=digest(manifest)),splits=15,members_per_split=12,public_source=dict(path=str(Path(module.__file__)),sha256=digest(Path(module.__file__))),supplier_source=dict(path=str(Path(importlib.import_module('sklearn.ensemble._bagging').__file__)),sha256=digest(Path(importlib.import_module('sklearn.ensemble._bagging').__file__))),primitive='One-based row draws and estimator seeds from public seed-15 BaggingClassifier with full feature selection and unit training weights. No fitted predictions or coefficients are supplied to Julia.')
    (REPORT/'mif_chapter17_random_input_preparation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(files=len(records),manifest_sha256=digest(manifest),bagged_members=len(seeds),sample_draws=sum(row['rows']*row['columns'] for row in records[:-1])),indent=2))


if __name__=='__main__':
    main()
