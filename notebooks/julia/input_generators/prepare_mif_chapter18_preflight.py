"""Prepare a bounded public-data SVC solver preflight for the original search."""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from input_common import REPORT, ROOT, PYTHON, digest, save_exact, toml_value
from prepare_mif_chapter16_inputs import public_data

FEATURES = ['Previous return','Distance from EWM 15','Distance from EWM 55','One-hour return','Past volatility','Log volume change']


def chapter_data():
    _,bars,_ = public_data()
    levels = np.log(bars.Close)
    returns = levels.diff()
    features = pd.DataFrame({'Previous return':returns.shift(1),'Distance from EWM 15':(levels-np.log(bars.Close.ewm(span=15,adjust=False).mean())).shift(1),'Distance from EWM 55':(levels-np.log(bars.Close.ewm(span=55,adjust=False).mean())).shift(1),'One-hour return':levels.diff(4).shift(1),'Past volatility':returns.shift(1).rolling(16).std(),'Log volume change':np.log1p(bars.Volume).diff().shift(1)})
    forward = (levels.shift(-8)-levels).rename('Forward return')
    direction = forward.gt(0).where(forward.notna()).astype('Int64').rename('Direction')
    ends = bars.index.to_series().shift(-8).rename('End time')
    frame = pd.concat([features,direction,forward,ends],axis=1).dropna()
    weights = (frame['Forward return'].abs()/frame['Forward return'].abs().median()).clip(.25,4.)
    return bars,frame,weights


def main():
    sys.path.insert(0,str(PYTHON.parent))
    module = importlib.import_module('RiskLabAI.backtest.validation')
    assert Path(module.__file__).resolve().is_relative_to(PYTHON)
    _,frame,weights = chapter_data()
    x,y = frame[FEATURES],frame.Direction.astype(int)
    train,test = next(module.PurgedKFold(n_splits=5,times=frame['End time'],embargo=.01).split(x,y))
    destination = REPORT/'chapter18_svc_preflight'
    destination.mkdir(parents=True,exist_ok=True)
    features = x.copy()
    features['target'],features['weight'] = y,weights
    features.to_csv(destination/'observations.csv',index=False)
    pd.DataFrame({'position':train+1}).to_csv(destination/'train.csv',index=False)
    pd.DataFrame({'position':test+1}).to_csv(destination/'test.csv',index=False)
    records = []
    for kernel,penalty in [('linear',1.),('linear',10.),('rbf',1.),('rbf',10.)]:
        name = kernel+'_'+str(int(penalty))
        model = Pipeline([('scale',StandardScaler()),('svc',SVC(kernel=kernel,C=penalty,gamma='scale'))]).fit(x.iloc[train],y.iloc[train],svc__sample_weight=weights.iloc[train].to_numpy())
        pd.DataFrame({'position':test+1,'prediction':model.predict(x.iloc[test]),'score':model.decision_function(x.iloc[test])}).to_csv(destination/(name+'_reference.csv'),index=False)
        records.append(dict(name=name,kernel=kernel,C=penalty,iterations=int(model['svc'].n_iter_[0]),support_vectors=len(model['svc'].support_),fit_status=int(model['svc'].fit_status_)))
        print(json.dumps(records[-1]),flush=True)
    report = dict(status='PREPARED',cases=records,files=[dict(path=str(path),sha256=digest(path)) for path in sorted(destination.glob('*.csv'))],public_helper_sha256=digest(Path(__file__)))
    (REPORT/'mif_chapter18_svc_preflight_preparation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
