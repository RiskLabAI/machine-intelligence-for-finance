"""Preserve the public temporal-validation shuffle without storing fitted outputs."""

import json

import numpy as np
import pandas as pd

from input_common import REPORT, ROOT, PYTHON, digest, save_exact, toml_value

SUBJECT = '16_temporal_cross_validation'
FEATURES = ['Previous return','One-hour return','Past volatility','Log volume change','Log quote-volume change','Log trade-count change']


def public_data():
    source = ROOT/'data'/'public'/'binance'/'BTCUSDT-1m-2021-01.zip'
    assert digest(source)=='9364a063efa0604d5b9d2536fb68f922da5afa3d8f94a26a527471a1e2f87801'
    names = ['Open time','Open','High','Low','Close','Volume','Close time','Quote volume','Trades','Buyer base volume','Buyer quote volume','Unused']
    raw = pd.read_csv(source,compression='zip',header=None,names=names)
    raw['time'] = pd.to_datetime(raw['Open time'],unit='ms',utc=True).dt.tz_convert(None)
    raw = raw.sort_values('time').drop_duplicates('time').set_index('time')
    bars = raw.resample('15min').agg({'Close':'last','Volume':'sum','Quote volume':'sum','Trades':'sum'}).dropna()
    levels = np.log(bars.Close)
    returns = levels.diff()
    features = pd.DataFrame({'Previous return':returns.shift(1),'One-hour return':levels.diff(4).shift(1),'Past volatility':returns.shift(1).rolling(16).std(),'Log volume change':np.log1p(bars.Volume).diff().shift(1),'Log quote-volume change':np.log1p(bars['Quote volume']).diff().shift(1),'Log trade-count change':np.log1p(bars.Trades).diff().shift(1)})
    forward = (levels.shift(-8)-levels).rename('Forward return')
    direction = forward.gt(0).where(forward.notna()).astype('Int64').rename('Direction')
    ends = bars.index.to_series().shift(-8).rename('End time')
    frame = pd.concat([features,direction,forward,ends],axis=1).dropna()
    return raw,bars,frame


def main():
    raw,bars,frame = public_data()
    assert len(raw)==44640 and len(bars)==2976 and len(frame)==2951
    permutation = np.arange(len(frame))
    np.random.default_rng(14).shuffle(permutation)
    destination = ROOT/'notebooks'/'julia'/'data'/'chapter16_random_inputs'
    destination.mkdir(parents=True,exist_ok=True)
    path = destination/'random_fold_permutation.csv'
    content = pd.DataFrame({'position':permutation+1}).to_csv(index=False).encode('utf-8')
    if path.exists():
        assert path.read_bytes()==content
    else:
        path.write_bytes(content)
    manifest = destination/'inputs.toml'
    manifest_content = ('[random_fold_permutation]\nfile = "random_fold_permutation.csv"\nsha256 = "'+digest(path)+'"\nseed = 14\nobservations = 2951\n').encode('utf-8')
    if manifest.exists():
        assert manifest.read_bytes()==manifest_content
    else:
        manifest.write_bytes(manifest_content)
    record = dict(status='PREPARED_PUBLIC_RANDOM_PRIMITIVE',observations=len(frame),files=[dict(path=str(path),sha256=digest(path))],manifest=dict(path=str(manifest),sha256=digest(manifest)),primitive='The one-based NumPy default_rng(14) permutation used by the released KFold. No model outputs are inputs.')
    (REPORT/'mif_chapter16_random_input_preparation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
