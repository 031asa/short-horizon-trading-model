"""Compare W-second observations against the same W-second factor history."""
from pathlib import Path
from itertools import product
from datetime import datetime,timezone
import sys,json,re,hashlib, warnings
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.cache/python-packages'))
import numpy as np
import pandas as pd
from utils.ic_statistics import pair_matrix,STATUS

OUT=ROOT/'result/opening_execution'
RULES=dict(version='short-window-curves-20260923-v1',windows=[1,2,3,4,5],
    primary='observation W, history W, future 1..30 seconds from each own observation end',
    diagnostic='same observation end T+5 for every history W, same future labels; history amount only',
    early_horizons=[1,2,3],ic_threshold=.05,minimum_valid_days=30,minimum_coverage=.8,minimum_agreeing_day_share=.6,
    compression_margin=.02,compression='all early horizons strong in short and 5s; paired lower 95% interval of prior-aligned difference >= -0.02 at all three horizons; research evidence, no global multiple-testing claim',
    bootstrap='2000 circular moving-block resamples of 38 days, block 5 trading days, seed 20260923; pointwise intervals',
    minimum_pairs=20,phase='all_sample',future_seconds=list(range(1,31)),arrival_is_screening_gate=False,
    averages_across_horizons_used_for_selection=False)

def catalog(registry):
    eligible=registry.loc[registry.kind.eq('computed') & registry.expected_sign_signed.isin(['positive','negative'])]
    groups=[];excluded=[]
    for _,r in eligible.loc[eligible.history_seconds.eq(0)&~eligible.factor.str.contains('_lag')].iterrows():
        groups.append(dict(expression=r.factor,name_zh=r.name_zh,prior=r.expected_sign_signed,
            mode='snapshot',factors=[r.factor]*5,definition=r.definition,family_id=r.family_id))
    rolling=eligible.loc[eligible.history_seconds.between(1,5)&eligible.factor.str.contains(r'_h\d+s$')].copy()
    rolling['expression']=rolling.factor.str.replace(r'_h\d+s$','',regex=True)
    for expression,g in rolling.groupby('expression',sort=False):
        g=g.sort_values('history_seconds');assert g.history_seconds.tolist()==[1,2,3,4,5]
        r=g.iloc[0];groups.append(dict(expression=expression,name_zh=r.name_zh,prior=r.expected_sign_signed,
            mode='rolling',factors=g.factor.tolist(),definition=r.definition,family_id=r.family_id))
    for stem in ['A08_OFIRateChange','B09_FullShareDifference','B09_RecoveryDifference','B09_SuccessTimeDifference']:
        excluded.append(dict(expression=stem,reason='前后两段各 2／5 秒，合计 4／10 秒；不属于五组普通回看' if stem.startswith('A08') else '固定 10 秒恢复历史及成熟门槛，不能称为 1–5 秒因子'))
    return groups,excluded

def bootstrap(values,draws):
    finite=np.isfinite(values);sample=values[draws];count=np.isfinite(sample).sum(axis=1)
    means=np.divide(np.nansum(sample,axis=1),count,out=np.full(len(draws),np.nan),where=count>0)
    return np.nanquantile(means,[.025,.975]).tolist() if finite.sum()>=2 else [np.nan,np.nan]

def diagonal_common_mask(x,y):
    """x is [matched W, task, expression]; y is [matched W, task, horizon]."""
    return np.isfinite(x).all(axis=0)&np.isfinite(y).all(axis=(0,2))[:,None]

def run():
    from utils.short_window_analysis import analyze
    registry=pd.read_csv(OUT/'feature_registry.csv')
    groups,excluded=catalog(registry)
    return analyze(groups,excluded,registry,RULES,OUT,ROOT)

if __name__=='__main__':run()
