"""Filter existing IC summaries by a declared positive prior and positive raw IC."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.cache/python-packages'))
import pandas as pd

def filter_positive(output):
    output=Path(output);registry=pd.read_csv(output/'feature_registry.csv');s=pd.read_parquet(output/'summary_ic.parquet')
    base=s.loc[s.phase.eq('all_sample')&s.observation_seconds.eq(1)&s.anchor.eq('decision')&s.label_type.eq('cumulative')&s.pair_set.eq('own')]
    candidates=registry.loc[registry.evaluate]
    selected=base.loc[base.target.eq('signed')&base.factor.isin(candidates.loc[candidates.expected_sign_signed.eq('positive'),'factor'])]
    green=selected.loc[selected.mean_rank_ic.gt(0)]
    def ranges(values):
        values=sorted(values);parts=[]
        if not values:return ''
        start=end=values[0]
        for value in values[1:]:
            if value==end+1:end=value
            else:parts.append(str(start) if start==end else f'{start}–{end}');start=end=value
        parts.append(str(start) if start==end else f'{start}–{end}')
        return '、'.join(parts)
    rows=[]
    for r in candidates.itertuples():
        g=green.loc[green.factor.eq(r.factor)]
        if g.empty:continue
        full=selected.loc[selected.factor.eq(r.factor)].set_index('horizon_seconds')
        rows.append(dict(family_id=r.family_id,factor=r.factor,name_zh=r.name_zh,history_seconds=r.history_seconds,
            positive_horizons=ranges(g.horizon_seconds),positive_horizon_count=len(g),all_30_positive=len(g)==30,
            observation_seconds=1,anchor='decision',label_type='cumulative',target='signed',pair_set='own',method='Spearman',
            **{f'IC_{u}s':full.loc[u,'mean_rank_ic'] for u in range(1,31)}))
    result=pd.DataFrame(rows);result.to_csv(output/'逻辑正向且绿色IC_筛选清单.csv',index=False,encoding='utf-8-sig')
    lines=['# 逻辑正向且绿色 IC 筛选','',
        '固定查看口径：全部 38 日；观察 1 秒；观察结束基准；有方向累计 LastPrice 价变；逐期限有效样本；Spearman Rank IC。',
        '筛选条件：原登记逻辑预期 positive，且平均日 Rank IC > 0。未重算 IC，未改写逻辑预期；别名不重复计入。',
        f'共 {len(result)} 个因子至少一期为正，其中 {int(result.all_30_positive.sum())} 个全部 30 期为正。一个绿色期限不代表整条曲线为正；本表仅按所选条件展示。','',
        '[打开筛选网页](全部因子IC与衰减.html#positive) · [完整筛选清单 CSV](逻辑正向且绿色IC_筛选清单.csv)','',
        '| 因子 | 定义 | 历史窗口 | 绿色期限（秒） | 绿色期数 |','|---|---|---|---|---:|']
    for r in result.itertuples():lines.append(f'| {r.factor} | {r.name_zh} | {r.history_seconds or "快照/两段"} | {r.positive_horizons} | {r.positive_horizon_count}/30 |')
    lines+=['','CSV 保留全部 30 期限的原始 IC，包括筛入因子的负 IC；网页可改为全部期限或指定期限筛选。绿色仅指正相关，不表示显著性或收益保证。','']
    (output/'逻辑正向且绿色IC_筛选说明.md').write_text('\n'.join(lines),encoding='utf-8')
    cases=[]
    for target in ('signed','absolute'):
        names=candidates.loc[candidates['expected_sign_'+target].eq('positive'),'factor']
        grid=base.loc[base.target.eq(target)&base.factor.isin(names)]
        for method in ('ic','rank_ic'):
            for scope in ('any','all','1','5','30'):
                g=grid.loc[grid['mean_'+method].gt(0)]
                if scope=='any':matched=set(g.factor)
                elif scope=='all':matched=set(g.groupby('factor').size().loc[lambda x:x.eq(30)].index)
                else:matched=set(g.loc[g.horizon_seconds.eq(int(scope)),'factor'])
                cases.append(dict(target=target,method=method,scope=scope,factors=candidates.loc[candidates.factor.isin(matched),'factor'].tolist()))
    (output/'positive_filter_expected.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'any_positive':len(result),'all_30_positive':int(result.all_30_positive.sum()),'all_positive_factors':result.loc[result.all_30_positive,'factor'].tolist()},ensure_ascii=False))
    return result

if __name__=='__main__':filter_positive(ROOT/'result/opening_execution')
