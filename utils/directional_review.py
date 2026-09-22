"""Mechanism-led research overlay. No labels, correlations or fitted parameters enter it.

The PDF catalog remains the original specification. This overlay preserves that
catalog and adds explicitly exploratory hypotheses and causal interaction factors.
"""
import json
import re
import numpy as np
import pandas as pd

REVIEW_VERSION = 'directional-mechanism-review-20260922-v1'
REVIEW_STATUS = '激进研究假设；已看过本批期货 IC；新增定义在重测前登记；不代表统计确定'
P, N, U = 'positive', 'negative', 'uncertain'


def _key(name):
    return re.split(r'_h\d+s|_lag\d+s', name)[0]


def _decisions():
    """One primary hypothesis per expression, chosen without consulting results."""
    rules = {}
    def add(names, sign, mechanism, condition, alternative):
        for name in names.split():
            assert name not in rules
            rules[name] = (sign, mechanism, condition, alternative)
    add('F05_LastChange', N, '末价成交侧跳动的短时回摆：上一跳越向上，下一段越可能回吐。',
        '单次 LastPrice 跳动包含买卖价之间的成交切换，信息性冲击较弱。', '开盘信息性跳价可继续同向；期限较长时回摆可能被趋势覆盖。')
    add('F05_Momentum F06_PathEfficiency F08_SignedVolume C02_Asymmetry C05_CountImbalance C06_SignedRun C08_MomentumSpeedChange M04_DeviationSpeed M05_MeanSpeed M06_MeanGap M11_TrendSlope R01_OpeningMove',
        P, '以信息消化和同向订单持续为主假设：越强的净上涨／买向历史对应越高的后续有方向价变。',
        '开盘订单拆分或价格发现尚未结束；窗口净方向可代表持续压力。', '订单耗尽、库存回补和末价回摆可能令 IC 反号。')
    add('B01_BidMean B03_BidSlope', P, '买方承接增强：买侧挂量水平或增厚趋势越强，上涨压力越大。',
        '卖侧供给相近；挂单具有持续性，补单反映真实承接。', '更深买队列也可能只是被动承接下跌，撤单会破坏推断。')
    add('B01_AskMean B03_AskSlope', N, '卖方供给增强：卖侧挂量水平或增厚趋势越强，后续价变越低。',
        '买侧需求相近；挂单具有持续性，补单反映真实供给。', '卖队列可在强势上涨中补充；挂量水平也受整体活跃度影响。')
    add('B06_BidUpRate B06_AskUpRate', P, '报价上移的延续假设：买方追价或卖方抬价对应后续向上价格发现。',
        '报价上移承载方向信息，不只是价差的暂时调整。', '买价上移可收窄价差，卖价上移可扩大价差；都不保证末价继续上涨。')
    add('B06_BidDownRate B06_AskDownRate', N, '报价下移的延续假设：买方退价或卖方降价对应后续向下价格发现。',
        '报价下移承载方向信息，不只是价差的暂时调整。', '报价可迅速恢复；价差调整与末价回摆可抵消延续。')
    add('B06_UpDifference', P, '激进采用买方追价主导：买价上移更频繁、卖价上移较少，反映买方主动性。',
        '上移频率差由买方主动追价主导，价差收窄后卖队列有被消耗的趋势。', '差值也可能来自卖方抬价减少，单凭该差不能识别全部机制。')
    add('B06_DownDifference', N, '激进采用卖方降价主导：卖价下移更频繁、买价下移较少，反映卖方主动性。',
        '下移频率差由卖方主动降价主导，价差收窄后买队列有被消耗的趋势。', '差值也可能来自买方退价减少，可能意味着买侧承接。')
    add('B09_BidRecovery B09_BidFullShare B09_AskSuccessTime', P,
        '买侧恢复更充分，或卖侧恢复更慢，分别偏向买方支撑增强／卖方供给减弱。',
        '只使用满足原门槛的同价成熟事件；另一侧状态相近。', '恢复可能是下跌中的被动补单；本批事件支持不足时仍不能测 IC。')
    add('B09_AskRecovery B09_AskFullShare B09_BidSuccessTime', N,
        '卖侧恢复更充分，或买侧恢复更慢，分别偏向卖方供给增强／买方支撑减弱。',
        '只使用满足原门槛的同价成熟事件；另一侧状态相近。', '恢复可能是上涨中的被动补单；本批事件支持不足时仍不能测 IC。')
    add('C02_UpSquares C04_Runup C05_UpCount', P, '上行分量的延续假设：上涨活动／上行路径越强，后续向上变化越强。',
        '下行分量和总活动规模相近，上行分量包含持续方向信息。', '这些是非负单侧量，也会随总波动一起增大；冲高回落可反号。')
    add('C02_DownSquares C04_Drawdown C05_DownCount', N, '下行分量的延续假设：下跌活动／下行路径越强，后续有方向价变越低。',
        '上行分量和总活动规模相近，下行分量包含持续方向信息。', '这些是非负单侧量，也会随总波动一起增大；超跌反弹可反号。')
    add('C03_Location R02_DistanceHigh R02_DistanceLow R02_OpeningLocation R03_PriorHighDistance R03_PriorLowDistance R03_NewHigh',
        P, '区间位置与突破采用延续假设：价格越靠上、越接近／突破高点，买向价格发现越强。',
        '极值附近持续成交能推动区间扩展，窗口内不是单纯来回震荡。', '边界阻力、突破失败和末价成交侧效应可导致反转。')
    add('R03_NewLow', N, '跌破此前低点采用向下延续假设。', '新低代表持续抛压，而非瞬时扫单后恢复。', '假突破与流动性补充可能迅速反弹。')
    add('D09_SignedVolumeCorr', P, '量价同向持续假设：成交更活跃的区间更倾向上涨时，买向压力可能持续。',
        '历史相关达到原支持门槛，且量价机制在随后的短期内延续。', '相关是局部状态估计；正相关也可由少数大价变驱动。')
    add('M01_MADeviation M02_OpeningMADeviation M03_StandardDeviation M09_MaxPositive M09_MinNegative M10_VWDeviation M10_OpeningVWDeviation M11_TrendResidual',
        N, '偏离类选择价格回归主假设：高于历史中心／趋势线越多，后续回吐越强；低于中心则倾向回补。',
        '历史中心仍有参考价值，偏离主要来自暂时冲击，持续新信息较弱。', '强趋势时均线滞后，偏离可持续扩大；不能把均线当真实价值。')
    add('M07_AboveShare M07_MeanState M08_CrossUpRate', P,
        '持续状态与完整上穿选择趋势延续假设：上方停留或向上切换代表方向已经建立。',
        '持续性信息占主导；这与偏离幅度的回归假设是两个待比较机制。', '反复穿越和趋势末端会产生反转；无条件 IC 可能冲突。')
    add('M07_BelowShare M08_CrossDownRate', N,
        '持续下方状态与完整下穿选择向下延续假设。',
        '持续性信息占主导；这与偏离幅度的回归假设是两个待比较机制。', '反复穿越和趋势末端会产生反转；无条件 IC 可能冲突。')
    return rules


def _remaining_reason(row):
    key = _key(row['factor'])
    if key.startswith('B09_'):
        return ('报价退出未区分上移／下移，退出比例本身没有单一方向。',
                '未来可按退出方向及事件时的队列方向分组；本轮不放宽成熟门槛。')
    if key in ('B02_CVDifference', 'B07_AgeDifference', 'B08_RVDifference'):
        return ('买减卖差具有侧别，但波动或驻留既可代表稳定承接，也可代表消耗与退价。',
                '需补充两侧变化方向或补单／消耗事件条件，单凭大小仍不能指定可靠主方向。')
    if key == 'D09_LagOFICorr':
        return ('历史传导相关的正负描述 OFI 是否有效，当前 OFI 的方向仍未进入该因子。',
                '可与当时的有方向 OFI 联用；当前估计支持有限，暂保留原定义。')
    return ('原表达描述强度、波动、活动或年龄；上下方向翻转时常保持相同，不能单独识别上涨还是下跌。',
            '可与当时已知的 QI、OFI 或趋势方向构造交互；新增项有明确父因子，原值继续保留。')


def variants(histories=(5,10)):
    """All constants fixed here; no optimization by horizon, day or observed IC."""
    out = []
    def add(name, parent, inputs, operation, title, formula, reason, condition,
            unit='ratio', prior=P):
        out.append(dict(factor=name, parent=parent, inputs=inputs, operation=operation,
                        name_zh=title, definition=formula, logical_reason=reason,
                        hypothesis_condition=condition, unit=unit, expected_sign_signed=prior))
    add('A01_QISpread', 'A01_Spread', ['F01_QI','A01_Spread'], 'product',
        '方向化价差压力', 'QI * SpreadTicks', '给价差加入当前队列方向；较宽价差放大已有方向压力的权重。',
        '队列失衡有预测信息，较宽价差对应更强冲击敏感性。', 'tick')
    add('A05_SignedStateAge', 'A05_StateAge', ['A05_QIState','A05_StateAge'], 'signed_log',
        '带方向的失衡持续年龄', 'QIState * log1p(StateAge / 1 second)', '将无方向的年龄与买／卖失衡状态结合，采用持续压力假设。',
        '年龄起点精确可知；持续状态尚未进入耗尽阶段。')
    for h in histories:
        s=f'_h{h}s'
        add('A02_QIThinDepth'+s,'A02_DepthLogRatio'+s,['F01_QI','A02_DepthLogRatio'+s],'thin',
            '相对薄深度加权队列方向','QI / (1 + exp(DepthLogRatio))','深度相对历史越薄，已有队列方向获得越大权重。',
            '当前失衡能代表压力，薄深度主要放大其冲击。')
        add('A03_QIStability'+s,'A03_QIStd'+s,['A03_QIMean'+s,'A03_QIStd'+s],'stability',
            '稳定性加权失衡方向','QIMean / (1 + QIStd)','在相同平均失衡下，削弱剧烈反复的队列状态。',
            '较稳定失衡比瞬时挂撤单更能代表持续压力。')
        add('A07_OFIActivityAligned'+s,'A07_OFIActivity'+s,['A07_OFIDirection'+s,'A07_OFIActivity'+s],'signed_log',
            '带 OFI 方向的活动强度','OFIDirection * log1p(OFIActivity * 1 second)','给无方向订单流活动加入该窗口净流向。',
            '净订单流向在活动增强时仍能延续。')
        add('A08_OFIPulseAligned'+s,'A08_OFIPulse'+s,['A07_OFIDirection'+s,'A08_OFIPulse'+s],'product',
            '净流向加权脉冲集中度','OFIDirection * OFIPulse','把最大绝对脉冲占比按全窗口净流向加权；不冒充最大事件自身的方向。',
            '集中脉冲与窗口净方向对应同一持续机制。')
        add('B08_QuoteRVAligned'+s,'B08_RVDifference'+s,
            [f'B06_{side}{d}Rate'+s for side in ('Bid','Ask') for d in ('Up','Down')]+['B08_BidRV'+s,'B08_AskRV'+s],'quote_rv',
            '报价方向加权双侧波动','((BidUp+AskUp-BidDown-AskDown)/sum(FourRates)) * (BidRV+AskRV)/2; sum=0 -> 0',
            '双侧波动补入报价净上移／下移方向，采用报价价格发现延续假设。',
            '报价净方向含持续信息；中性报价活动记零。','tick/sqrt(second)')
        add('C01_QIRV'+s,'C01_PriceRV'+s,['F01_QI','C01_PriceRV'+s],'product',
            '队列方向加权末价波动','QI * PriceRV','波动提供强度，当前队列提供方向，测试高波动是否放大队列压力。',
            '高波动主要增强压力传导，而非增加反转。','tick/sqrt(second)')
        add('D01_VolumeRateAligned'+s,'D01_VolumeRate'+s,['F08_SignedVolume'+s,'D01_VolumeRate'+s],'signed_log',
            '末价成交方向加权活跃度','SignedVolume * log1p(VolumeRate / (1 volume/second))','用末价方向成交代理给总活动加入方向，并压缩极端活动。',
            '末价方向成交代理有信息且压力持续；该代理不是已识别的真实主动买卖。')
        add('D05_QIVolumeDepth'+s,'D05_VolumeDepthRate'+s,['A03_QIMean'+s,'D05_VolumeDepthRate'+s],'signed_log',
            '队列方向加权成交深度速率','QIMean * log1p(VolumeDepthRate * 1 second)','队列方向结合单位深度成交活动，测试活跃消耗下的方向传导。',
            '历史队列偏向与成交消耗方向一致。')
        for stem,op,title,prior in [('TrendConfirmed','aligned','订单流同向偏离',P),('CounterflowDeviation','opposed','订单流反向偏离',N)]:
            add('M01_'+stem+s,'M01_MADeviation'+s,['M01_MADeviation'+s,'A07_OFIDirection'+s],op,title,
                'MADeviation * I(MADeviation * OFIDirection '+('> 0)' if op=='aligned' else '< 0)'),
                '同向订单流确认偏离继续扩大。' if op=='aligned' else '相反订单流确认偏离倾向回归；保留原偏离符号，预期负相关。',
                '先满足两个父因子的原有效性要求；不满足方向条件的有效任务记 0。','tick',prior)
        add('M07_SignedMeanAge'+s+'_g5s','M07_MeanStateAge'+s+'_g5s',
            ['M07_MeanState'+s+'_g5s','M07_MeanStateAge'+s+'_g5s'],'signed_log',
            '带方向的均线状态年龄','MeanState * log1p(MeanStateAge / 1 second)','无方向持续时间补入均线上／下状态，选择趋势持续假设。',
            '年龄必须精确可知；不使用左删失下界。')
        add('R04_QIVolumeState'+s,'R04_VolumeRatePercentile'+s,['F01_QI','R04_VolumeRatePercentile'+s],'product',
            '成交历史分位加权队列方向','QI * VolumeRatePercentile','自身开盘成交活动越高，当前队列方向权重越大。',
            '历史分位只用当时已完成数据；活跃时压力传导较强。')
    for lag in (2,5):
        s=f'_lag{lag}s'
        add('R05_QIVolumeAcceleration'+s,'D03_VolumeRateImbalance'+s,['F01_QI','D03_VolumeRateImbalance'+s],'positive_activity',
            '成交加速时的队列方向','QI * max(VolumeRateImbalance, 0)','只在成交较前段加速时激活当前队列方向。',
            '成交加速与持续需求／供给同源；有效减速任务记 0。')
    return out


def reviewed_registry(base):
    base=base.copy()
    assert 'review_origin' not in base, 'Pass the original PDF catalog, not an already reviewed registry'
    rules=_decisions()
    base['original_expected_sign_signed']=base.expected_sign_signed
    base['original_expected_sign_absolute']=base.expected_sign_absolute
    base['original_logical_reason']=base.logical_reason
    base['original_prior_version']=base.prior_version
    base['review_origin']='original'
    base['review_change']='retained'
    base['hypothesis_condition']='沿用原登记的买卖压力机制及原有效性门槛。'
    base['alternative_mechanism']='统计方向仍可能随期限和状态改变。'
    base['parent_factors']='[]'
    base['review_version']=REVIEW_VERSION
    for i,r in base.iterrows():
        if r.kind=='quality':
            base.loc[i,['hypothesis_condition','alternative_mechanism']]='不适用'
            continue
        if r.expected_sign_signed==U:
            if _key(r.factor) in rules:
                sign,reason,condition,alternative=rules[_key(r.factor)]
                base.loc[i,['expected_sign_signed','logical_reason','hypothesis_condition','alternative_mechanism','review_change']]=[sign,reason,condition,alternative,'new_hypothesis']
            else:
                reason,potential=_remaining_reason(r)
                base.loc[i,['logical_reason','hypothesis_condition','alternative_mechanism']]=[reason,'当前定义未加入足以指定单一方向的信息。',potential]
        base.loc[i,'prior_status']=REVIEW_STATUS
        base.loc[i,'prior_version']=REVIEW_VERSION
    # Aliases inherit the actual target's reviewed sign and explanation.
    for i,r in base.loc[base.kind.eq('alias')].iterrows():
        target=base.set_index('factor').loc[r.alias_of]
        for col in ('expected_sign_signed','logical_reason','hypothesis_condition','alternative_mechanism','review_change'):
            base.loc[i,col]=target[col]
    additions=[]
    histories=tuple(sorted(base.loc[base.factor.str.match(r'A03_QIMean_h\d+s$'),'history_seconds'].unique()))
    for spec in variants(histories):
        parent=base.set_index('factor').loc[spec['parent']].to_dict()
        row={**parent,**{k:spec[k] for k in ('factor','name_zh','definition','logical_reason','hypothesis_condition','unit','expected_sign_signed')}}
        row.update(kind='computed',evaluate=True,alias_of='',role='direction',
            review_origin='derived',review_change='new_variant',
            original_expected_sign_signed='not_applicable',original_expected_sign_absolute='not_applicable',
            original_logical_reason='原版没有此派生表达。',original_prior_version='',
            expected_sign_absolute=U,prior_version=REVIEW_VERSION,prior_status=REVIEW_STATUS,
            family_id=spec['factor'].split('_')[0],
            definition_zh=spec['name_zh']+'；在信号时点组合已知父因子，全部父因子有效才计算。',
            parent_factors=json.dumps(spec['inputs']),
            parameters=json.dumps(dict(operation=spec['operation'],activity_scale=1,age_scale_seconds=1),sort_keys=True),
            alternative_mechanism='加入方向仅提供可检验假设；压力耗尽或反转可反号。交互与父因子共享信息，不能当独立证据。',
            reuse_expression='研究扩展（非 PDF 原式）；父因子：'+', '.join(spec['inputs']))
        # R05 derived activity deliberately belongs to the R05 comparison family.
        from utils.factor_catalog import FAMILIES
        row['family']=FAMILIES[row['family_id']]
        additions.append(row)
    return pd.concat([base,pd.DataFrame(additions)],ignore_index=True)


def variant_values(spec, values):
    """Vectorized row-local expressions. Caller must propagate missing parents."""
    x=[np.asarray(values[k],dtype=float) for k in spec['inputs']]
    op=spec['operation']
    if op=='product':return x[0]*x[1]
    if op=='signed_log':return x[0]*np.log1p(x[1])
    if op=='thin':return x[0]*np.exp(-np.logaddexp(0,x[1]))
    if op=='stability':return x[0]/(1+x[1])
    if op=='positive_activity':return x[0]*np.maximum(x[1],0)
    if op in ('aligned','opposed'):
        gate=x[0]*x[1]>0 if op=='aligned' else x[0]*x[1]<0
        return np.where(gate,x[0],0.)
    if op=='quote_rv':
        den=x[0]+x[1]+x[2]+x[3]
        signed=x[0]-x[1]+x[2]-x[3]
        bias=np.divide(signed,den,out=np.zeros_like(den),where=den!=0)
        return bias*(x[4]+x[5])/2
    raise ValueError(op)


def enrich_atomic(atomic,histories=(5,10)):
    """Only current-row factor columns are read; never label or future columns."""
    add={}
    for spec in variants(histories):
        valid=np.ones(len(atomic),dtype=bool)
        statuses=np.full(len(atomic),'ok',dtype=object)
        for parent in spec['inputs']:
            good=np.isfinite(atomic[parent].to_numpy(float)) & atomic[parent+'__status'].eq('ok').to_numpy()
            first=valid & ~good
            statuses[first]='PARENT_INVALID:'+parent+':'+atomic.loc[first,parent+'__status'].astype(str).to_numpy()
            valid &= good
        with np.errstate(invalid='ignore',divide='ignore',over='ignore'):
            value=variant_values(spec,atomic)
        statuses[valid & ~np.isfinite(value)]='INVALID_DERIVED_VALUE'
        value[~valid]=np.nan
        add[spec['factor']]=value
        add[spec['factor']+'__status']=statuses
    names=list(add)
    return pd.concat([atomic.drop(columns=names,errors='ignore'),pd.DataFrame(add,index=atomic.index)],axis=1)
