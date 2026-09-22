"""Declared outputs and hypotheses, frozen independently of IC results."""
from functools import lru_cache
import json
import pandas as pd

PRIOR_VERSION = "l1-logical-hypotheses-20260922"
FAMILIES = {
    'F01':'当前队列失衡','F02':'最新失衡变化','F03':'快照订单流失衡','F04':'最新报价整体移动',
    'F05':'末价净位移','F06':'有方向路径效率','F07':'末价相对盘口位置','F08':'按末价方向加权成交',
    'A01':'价差状态与路径','A02':'深度水平与异常','A03':'失衡均值与波动','A04':'失衡偏离与趋势',
    'A05':'失衡持续性','A06':'盘口偏离历史','A07':'OFI 方向与活动','A08':'OFI 脉冲与速率变化',
    'B01':'时均队列失衡','B02':'两侧挂量相对波动','B03':'两侧挂量趋势','B04':'端点同价净增厚',
    'B05':'端点同价净变薄','B06':'两侧报价变化频率','B07':'报价驻留时间','B08':'报价波动与同步','B09':'可观测队列恢复',
    'C01':'末价路径波动','C02':'上下行平方变化','C03':'振幅与区间位置','C04':'有顺序的回落与回升',
    'C05':'价变频率与方向次数','C06':'反转与连续次数','C07':'末价变化年龄','C08':'净位移速度变化',
    'D01':'成交活动强度','D02':'成交脉冲','D03':'成交活动前后变化','D04':'成交与末价不变',
    'D05':'成交相对深度','D06':'成交加权前置失衡','D07':'末价不变时的 OFI','D08':'单位成交量价变','D09':'历史量价流相关',
    'M01':'滚动均线偏离','M02':'开盘累计均线偏离','M03':'标准化均线偏离','M04':'偏离变化速度',
    'M05':'均线变化速度','M06':'短长均线距离','M07':'均线上下持续性','M08':'均线穿越节奏',
    'M09':'偏离路径形态','M10':'成交加权末价代理偏离','M11':'局部趋势与残差',
    'R01':'相对观测开盘起点','R02':'已观测开盘区间','R03':'此前极值突破','R04':'相对自身开盘历史',
    'R05':'近期与较早状态比较','R06':'开盘阶段与数据质量',
}
PARAMETERS = dict(history_seconds=[5,10], segment_seconds=[2,5], outer_seconds=5,
                  lag_seconds=2, qi_band=.2, mean_band_ticks=.5, recovery_seconds=2,
                  recovery_drop=.1, recovery_min_same_events=3, recovery_min_successes=1,
                  historical_correlation_min_pairs=8, slope_min_states=3)


@lru_cache(maxsize=8)
def _rows(histories):
    out=[]
    def add(fid,stem,title,formula,role='direction',h=0,prior='uncertain',absolute='uncertain',
            dep='price',unit='ratio',suffix='',params=None,kind='computed',alias=''):
        name=f'{fid}_{stem}'+(f'_h{h}s' if h else '')+suffix
        if kind=='quality':role='quality';prior=absolute='not_applicable'
        elif role=='magnitude_state':prior='uncertain'
        reason={'positive':'买方压力增强、卖方供给减弱或盘口位置向上传递的假设，非结果保证。',
                'negative':'与买方压力相反的供给或恢复耗时机制假设，非结果保证。',
                'uncertain':'可能存在延续、反转或条件依赖，未指定单一方向。',
                'not_applicable':'描述幅度、状态或质量，不预设上涨方向。'}[prior]
        out.append(dict(factor=name,family_id=fid,family=FAMILIES[fid],name_zh=title,
                        definition=formula,definition_zh=title+'；只使用信号时刻已经观测且满足有效性条件的数据。',role=role,history_seconds=h,unit=unit,dependency=dep,
                        parameters=json.dumps(params or {},ensure_ascii=False,sort_keys=True),kind=kind,
                        alias_of=alias,evaluate=kind=='computed',expected_sign_signed=prior,
                        expected_sign_absolute=absolute,logical_reason=reason,
                        prior_version=PRIOR_VERSION,prior_status='研究假设；已看过部分期货结果；ETF 分析前可冻结'))
        return name
    P='positive';N='negative';U='uncertain';S='magnitude_state'
    add('F01','QI','当前队列失衡','(BidVolume1-AskVolume1)/(BidVolume1+AskVolume1)',prior=P,dep='queue')
    add('F02','QIChange','最新失衡变化','QI[t]-QI[t-1]',prior=P,dep='queue')
    add('F03','OFI_latest','最新原始 OFI','BidFlow[t]-AskFlow[t]，换价分支沿用 PDF',prior=P,dep='ofi',unit='volume')
    add('F03','NOFI_latest','最新归一化 OFI','OFI_latest / mean(Depth[t-1],Depth[t])',prior=P,dep='ofi')
    add('F04','QuoteShift','最新报价整体移动','(DeltaAsk+DeltaBid)/(2*tickSize)',prior=P,dep='book',unit='tick')
    add('F05','LastChange','最新末价变化','(LastPrice[t]-LastPrice[t-1])/tickSize',unit='tick')
    add('F07','QuotePosition','末价相对盘口位置','(Ask+Bid-2*LastPrice)/(Ask-Bid)，不裁剪',prior=P,dep='position')
    add('A01','Spread','当前价差','(Ask-Bid)/tickSize',S,dep='book',unit='tick')
    add('A01','SpreadChange','最新价差变化','Spread[t]-Spread[t-1]',S,dep='book',unit='tick')
    add('A02','Depth','当前总深度','BidVolume1+AskVolume1',S,dep='queue',unit='volume')
    add('A05','QIState','当前失衡三状态','QI>0.2 为 +1；QI<-0.2 为 -1；其余为 0',prior=P,dep='queue',params={'band':.2})
    add('A05','StateAge','当前失衡状态年龄','距最近一次三状态变化；左删失时主值缺失',S,dep='queue',unit='second')
    for fid,stem,title,dep in [('A05','StateAge','失衡状态','queue'),('B07','BidAge','买价','book'),('B07','AskAge','卖价','book'),('C07','PriceAge','末价','price'),('D04','VolumeAge','成交增量','volume')]:
        if fid!='A05':add(fid,stem,title+'事件年龄','距最近合法事件；未观察到事件时主值缺失',S,dep=dep,unit='second')
        add(fid,stem+'LowerBound',title+'年龄下界','从当前连续有效片段起点起算的可观测下界',h=0,kind='quality',dep=dep,unit='second')
        add(fid,stem+'LeftCensored',title+'年龄左删失标记','1 表示尚未观察到事件真实起点',kind='quality',dep=dep,unit='flag')
    add('B07','AgeDifference','买价减卖价驻留年龄','BidAge-AskAge；任一年龄删失则缺失',dep='book',unit='second')
    add('M02','OpeningMADeviation','开盘时间均价偏离','(p-合法已完成区间左端价格时间均值)/tickSize',unit='tick')
    add('M10','OpeningVWDeviation','开盘成交加权末价偏离','(p-sum(p[i]*dV[i])/sum(dV[i]))/tickSize',dep='price_volume',unit='tick')
    add('R01','OpeningMove','相对首个有效末价位移','(p-本次开盘首个有效 p)/tickSize',unit='tick')
    for stem,title,formula,role in [('DistanceHigh','距已观测高点','(p-H_open)/tickSize', 'direction'),('DistanceLow','距已观测低点','(p-L_open)/tickSize','direction'),('OpeningRange','已观测开盘振幅','(H_open-L_open)/tickSize',S),('OpeningLocation','已观测开盘位置','(2*p-H_open-L_open)/(H_open-L_open)','direction')]:
        add('R02',stem,title,formula,role,unit='ratio' if stem.endswith('Location') else 'tick')
    for stem,title,formula,unit in [('PriorHighDistance','相对此前高点','(p-H_prior)/tickSize','tick'),('PriorLowDistance','相对此前低点','(p-L_prior)/tickSize','tick'),('NewHigh','突破此前高点','p>H_prior','flag'),('NewLow','突破此前低点','p<L_prior','flag')]:
        add('R03',stem,title,formula,unit=unit)
    for h in histories:
        def a(fid,stem,title,formula,role='direction',prior=U,absolute=U,dep='price',unit='ratio',suffix='',params=None,kind='computed',alias=''):
            return add(fid,stem,title,formula,role,h,prior,absolute,dep,unit,suffix,params,kind,alias)
        a('F03','OFI','窗口原始 OFI','sum(OFI[i])',prior=P,dep='ofi',unit='volume')
        a('F03','NOFI','窗口归一化 OFI','sum(OFI[i])/timeMean(Depth)',prior=P,dep='ofi')
        a('F05','Momentum','窗口末价净位移','(p[t]-p[s])/tickSize',unit='tick')
        a('F06','PathEfficiency','有方向路径效率','sum(r)/sum(abs(r))；合格静止路径记 0')
        a('F08','SignedVolume','按末价方向加权成交','sum(sign(r)*dV)/sum(dV)，零价变权重为 0',dep='price_volume')
        for stem,title,formula in [('SpreadMean','时间平均价差','timeMean(Spread)'),('SpreadMax','最大价差','max(Spread)，含当前点'),('WideSpreadShare','宽价差占时比','timeMean(Spread>1)')]:
            a('A01',stem,title,formula,S,dep='book',unit='ratio' if stem.endswith('Share') else 'tick')
        for stem,title,formula in [('DepthMean','时间平均深度','timeMean(Depth)'),('DepthCV','深度相对波动','timeStd(Depth)/timeMean(Depth)'),('DepthLogRatio','当前深度对数比','log(Depth[t]/timeMean(Depth))')]:
            a('A02',stem,title,formula,S,dep='queue',unit='volume' if stem=='DepthMean' else 'ratio')
        a('A03','QIMean','失衡时间均值','timeMean(QI)',prior=P,dep='queue')
        a('A03','QIStd','失衡时间标准差','timeStd(QI)',S,dep='queue')
        a('A04','QIDeviation','失衡相对历史偏离','QI[t]-timeMean(QI)',prior=P,dep='queue')
        a('A04','QISlope','失衡历史斜率','timeWeightedSlope(QI)',prior=P,dep='queue',unit='per_second')
        for stem,title,formula,role,prior in [('BuyShare','偏买占时比','timeMean(QI>0.2)','direction',P),('SellShare','偏卖占时比','timeMean(QI<-0.2)','direction',N),('SwitchRate','失衡状态切换率','count(state[i]!=state[i-1])/coverage',S,U)]:
            a('A05',stem,title,formula,role,prior,dep='queue',unit='per_second' if stem=='SwitchRate' else 'ratio')
        for stem,title,formula,role,prior in [('PositionMean','盘口偏离均值','timeMean(G)','direction',P),('PositionAbsMean','盘口绝对偏离均值','timeMean(abs(G))',S,U),('PositionDeviation','当前盘口偏离变化','G[t]-timeMean(G)','direction',P)]:
            a('A06',stem,title,formula,role,prior,dep='position')
        a('A07','OFIDirection','OFI 单向程度','sum(e)/sum(abs(e))',prior=P,dep='ofi')
        a('A07','OFIActivity','OFI 总活动速率','sum(abs(e))/(timeMean(Depth)*coverage)',S,dep='ofi',unit='per_second')
        a('A07','OFIRate','有方向 OFI 速率','sum(e)/(timeMean(Depth)*coverage)',prior=P,dep='ofi',unit='per_second')
        a('A08','OFIPulse','最大 OFI 脉冲占比','max(abs(e))/sum(abs(e))',S,dep='ofi')
        a('B01','MeanQueueImbalance','时均挂量失衡','(timeMean(qb)-timeMean(qa))/(timeMean(qb)+timeMean(qa))',prior=P,dep='queue')
        for side in ('Bid','Ask'):
            a('B01',side+'Mean',side+'挂量时间均值','timeMean(q'+side+')',S,dep='queue',unit='volume')
            a('B02',side+'CV',side+'挂量相对波动','timeStd(q)/timeMean(q)',S,dep='queue')
            a('B03',side+'Slope',side+'对数挂量斜率','timeWeightedSlope(log(q))',dep='queue',unit='per_second')
            a('B04',side+'Addition',side+'同价净增厚','sum(sameQuote*positive(dq))/(coverage*timeMean(q))',prior=P if side=='Bid' else N,dep='ofi',unit='per_second')
            a('B04',side+'SameShare',side+'同价区间占时比','sum(dt*sameQuote)/coverage',S,dep='ofi')
            a('B04',side+'SameCount',side+'同价区间数','count(sameQuote)',dep='ofi',unit='count',kind='quality')
            a('B05',side+'Depletion',side+'同价净变薄','sum(sameQuote*positive(-dq))/(coverage*timeMean(q))',prior=N if side=='Bid' else P,dep='ofi',unit='per_second')
            for direction in ('Up','Down'):
                a('B06',side+direction+'Rate',side+direction+'报价变化率','count(quoteMove has specified sign)/coverage',dep='book',unit='per_second')
            a('B08',side+'RV',side+'报价路径波动','sqrt(sum(quoteMoveTicks**2)/coverage)',S,dep='book',unit='tick/sqrt(second)')
        for fid,stem,title,formula,role,prior,dep,unit in [
            ('B02','CVDifference','买减卖相对波动','BidCV-AskCV',S,U,'queue','ratio'),
            ('B03','SlopeDifference','买减卖挂量趋势','BidSlope-AskSlope','direction',P,'queue','per_second'),
            ('B04','AdditionDifference','买减卖同价增厚','BidAddition-AskAddition','direction',P,'ofi','per_second'),
            ('B05','DepletionDifference','卖减买同价变薄','AskDepletion-BidDepletion','direction',P,'ofi','per_second'),
            ('B06','UpDifference','买减卖上移频率','BidUpRate-AskUpRate','direction',U,'book','per_second'),
            ('B06','DownDifference','卖减买下移频率','AskDownRate-BidDownRate','direction',U,'book','per_second'),
            ('B08','RVDifference','买减卖报价波动','BidRV-AskRV',S,U,'book','tick/sqrt(second)')]:
            a(fid,stem,title,formula,role,prior,dep=dep,unit=unit)
        for stem,title in [('CoShare','双侧同向占比'),('OppositeShare','双侧反向占比'),('SingleShare','仅单侧移动占比')]:
            a('B08',stem,title,'specified quote move intervals / any quote move intervals',S,dep='book')
        for stem,title,formula,unit in [('PriceRV','价格平方路径强度','sqrt(sum(r**2)/coverage)','tick/sqrt(second)'),('PriceAV','价格绝对路径速率','sum(abs(r))/coverage','tick/second')]:
            a('C01',stem,title,formula,S,absolute=P,unit=unit)
        for stem,title,formula,role in [('UpSquares','上涨平方和','sum(r**2*(r>0))',S),('DownSquares','下跌平方和','sum(r**2*(r<0))',S),('Asymmetry','上下行平方不对称','(UpSquares-DownSquares)/(UpSquares+DownSquares)','direction')]:
            a('C02',stem,title,formula,role,unit='ratio' if stem=='Asymmetry' else 'tick_squared')
        a('C02','TotalSquares','上下行平方和分母','UpSquares+DownSquares',kind='quality',unit='tick_squared')
        a('C03','Range','窗口振幅','max(p)-min(p) in ticks',S,absolute=P,unit='tick')
        a('C03','Location','窗口区间位置','(2*p[t]-max(p)-min(p))/(max(p)-min(p))')
        for stem,title,formula in [('Drawdown','有顺序最大回落','max(p[i]-p[j]), i<=j in ticks'),('Runup','有顺序最大回升','max(p[j]-p[i]), i<=j in ticks')]:a('C04',stem,title,formula,S,unit='tick')
        for stem,title,formula,role,unit in [('MoveRate','价变频率','count(r!=0)/coverage',S,'per_second'),('CountImbalance','上下行次数不对称','(count(r>0)-count(r<0))/count(r!=0)','direction','ratio'),('ZeroCountShare','零价变区间数占比','count(r==0)/nIntervals',S,'ratio'),('ZeroTimeShare','零价变区间时长占比','sum(dt*(r==0))/coverage',S,'ratio'),('UpCount','上涨区间数','count(r>0)',S,'count'),('DownCount','下跌区间数','count(r<0)',S,'count'),('ZeroCount','零价变区间数','count(r==0)',S,'count')]:a('C05',stem,title,formula,role,unit=unit)
        for stem,title,formula,role,unit in [('ReversalShare','非零方向反转比例','adjacent nonzero sign changes/(K-1)',S,'ratio'),('RunCount','末尾同向连续次数','same-sign suffix length of nonzero moves',S,'count'),('SignedRun','带方向连续次数','lastNonzeroSign*RunCount','direction','count')]:a('C06',stem,title,formula,role,unit=unit)
        a('C06','NonzeroCount','非零价变支持数','count(r!=0)',kind='quality',unit='count')
        a('D01','VolumeTotal','成交总增量','sum(dV)',S,dep='volume',unit='volume')
        a('D01','VolumeRate','成交速率','sum(dV)/coverage',S,absolute=P,dep='volume',unit='volume/second')
        for stem,title,formula,unit in [('RateStd','区间成交速率标准差','durationWeightedStd(dV/dt)','volume/second'),('RateCV','成交速率相对波动','RateStd/VolumeRate','ratio'),('LatestRateRatio','最新成交速率比','(last_dV/last_dt)/VolumeRate','ratio')]:a('D02',stem,title,formula,S,dep='volume',unit=unit)
        a('D04','UnchangedPriceVolumeShare','末价未变成交占比','sum(dV*(r==0))/sum(dV)',S,dep='price_volume')
        a('D04','ZeroVolumeTimeShare','无成交增量时长占比','sum(dt*(dV==0))/coverage',S,dep='volume')
        a('D05','VolumeDepthRate','成交相对深度速率','sum(dV)/(coverage*timeMean(Depth))',S,dep='volume_depth',unit='per_second')
        a('D06','PreQIVolume','成交加权前置失衡','sum(QI[i-1]*dV[i])/sum(dV)',prior=P,dep='volume_depth')
        a('D07','UnchangedPriceOFIRate','末价未变 OFI 速率','sum(e*(r==0))/(coverage*timeMean(Depth))',prior=P,dep='price_ofi',unit='per_second')
        a('D08','MovePerVolume','单位成交量末价路径','sum(abs(r))/sum(dV)',S,dep='price_volume',unit='tick/volume')
        for stem,title,formula in [('SignedVolumeCorr','历史价变速度与成交速率相关','weightedCorr(r/dt,dV/dt)'),('AbsoluteVolumeCorr','历史绝对价变速度与成交速率相关','weightedCorr(abs(r/dt),dV/dt)')]:a('D09',stem,title,formula,S,dep='price_volume',params={'min_pairs':8})
        a('D09','SyncPairCount','同期历史配对数','legal price-volume intervals',kind='quality',dep='price_volume',unit='count')
        a('M01','MADeviation','当前末价相对滚动均线','p[t]-timeMean(p) in ticks',unit='tick')
        a('M03','StandardDeviation','标准化均线偏离','(p[t]-timeMean(p))/timeStd(p)')
        for fid,stem,title,formula,role,unit in [('M04','DeviationSpeed','有方向偏离变化速度','(B[t]-B[k])/actualLag','direction','tick/second'),('M04','AbsDeviationSpeed','绝对偏离扩大速度','(abs(B[t])-abs(B[k]))/actualLag',S,'tick/second'),('M05','MeanSpeed','均线变化速度','(MA[t]-MA[k])/actualLag','direction','tick/second')]:a(fid,stem,title,formula,role,unit=unit,suffix='_lag2s',params={'lag':2})
        for stem,title,formula,role in [('AboveShare','均线上方占时比','outerTimeMean(B>0.5)','direction'),('BelowShare','均线下方占时比','outerTimeMean(B<-0.5)','direction'),('MeanState','当前均线三状态','state(B, band=0.5)','direction'),('MeanStateAge','均线状态年龄','age since observed state change',S)]:a('M07',stem,title,formula,role,suffix='_g5s',params={'outer':5,'band_ticks':.5},unit='second' if stem.endswith('Age') else 'ratio')
        for stem,title,formula,unit in [('CrossUpRate','向上完整穿越频率','count(lastNonzeroState=-1,currentState=+1)/outerCoverage','per_second'),('CrossDownRate','向下完整穿越频率','count(lastNonzeroState=+1,currentState=-1)/outerCoverage','per_second'),('CrossAge','距最近完整穿越时间','age since observed complete crossing','second')]:a('M08',stem,title,formula,S,suffix='_g5s',params={'outer':5,'band_ticks':.5},unit=unit)
        for fid,stem in [('M07','MeanStateAge'),('M08','CrossAge')]:
            a(fid,stem+'LowerBound','派生状态年龄下界','observed age lower bound',kind='quality',unit='second',suffix='_g5s')
            a(fid,stem+'LeftCensored','派生状态年龄删失标记','1 if exact event start is unknown',kind='quality',unit='flag',suffix='_g5s')
        for stem,title,formula,role in [('AbsDeviationMean','绝对偏离历史均值','outerTimeMean(abs(B))',S),('DeviationStd','偏离历史标准差','outerTimeStd(B)',S),('MaxPositive','历史最大正偏离','max(0,max(B))','direction'),('MinNegative','历史最小负偏离','min(0,min(B))','direction')]:a('M09',stem,title,formula,role,unit='tick',suffix='_g5s',params={'outer':5})
        a('M10','VWDeviation','成交加权末价代理偏离','p[t]-sum(p[i]*dV[i])/sum(dV) in ticks',dep='price_volume',unit='tick')
        a('M11','TrendSlope','历史价格趋势斜率','timeWeightedSlope(p) excluding current point',unit='tick/second')
        a('M11','TrendResidual','当前价格趋势线残差','p[t]-(alpha+beta*time[t]) in ticks; fit excludes t',unit='tick')
        for stem,title,dep in [('VolumeRate','成交速率','volume'),('PriceRV','价格波动','price')]:
            for expr in ('Ratio','Percentile'):a('R04',stem+expr,title+'开盘历史'+expr,'current / prior timeMean; or weighted midpoint percentile',S,dep=dep)
        for dep in ('price','book','queue','ofi','volume','price_volume','position','volume_depth','price_ofi','full'):
            for stem,title,unit in [('Coverage','实际覆盖','second'),('Count','窗口快照数','count'),('Warmup','预热不足','flag'),('CoverageRatio','窗口覆盖比例','ratio'),('StartSeconds','窗口起点开盘秒数','second'),('EndSeconds','窗口终点开盘秒数','second'),('IncrementCount','有效相邻增量数','count'),('MaxInterval','有效窗口最大间隔','second'),('GapFlag','名义窗口包含无效片段','flag')]:a('R06',dep+'_'+stem,dep+title,'dependency-specific history support',kind='quality',dep=dep,unit=unit)
        a('R06','price_MoveCount','合法窗口价变数','count(nonzero legal price increments)',kind='quality',dep='price',unit='count')
        a('R06','volume_PositiveCount','合法窗口成交正增量数','count(positive legal volume increments)',kind='quality',dep='volume',unit='count')
        a('R06','DerivedCoverage','均线外层有效覆盖','contiguous valid historical MA deviation coverage',kind='quality',unit='second',suffix='_g5s')
    add('M06','MeanGap','短长均线距离','M01_h10s-M01_h5s',h=10,suffix='_short5s',unit='tick',params={'short':5,'long':10})
    add('M06','SameWindow','短长均线同一有效起点','1 if effective starts are equal',h=10,suffix='_short5s',kind='quality',unit='flag')
    for lag in (2,5):
        suff=f'_lag{lag}s';params={'segment_seconds':lag}
        for fid,stem,title,formula,role,prior,dep,unit in [
            ('A08','OFIRateChange','前后 OFI 速率差','OFIRate_recent-OFIRate_earlier','direction',P,'ofi','per_second'),
            ('C08','MomentumSpeedChange','前后净位移速度差','priceSpeed_recent-priceSpeed_earlier','direction',U,'price','tick/second'),
            ('D03','VolumeRateChange','前后成交速率差','VolumeRate_recent-VolumeRate_earlier',S,U,'volume','volume/second'),
            ('D03','VolumeRateImbalance','前后成交速率归一化差','(recent-earlier)/(recent+earlier)',S,U,'volume','ratio'),
            ('R05','DepthChange','前后深度均值差','DepthMean_recent-DepthMean_earlier',S,U,'queue','volume'),
            ('R05','RVChange','前后价格波动差','PriceRV_recent-PriceRV_earlier',S,U,'price','tick/sqrt(second)')]:add(fid,stem,title,formula,role,prior=prior,dep=dep,unit=unit,suffix=suff,params=params)
        for stem,target,title,prior in [('VolumeRateChange','D03_VolumeRateChange','成交速率变化复用',U),('OFIRateChange','A08_OFIRateChange','OFI 速率变化复用',P),('MomentumSpeedChange','C08_MomentumSpeedChange','价格速度变化复用',U)]:add('R05',stem,title,'alias of '+target+suff,prior=prior,suffix=suff,params=params,kind='alias',alias=target+suff,unit={'VolumeRateChange':'volume/second','OFIRateChange':'per_second','MomentumSpeedChange':'tick/second'}[stem],role=S if stem=='VolumeRateChange' else 'direction')
    for stem,title,exprs,dep in [('Spread','价差',('Difference','Percentile'),'book'),('Depth','深度',('LogRatio','Percentile'),'queue')]:
        for expr in exprs:add('R04',stem+expr,title+'开盘历史'+expr,'current versus completed historical states, time weighted',S,dep=dep,unit='tick' if stem=='Spread' and expr=='Difference' else 'ratio')
    for side in ('Bid','Ask'):
        for stem,title,unit in [('Recovery','平均恢复比例','ratio'),('FullShare','完全恢复比例','ratio'),('SuccessTime','成功子集平均恢复耗时','second'),('ExitShare','报价退出成熟事件占比','ratio')]:add('B09',side+stem,side+title,'fixed matured nonoverlapping recovery events; see PDF B09',h=10,dep='ofi',unit=unit,params={'event_seconds':2,'drop':.1,'min_same_events':3})
        for stem in ('MatureCount','SameCount','ExitCount','SuccessCount','ImmatureCount','GappedCount','UnrecoveredCount'):
            add('B09',side+stem,side+'恢复事件 '+stem,'event support count',h=10,kind='quality',dep='ofi',unit='count')
    for stem,prior,unit in [('RecoveryDifference',P,'ratio'),('FullShareDifference',P,'ratio'),('SuccessTimeDifference',N,'second')]:add('B09',stem,'买减卖恢复指标差','Bid-Ask; both estimates must be valid',h=10,prior=prior,dep='ofi',unit=unit)
    add('D09','LagOFICorr','历史 OFI 与后续历史价变相关','weightedCorr(e/(meanAdjacentDepth*dt), historical price speed at lag2)',S,h=10,dep='full',suffix='_lag2s',params={'lag':2,'min_pairs':8})
    add('D09','LagPairCount','历史滞后配对数','matured legal lagged pairs',h=10,kind='quality',dep='full',unit='count',suffix='_lag2s')
    add('D09','LagAlignmentMax','历史滞后最大对齐误差','max matched time-target time',h=10,kind='quality',dep='full',unit='second',suffix='_lag2s')
    for dep in ('price','book','queue','ofi','volume','price_volume','position','volume_depth','price_ofi','full'):
        for stem,title,unit in [('SegmentAge','连续片段年龄','second'),('OpeningCoverage','累计合法覆盖','second'),('OpeningCoverageRatio','累计开盘覆盖比例','ratio'),('StartDelay','首个有效观测延迟','second'),('GapCount','截至当前历史中断数','count')]:add('R06',dep+'_'+stem,dep+title,'dependency-specific opening quality',kind='quality',dep=dep,unit=unit)
    for stem,title,unit in [('Elapsed','开盘经过时间','second'),('LatestInterval','最新间隔','second'),('AsyncPriceVolume','零新增成交但末价变化','flag')]:add('R06',stem,title,'as observed at feature source snapshot',kind='quality',unit=unit)
    pages={'F01':5,'F02':5,'F03':5,'F04':6,'F05':6,'F06':6,'F07':7,'F08':7,
           'A01':7,'A02':8,'A03':8,'A04':8,'A05':9,'A06':9,'A07':9,'A08':10,
           'B01':10,'B02':10,'B03':11,'B04':11,'B05':11,'B06':12,'B07':12,'B08':12,'B09':13,
           'C01':13,'C02':14,'C03':14,'C04':14,'C05':15,'C06':15,'C07':15,'C08':16,
           'D01':16,'D02':16,'D03':17,'D04':17,'D05':17,'D06':17,'D07':18,'D08':18,'D09':18,
           'M01':19,'M02':19,'M03':20,'M04':20,'M05':20,'M06':21,'M07':21,'M08':21,'M09':22,'M10':22,'M11':22,
           'R01':23,'R02':23,'R03':23,'R04':24,'R05':24,'R06':25}
    reuse={'M06':'M01_h10s-M01_h5s；代数复用，非独立信息源','M04':'有方向偏离速度=同对齐价格速度-M05均线速度',
           'A07':'OFIRate=F03_NOFI/实际覆盖；共享 OFI 底层量','C01':'绝对价变路径与 F06/D08 共用',
           'D08':'绝对价变路径与 F06/C01 共用','R02':'与 C03 共用极值定义，历史范围不同','C02':'TotalSquares 为上下行分量之和'}
    for row in out:
        row['source_pdf_page']=pages[row['family_id']]
        row['reuse_expression']=('alias of '+row['alias_of']) if row['kind']=='alias' else reuse.get(row['family_id'],'')
    ids=[r['factor'] for r in out]
    assert len(ids)==len(set(ids))
    assert set(r['family_id'] for r in out)==set(FAMILIES)
    order={fid:i for i,fid in enumerate(FAMILIES)}
    return sorted(out,key=lambda r:(order[r['family_id']],r['history_seconds'],r['factor']))


def full_registry(config):
    return pd.DataFrame(_rows(tuple(config.history_seconds)))
