"""Selected L1 factors strictly confined to each task's observation interval."""
import numpy as np
from .opening_features import weighted_mean

BASE=['F03_OFI_latest','F04_QuoteShift','F07_QuotePosition']
CATALOG=[
 ('F03_OFI_latest','最新OFI','瞬时','positive','最新两快照OFI'),
 ('F04_QuoteShift','报价位移','瞬时','positive','(delta ask + delta bid)/2，tick'),
 ('F07_QuotePosition','末价盘口位置','瞬时','positive','(ask+bid-2*LastPrice)/(ask-bid)'),
 ('F01_QI','当前队列失衡','瞬时','positive','(bidSize-askSize)/(bidSize+askSize)'),
 ('F02_QIChange','最新失衡变化','瞬时','positive','QI最后值减前值'),
 ('F03_OFI_window','窗口OFI','持续压力','positive','区间内相邻快照OFI之和'),
 ('A03_QIMean','失衡均值','持续压力','positive','区间时间加权QI均值'),
 ('A04_QIDeviation','失衡偏离','持续压力','positive','当前QI减时间均值'),
 ('A07_OFIDirection','OFI方向持续程度','持续压力','positive','sum(OFI)/sum(abs(OFI))'),
 ('A07_OFIRate','归一OFI速率','持续压力','positive','sum(OFI)/(平均深度*实际跨度)'),
 ('A06_PositionDeviation','盘口位置偏离','持续压力','positive','当前位置减时间均值'),
 ('B04_AdditionDifference','买减卖增厚差','持续压力','positive','同价增厚量/(时间加权队列积分)，买减卖'),
 ('B05_DepletionDifference','卖减买变薄差','持续压力','positive','同价变薄量/(时间加权队列积分)，卖减买'),
 ('F08_SignedVolume','价格签名成交失衡','成交确认','uncertain','sum(sign(delta LastPrice)*delta Volume)/sum(delta Volume)'),
 ('D06_PreQIVolume','成交前失衡加权','成交确认','positive','sum(delta Volume*previous QI)/sum(delta Volume)'),
 ('D07_UnchangedPriceOFIRate','末价不变时OFI速率','成交确认','positive','末价未变边上的OFI/(平均深度*跨度)'),
 ('F05_Momentum','末价净位移','价格路径','uncertain','末价减区间第一价，tick'),
 ('F06_PathEfficiency','价格路径效率','价格路径','uncertain','净位移/累计绝对位移；无位移=0'),
 ('M01_MADeviation','末价偏离均价','价格路径','uncertain','末价减时间加权均价，tick'),
 ('C05_CountImbalance','涨跌次数失衡','价格路径','uncertain','(上涨次数-下跌次数)/(上涨次数+下跌次数)'),
 ('A01_Spread','当前价差','状态','uncertain','ask-bid，tick'),
 ('A02_LogDepth','当前对数深度','状态','uncertain','log1p(bidSize+askSize)，固定表达变换'),
 ('A02_DepthLogRatio','深度相对窗口均值','状态','uncertain','log(当前深度/时间均值)'),
 ('C01_PriceRV','价格波动强度','状态','uncertain','sqrt(sum(delta LastPrice²)/实际跨度)'),
 ('D01_LogVolumeRate','对数成交速率','状态','uncertain','log1p(sum(delta Volume)/实际跨度)，固定表达变换'),
]
INTERACTIONS={
 'X_OFI_Volume':('F03_OFI_latest','D01_LogVolumeRate'),
 'X_Position_Volume':('F07_QuotePosition','D01_LogVolumeRate'),
 'X_OFI_Spread':('F03_OFI_latest','A01_Spread'),
 'X_Pressure_Path':('A07_OFIDirection','F06_PathEfficiency'),
 'X_QI_Volume':('A03_QIMean','D01_LogVolumeRate'),
 'X_Position_RV':('F07_QuotePosition','C01_PriceRV'),
}
NAMES={r[0]:r[1] for r in CATALOG}
for name,parents in INTERACTIONS.items():NAMES[name]=' × '.join(NAMES[f] for f in parents)
ALL=[r[0] for r in CATALOG]+list(INTERACTIONS)

def bounded_features(d,task,window):
    end=task+window;lo=int(np.searchsorted(d.times,task,'left'));i=d.source_index(end)
    out={f:np.nan for f in ALL};out.update(snapshot_count=max(0,i-lo+1),source_seconds=np.nan,used_start=np.nan,
                                      source_age=np.nan,min_coverage=np.nan)
    if i<lo or i<0 or end-d.times[i]>.5+1e-10:return out
    out.update(source_seconds=float(d.times[i]),source_age=float(end-d.times[i]))
    used=[i];cover=[]
    def edge(group):return i>lo and d.edge[group][i]
    if edge('ofi'):out['F03_OFI_latest']=d.ofi[i];used.append(i-1)
    if edge('book'):out['F04_QuoteShift']=(d.a[i]-d.a[i-1]+d.b[i]-d.b[i-1])/2;used.append(i-1)
    if d.valid['position'][i]:out['F07_QuotePosition']=(d.a[i]+d.b[i]-2*d.p[i])/(d.a[i]-d.b[i])
    if d.valid['queue'][i]:out['F01_QI']=d.qi[i];out['A02_LogDepth']=np.log1p(d.depth[i])
    if edge('queue'):out['F02_QIChange']=d.qi[i]-d.qi[i-1];used.append(i-1)
    if d.valid['book'][i]:out['A01_Spread']=d.a[i]-d.b[i]
    def win(*groups):
        if not all(d.valid[g][i] for g in groups):return None
        start=max([lo]+[int(d.segment[g][i]) for g in groups])
        span=d.times[i]-d.times[start]
        if span<=0 or span/window<.8-1e-10:return None
        used.append(start);cover.append(span/window)
        return start,span,np.diff(d.times[start:i+1])
    z=win('queue')
    if z:
        s,c,w=z;mean=weighted_mean(d.qi[s:i],w)
        out['A03_QIMean']=mean;out['A04_QIDeviation']=d.qi[i]-mean
        out['A02_DepthLogRatio']=np.log(d.depth[i]/weighted_mean(d.depth[s:i],w))
    z=win('position')
    if z:
        s,c,w=z;position=(d.a[s:i]+d.b[s:i]-2*d.p[s:i])/(d.a[s:i]-d.b[s:i])
        out['A06_PositionDeviation']=out['F07_QuotePosition']-weighted_mean(position,w)
    z=win('ofi')
    if z:
        s,c,w=z;e=d.ofi[s+1:i+1];depth=weighted_mean(d.depth[s:i],w)
        out['F03_OFI_window']=e.sum();out['A07_OFIRate']=e.sum()/(depth*c)
        if np.abs(e).sum()>0:out['A07_OFIDirection']=e.sum()/np.abs(e).sum()
        addition=[];depletion=[]
        for q,p in [(d.qb,d.b),(d.qa,d.a)]:
            dq=np.diff(q[s:i+1]);same=np.diff(p[s:i+1])==0;den=np.dot(q[s:i],w)
            addition.append(np.sum(np.maximum(dq,0)*same)/den);depletion.append(np.sum(np.maximum(-dq,0)*same)/den)
        out['B04_AdditionDifference']=addition[0]-addition[1];out['B05_DepletionDifference']=depletion[1]-depletion[0]
    z=win('price')
    if z:
        s,c,w=z;moves=np.diff(d.p[s:i+1]);net=d.p[i]-d.p[s];path=np.abs(moves).sum()
        out['F05_Momentum']=net;out['F06_PathEfficiency']=net/path if path>0 else 0.
        out['M01_MADeviation']=d.p[i]-weighted_mean(d.p[s:i],w)
        out['C01_PriceRV']=np.sqrt(np.dot(moves,moves)/c)
        nz=np.count_nonzero(moves)
        if nz:out['C05_CountImbalance']=(np.count_nonzero(moves>0)-np.count_nonzero(moves<0))/nz
    z=win('volume')
    if z:
        s,c,w=z;out['D01_LogVolumeRate']=np.log1p(d.dv[s+1:i+1].sum()/c)
    z=win('price_volume')
    if z:
        s,c,w=z;v=d.dv[s+1:i+1]
        if v.sum()>0:out['F08_SignedVolume']=np.dot(np.sign(np.diff(d.p[s:i+1])),v)/v.sum()
    z=win('volume','queue')
    if z:
        s,c,w=z;v=d.dv[s+1:i+1]
        if v.sum()>0:out['D06_PreQIVolume']=np.dot(d.qi[s:i],v)/v.sum()
    z=win('price','ofi')
    if z:
        s,c,w=z;out['D07_UnchangedPriceOFIRate']=np.dot(d.ofi[s+1:i+1],np.diff(d.p[s:i+1])==0)/(c*weighted_mean(d.depth[s:i],w))
    for name,(a,b) in INTERACTIONS.items():out[name]=out[a]*out[b]
    out['used_start']=float(d.times[min(used)]);out['min_coverage']=min(cover) if cover else np.nan
    assert out['used_start']>=task and out['source_seconds']<=end
    return out
