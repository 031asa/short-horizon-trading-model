"""Sequential full L1 feature stream. No feature accesses rows after its index."""
import math
import numpy as np
from .factor_catalog import full_registry


def weighted_mean(x,w):
    return float(np.dot(x,w)/w.sum())


def weighted_std(x,w):
    mean=weighted_mean(x,w)
    return math.sqrt(max(0.,weighted_mean((x-mean)**2,w)))


def slope(x,t,w):
    if len(x)<3 or len(np.unique(t[w>0]))<3:return np.nan
    z=t-weighted_mean(t,w)
    denominator=np.dot(w,z*z)
    return float(np.dot(w*z,x-weighted_mean(x,w))/denominator) if denominator>0 else np.nan


def weighted_corr(x,y,w,minimum=8):
    if len(x)<minimum:return np.nan,'NOT_ENOUGH_EVENTS'
    x=x-weighted_mean(x,w);y=y-weighted_mean(y,w)
    scale=math.sqrt(np.dot(w,x*x)*np.dot(w,y*y))
    return (float(np.clip(np.dot(w*x,y)/scale,-1,1)),'ok') if scale>0 else (np.nan,'ZERO_VARIANCE')


class FeatureEngine:
    def __init__(self,data):
        self.d=data;self.cfg=data.config
        self.catalog=full_registry(self.cfg).to_dict('records')
        self.by_name={x['factor']:x for x in self.catalog}
        self.records=[]
        self.events={'Bid':[],'Ask':[]}
        self.block_until={'Bid':-np.inf,'Ask':-np.inf}
        self.spread=data.a-data.b
        self.position=np.divide(data.a+data.b-2*data.p,self.spread,
                                out=np.full(data.n,np.nan),where=data.valid['position'])
        self.qstate=np.where(data.valid['queue'],np.where(data.qi>.2,1,np.where(data.qi<-.2,-1,0)),np.nan)
        for name,groups in [('volume_depth',('volume','queue')),('price_ofi',('price','ofi')),('full',('price','ofi','volume'))]:
            valid=np.logical_and.reduce([data.valid[g] for g in groups])
            edge=np.logical_and.reduce([data.edge[g] for g in groups])
            segment=np.full(data.n,-1,dtype=int)
            for i in range(data.n):
                if valid[i]:segment[i]=segment[i-1] if edge[i] else i
            data.valid[name]=valid;data.edge[name]=edge;data.segment[name]=segment

    def win(self,i,h,group):
        w=self.d.window(i,h,group)
        if w is None:return None,'INVALID_INPUT'
        s,c,weights=w
        if c<=0:return None,'NO_COVERAGE'
        if c/h<self.cfg.minimum_coverage-1e-10:return None,'WARMUP'
        return (s,c,weights),'ok'

    def put(self,r,name,value,reason='ZERO_DENOM'):
        if name not in self.by_name:raise KeyError('Undeclared output '+name)
        value=float(value)
        if np.isinf(value):raise ValueError('Infinite feature '+name)
        r[name]=value;r[name+'__status']='ok' if np.isfinite(value) else reason

    def ratio(self,a,b):
        return a/b if np.isfinite(a) and np.isfinite(b) and b>0 else np.nan

    def age(self,i,group,event):
        d=self.d
        if not d.valid[group][i]:return np.nan,np.nan,np.nan
        start=d.segment[group][i]
        where=np.flatnonzero(event[start+1:i+1])
        if len(where):
            age=d.times[i]-d.times[start+1+where[-1]]
            return age,age,0.
        return np.nan,d.times[i]-d.times[start],1.

    def age_outputs(self,r,prefix,values):
        age,bound,censored=values
        self.put(r,prefix,age,'LEFT_CENSORED' if censored==1 else 'INVALID_INPUT')
        self.put(r,prefix+'LowerBound',bound,'INVALID_INPUT')
        self.put(r,prefix+'LeftCensored',censored,'INVALID_INPUT')

    def at(self,i):
        while len(self.records)<=i:
            j=len(self.records)
            r=self.d._base_features(self.d.times[j])
            self.records.append(r)
            self.extend(j,r)
        return self.records[i]

    def extend(self,i,r):
        d=self.d;cfg=self.cfg
        # A reason is populated for every output, even if no formula is eligible.
        for f in self.catalog:
            name=f['factor'];r.setdefault(name,np.nan)
            valid=d.valid[f['dependency']][i]
            reason='INVALID_INPUT' if not valid else 'NO_HISTORY'
            if valid and f['history_seconds']:
                _,reason=self.win(i,f['history_seconds'],f['dependency'])
                if reason=='ok':reason='ZERO_DENOM'
            r[name+'__status']='ok' if np.isfinite(r[name]) else reason
        put=lambda name,value,reason='ZERO_DENOM':self.put(r,name,value,reason)
        if d.edge['ofi'][i]:put('F03_OFI_latest',d.ofi[i])
        if d.valid['book'][i]:put('A01_Spread',self.spread[i])
        if d.edge['book'][i]:put('A01_SpreadChange',self.spread[i]-self.spread[i-1])
        if d.valid['queue'][i]:
            put('A02_Depth',d.depth[i]);put('A05_QIState',self.qstate[i])
        change=lambda x:np.r_[False,np.diff(x)!=0] if len(x) else np.array([],bool)
        self.age_outputs(r,'A05_StateAge',self.age(i,'queue',change(self.qstate)&d.edge['queue']))
        for name,x,group in [('B07_BidAge',d.b,'book'),('B07_AskAge',d.a,'book'),('C07_PriceAge',d.p,'price')]:
            self.age_outputs(r,name,self.age(i,group,change(x)&d.edge[group]))
        self.age_outputs(r,'D04_VolumeAge',self.age(i,'volume',(d.dv>0)&d.edge['volume']))
        put('B07_AgeDifference',r['B07_BidAge']-r['B07_AskAge'],'LEFT_CENSORED')
        # Starting an event is causal; a side remains occupied until its fixed deadline.
        for side,q in [('Bid',d.qb),('Ask',d.qa)]:
            price=d.b if side=='Bid' else d.a
            if not d.edge['ofi'][i]:self.block_until[side]=-np.inf
            elif (d.times[i]>=self.block_until[side] and price[i]==price[i-1]
                  and q[i-1]-q[i]>0 and (q[i-1]-q[i])/q[i-1]>=.1-1e-12):
                self.events[side].append(i)
                self.block_until[side]=d.times[i]+2
        for h in cfg.history_seconds:
            suff=f'_h{h}s'
            windows={}
            for group in d.valid:
                w,reason=self.win(i,h,group)
                if w:windows[group]=w
                raw=d.window(i,h,group)
                put(f'R06_{group}_Coverage'+suff,raw[1] if raw else np.nan,'INVALID_INPUT')
                put(f'R06_{group}_Count'+suff,i-raw[0]+1 if raw else 0)
                put(f'R06_{group}_Warmup'+suff,float(w is None))
                extra={'CoverageRatio':raw[1]/h if raw else np.nan,'StartSeconds':d.times[raw[0]] if raw else np.nan,
                       'EndSeconds':d.times[i] if raw else np.nan,'IncrementCount':i-raw[0] if raw else 0,
                       'MaxInterval':float(np.max(raw[2])) if raw and len(raw[2]) else np.nan}
                nominal=int(np.searchsorted(d.times,max(0,d.times[i]-h),side='left'))
                extra['GapFlag']=float(not d.valid[group][nominal:i+1].all() or not d.edge[group][nominal+1:i+1].all())
                for stem,value in extra.items():put(f'R06_{group}_'+stem+suff,value,'NO_COVERAGE')

            def p(fid,stem,value,reason='ZERO_DENOM'):
                put(fid+'_'+stem+suff,value,reason)
            if 'book' in windows:
                s,c,w=windows['book'];sl=slice(s,i)
                p('A01','SpreadMean',weighted_mean(self.spread[sl],w))
                p('A01','SpreadMax',np.max(self.spread[s:i+1]))
                p('A01','WideSpreadShare',weighted_mean(self.spread[sl]>1,w))
                db=np.diff(d.b[s:i+1]);da=np.diff(d.a[s:i+1])
                for side,moves in [('Bid',db),('Ask',da)]:
                    p('B06',side+'UpRate',np.count_nonzero(moves>0)/c)
                    p('B06',side+'DownRate',np.count_nonzero(moves<0)/c)
                    p('B08',side+'RV',np.sqrt(np.dot(moves,moves)/c))
                p('B06','UpDifference',r['B06_BidUpRate'+suff]-r['B06_AskUpRate'+suff])
                p('B06','DownDifference',r['B06_AskDownRate'+suff]-r['B06_BidDownRate'+suff])
                p('B08','RVDifference',r['B08_BidRV'+suff]-r['B08_AskRV'+suff])
                count=np.count_nonzero((db!=0)|(da!=0))
                for stem,mask in [('CoShare',db*da>0),('OppositeShare',db*da<0),('SingleShare',(db==0)^(da==0))]:p('B08',stem,self.ratio(np.count_nonzero(mask),count))
            if 'queue' in windows:
                s,c,w=windows['queue'];sl=slice(s,i);depth=d.depth[sl]
                mean=weighted_mean(depth,w)
                p('A02','DepthMean',mean);p('A02','DepthCV',weighted_std(depth,w)/mean)
                p('A02','DepthLogRatio',math.log(d.depth[i]/mean))
                p('A04','QIDeviation',d.qi[i]-weighted_mean(d.qi[sl],w))
                p('A04','QISlope',slope(d.qi[sl],d.times[sl],w),'NOT_ENOUGH_EVENTS')
                p('A05','BuyShare',weighted_mean(self.qstate[sl]==1,w))
                p('A05','SellShare',weighted_mean(self.qstate[sl]==-1,w))
                p('A05','SwitchRate',np.count_nonzero(np.diff(self.qstate[s:i+1]))/c)
                for side,q in [('Bid',d.qb),('Ask',d.qa)]:
                    mean=weighted_mean(q[sl],w)
                    p('B01',side+'Mean',mean);p('B02',side+'CV',weighted_std(q[sl],w)/mean)
                    p('B03',side+'Slope',slope(np.log(q[sl]),d.times[sl],w),'NOT_ENOUGH_EVENTS')
                p('B02','CVDifference',r['B02_BidCV'+suff]-r['B02_AskCV'+suff])
                p('B03','SlopeDifference',r['B03_BidSlope'+suff]-r['B03_AskSlope'+suff],'NOT_ENOUGH_EVENTS')
            if 'position' in windows:
                s,c,w=windows['position'];mean=weighted_mean(self.position[s:i],w)
                p('A06','PositionMean',mean);p('A06','PositionAbsMean',weighted_mean(abs(self.position[s:i]),w))
                p('A06','PositionDeviation',self.position[i]-mean)
            if 'ofi' in windows:
                s,c,w=windows['ofi'];e=d.ofi[s+1:i+1];total=e.sum();absolute=abs(e).sum()
                p('F03','OFI',total);p('A07','OFIRate',total/(weighted_mean(d.depth[s:i],w)*c))
                p('A08','OFIPulse',self.ratio(np.max(abs(e)),absolute))
                for side,q,price in [('Bid',d.qb,d.b),('Ask',d.qa,d.a)]:
                    same=np.diff(price[s:i+1])==0;dq=np.diff(q[s:i+1]);den=c*weighted_mean(q[s:i],w)
                    p('B04',side+'Addition',np.sum(same*np.maximum(dq,0))/den)
                    p('B04',side+'SameShare',np.dot(w,same)/c);p('B04',side+'SameCount',np.count_nonzero(same))
                    p('B05',side+'Depletion',np.sum(same*np.maximum(-dq,0))/den)
                p('B04','AdditionDifference',r['B04_BidAddition'+suff]-r['B04_AskAddition'+suff])
                p('B05','DepletionDifference',r['B05_AskDepletion'+suff]-r['B05_BidDepletion'+suff])
            if 'price' in windows:
                s,c,w=windows['price'];price=d.p[s:i+1];moves=np.diff(price)
                up=np.dot(moves[moves>0],moves[moves>0]);down=np.dot(moves[moves<0],moves[moves<0])
                p('C01','PriceAV',abs(moves).sum()/c)
                p('C02','UpSquares',up);p('C02','DownSquares',down);p('C02','TotalSquares',up+down);p('C02','Asymmetry',self.ratio(up-down,up+down))
                high=price.max();low=price.min();p('C03','Range',high-low)
                p('C03','Location',self.ratio(2*price[-1]-high-low,high-low))
                p('C04','Drawdown',np.max(np.maximum.accumulate(price)-price))
                p('C04','Runup',np.max(price-np.minimum.accumulate(price)))
                nu=np.count_nonzero(moves>0);nd=np.count_nonzero(moves<0);nz=np.count_nonzero(moves==0)
                p('R06','price_MoveCount',nu+nd)
                for stem,value in [('MoveRate',(nu+nd)/c),('CountImbalance',self.ratio(nu-nd,nu+nd)),('ZeroCountShare',nz/len(moves)),('ZeroTimeShare',np.dot(w,moves==0)/c),('UpCount',nu),('DownCount',nd),('ZeroCount',nz)]:p('C05',stem,value)
                signs=np.sign(moves[moves!=0]);K=len(signs);p('C06','NonzeroCount',K)
                p('C06','ReversalShare',np.count_nonzero(np.diff(signs))/(K-1) if K>=2 else np.nan,'NOT_ENOUGH_EVENTS')
                if K:
                    run=1
                    for sign in signs[-2::-1]:
                        if sign!=signs[-1]:break
                        run+=1
                    p('C06','RunCount',run);p('C06','SignedRun',signs[-1]*run)
                else:
                    p('C06','RunCount',np.nan,'NOT_ENOUGH_EVENTS');p('C06','SignedRun',np.nan,'NOT_ENOUGH_EVENTS')
                mean=weighted_mean(price[:-1],w);sd=weighted_std(price[:-1],w)
                r['_mean'+suff]=mean
                p('M03','StandardDeviation',self.ratio(price[-1]-mean,sd),'ZERO_VARIANCE')
                beta=slope(price[:-1],d.times[s:i],w)
                p('M11','TrendSlope',beta,'NOT_ENOUGH_EVENTS')
                intercept=mean-beta*weighted_mean(d.times[s:i],w)
                p('M11','TrendResidual',price[-1]-(intercept+beta*d.times[i]),'NOT_ENOUGH_EVENTS')
            if 'volume' in windows:
                s,c,w=windows['volume'];v=d.dv[s+1:i+1];rates=v/w;mean=v.sum()/c;sd=weighted_std(rates,w)
                p('R06','volume_PositiveCount',np.count_nonzero(v>0))
                p('D01','VolumeTotal',v.sum());p('D02','RateStd',sd)
                p('D02','RateCV',self.ratio(sd,mean));p('D02','LatestRateRatio',self.ratio(rates[-1],mean))
                p('D04','ZeroVolumeTimeShare',np.dot(w,v==0)/c)
            if 'price_volume' in windows:
                s,c,w=windows['price_volume'];v=d.dv[s+1:i+1];moves=np.diff(d.p[s:i+1]);total=v.sum()
                p('D04','UnchangedPriceVolumeShare',self.ratio(np.dot(v,moves==0),total))
                p('D08','MovePerVolume',self.ratio(abs(moves).sum(),total))
                p('M10','VWDeviation',d.p[i]-self.ratio(np.dot(v,d.p[s+1:i+1]),total))
                p('D09','SyncPairCount',len(v))
                for stem,x in [('SignedVolumeCorr',moves/w),('AbsoluteVolumeCorr',abs(moves)/w)]:
                    value,reason=weighted_corr(x,v/w,w);p('D09',stem,value,reason)
            if 'volume_depth' in windows:
                s,c,w=windows['volume_depth'];v=d.dv[s+1:i+1];total=v.sum()
                p('D05','VolumeDepthRate',total/(c*weighted_mean(d.depth[s:i],w)))
                p('D06','PreQIVolume',self.ratio(np.dot(v,d.qi[s:i]),total))
            if 'price_ofi' in windows:
                s,c,w=windows['price_ofi'];unchanged=np.diff(d.p[s:i+1])==0
                p('D07','UnchangedPriceOFIRate',np.dot(d.ofi[s+1:i+1],unchanged)/(c*weighted_mean(d.depth[s:i],w)))
            self.derived(i,h,r)
        self.two_segments(i,r)
        self.opening(i,r)
        self.recovery(i,r)
        self.lagged_relation(i,r)
        if 5 in cfg.history_seconds and 10 in cfg.history_seconds:
            put('M06_MeanGap_h10s_short5s',r['M01_MADeviation_h10s']-r['M01_MADeviation_h5s'],'WARMUP')
            w5=d.window(i,5,'price');w10=d.window(i,10,'price')
            put('M06_SameWindow_h10s_short5s',float(w5[0]==w10[0]) if w5 and w10 else np.nan,'INVALID_INPUT')
        # Quality describes all legal observed segments, never imputes the gaps.
        for group in d.valid:
            valid=d.valid[group][:i+1];edges=d.edge[group][:i+1]
            coverage=float(np.nansum(d.dt[:i+1][edges]));seen=np.flatnonzero(valid)
            values={'SegmentAge':d.times[i]-d.times[d.segment[group][i]] if valid[i] else np.nan,
                    'OpeningCoverage':coverage,'OpeningCoverageRatio':coverage/d.times[i] if d.times[i]>0 else np.nan,
                    'StartDelay':d.times[seen[0]] if len(seen) else np.nan,
                    'GapCount':max(0,int(np.count_nonzero(valid & ~edges))-1)}
            for stem,value in values.items():put('R06_'+group+'_'+stem,value,'INVALID_INPUT')
        put('R06_Elapsed',d.times[i]);put('R06_LatestInterval',d.dt[i],'NO_HISTORY')
        put('R06_AsyncPriceVolume',float(d.dv[i]==0 and d.p[i]!=d.p[i-1]) if d.edge['price_volume'][i] else np.nan,'NO_HISTORY')
        for f in self.catalog:
            if f['kind']=='alias':
                r[f['factor']]=r.get(f['alias_of'],np.nan)
                r[f['factor']+'__status']=r.get(f['alias_of']+'__status','NO_HISTORY')

    def derived(self,i,h,r):
        d=self.d;suff=f'_h{h}s'
        current=r.get('M01_MADeviation'+suff,np.nan)
        if not np.isfinite(current):return
        put=lambda fid,stem,v,why='ZERO_DENOM':self.put(r,fid+'_'+stem+suff+'_g5s',v,why)
        values=np.array([row.get('M01_MADeviation'+suff,np.nan) for row in self.records])
        start=i
        while start>0 and np.isfinite(values[start-1]) and d.edge['price'][start]:start-=1
        state=np.where(values>.5,1,np.where(values<-.5,-1,0))
        changes=np.flatnonzero(np.diff(state[start:i+1])!=0)
        put('M07','MeanState',state[i])
        exact=d.times[i]-d.times[start+1+changes[-1]] if len(changes) else np.nan
        bound=exact if len(changes) else d.times[i]-d.times[start]
        put('M07','MeanStateAge',exact,'LEFT_CENSORED');put('M07','MeanStateAgeLowerBound',bound);put('M07','MeanStateAgeLeftCensored',float(not len(changes)))
        up=np.zeros(i+1,bool);down=up.copy();last=0
        for j in range(start,i+1):
            if state[j]!=0:
                up[j]=state[j]==1 and last==-1;down[j]=state[j]==-1 and last==1;last=state[j]
        where=np.flatnonzero(up|down)
        exact=d.times[i]-d.times[where[-1]] if len(where) else np.nan
        bound=exact if len(where) else d.times[i]-d.times[start]
        put('M08','CrossAge',exact,'LEFT_CENSORED');put('M08','CrossAgeLowerBound',bound);put('M08','CrossAgeLeftCensored',float(not len(where)))
        s=max(start,int(np.searchsorted(d.times,d.times[i]-5,side='left')));c=d.times[i]-d.times[s]
        self.put(r,'R06_DerivedCoverage'+suff+'_g5s',c)
        if c>=5*self.cfg.minimum_coverage-1e-10 and c>0:
            w=np.diff(d.times[s:i+1]);x=values[s:i]
            put('M07','AboveShare',weighted_mean(state[s:i]==1,w));put('M07','BelowShare',weighted_mean(state[s:i]==-1,w))
            put('M08','CrossUpRate',np.count_nonzero(up[s+1:i+1])/c);put('M08','CrossDownRate',np.count_nonzero(down[s+1:i+1])/c)
            put('M09','AbsDeviationMean',weighted_mean(abs(x),w));put('M09','DeviationStd',weighted_std(x,w))
            put('M09','MaxPositive',max(0,np.max(values[s:i+1])));put('M09','MinNegative',min(0,np.min(values[s:i+1])))
        else:
            for fid,stems in [('M07',['AboveShare','BelowShare']),('M08',['CrossUpRate','CrossDownRate']),('M09',['AbsDeviationMean','DeviationStd','MaxPositive','MinNegative'])]:
                for stem in stems:put(fid,stem,np.nan,'WARMUP')
        k=int(np.searchsorted(d.times,d.times[i]-2,side='right'))-1
        if k>=start and d.times[i]-2-d.times[k]<=self.cfg.maximum_label_error_seconds+1e-10:
            old=self.records[k];previous=old.get('M01_MADeviation'+suff,np.nan);dt=d.times[i]-d.times[k]
            for fid,stem,value in [('M04','DeviationSpeed',(current-previous)/dt),('M04','AbsDeviationSpeed',(abs(current)-abs(previous))/dt),('M05','MeanSpeed',(r.get('_mean'+suff,np.nan)-old.get('_mean'+suff,np.nan))/dt)]:self.put(r,fid+'_'+stem+suff+'_lag2s',value,'WARMUP')
        else:
            for fid,stem in [('M04','DeviationSpeed'),('M04','AbsDeviationSpeed'),('M05','MeanSpeed')]:self.put(r,fid+'_'+stem+suff+'_lag2s',np.nan,'NO_HISTORY')

    def two_segments(self,i,r):
        d=self.d
        for lag in (2,5):
            suffix=f'_lag{lag}s'
            for group in ('price','ofi','volume','queue'):
                recent,reason=self.win(i,lag,group)
                if not recent:continue
                k,c1,w1=recent;earlier,reason=self.win(k,lag,group)
                if not earlier:continue
                s,c0,w0=earlier
                if d.segment[group][i]!=d.segment[group][k]:continue
                def put(fid,stem,v):self.put(r,fid+'_'+stem+suffix,v)
                if group=='price':
                    put('C08','MomentumSpeedChange',(d.p[i]-d.p[k])/c1-(d.p[k]-d.p[s])/c0)
                    put('R05','RVChange',np.sqrt(np.sum(np.diff(d.p[k:i+1])**2)/c1)-np.sqrt(np.sum(np.diff(d.p[s:k+1])**2)/c0))
                elif group=='ofi':
                    v1=d.ofi[k+1:i+1].sum()/(weighted_mean(d.depth[k:i],w1)*c1)
                    v0=d.ofi[s+1:k+1].sum()/(weighted_mean(d.depth[s:k],w0)*c0)
                    put('A08','OFIRateChange',v1-v0)
                elif group=='volume':
                    v1=d.dv[k+1:i+1].sum()/c1;v0=d.dv[s+1:k+1].sum()/c0
                    put('D03','VolumeRateChange',v1-v0);put('D03','VolumeRateImbalance',self.ratio(v1-v0,v1+v0))
                else:put('R05','DepthChange',weighted_mean(d.depth[k:i],w1)-weighted_mean(d.depth[s:k],w0))

    def opening(self,i,r):
        d=self.d;put=lambda name,v,why='ZERO_DENOM':self.put(r,name,v,why)
        valid=d.valid['price'][:i+1];observed=d.p[:i+1][valid]
        if valid[i]:
            put('R01_OpeningMove',d.p[i]-observed[0]);high=observed.max();low=observed.min()
            for stem,value in [('DistanceHigh',d.p[i]-high),('DistanceLow',d.p[i]-low),('OpeningRange',high-low),('OpeningLocation',self.ratio(2*d.p[i]-high-low,high-low))]:put('R02_'+stem,value)
            prior=d.p[:i][valid[:i]]
            if len(prior):
                dh=d.p[i]-prior.max();dl=d.p[i]-prior.min()
                for stem,value in [('PriorHighDistance',dh),('PriorLowDistance',dl),('NewHigh',float(dh>0)),('NewLow',float(dl<0))]:put('R03_'+stem,value)
            edges=np.flatnonzero(d.edge['price'][:i+1]);weights=d.dt[edges]
            put('M02_OpeningMADeviation',d.p[i]-weighted_mean(d.p[edges-1],weights) if len(edges) else np.nan,'NO_COVERAGE')
        if d.valid['price_volume'][i]:
            edges=np.flatnonzero(d.edge['price_volume'][:i+1]);v=d.dv[edges]
            put('M10_OpeningVWDeviation',d.p[i]-self.ratio(np.dot(d.p[edges],v),v.sum()))
        states=[('Spread',self.spread,'book',0,('Difference','Percentile')),('Depth',d.depth,'queue',0,('LogRatio','Percentile'))]
        for h in self.cfg.history_seconds:
            for name,source,dep in [('VolumeRate','D01_VolumeRate','volume'),('PriceRV','C01_PriceRV','price')]:
                array=np.array([x.get(source+f'_h{h}s',np.nan) for x in self.records])
                states.append((name,array,dep,h,('Ratio','Percentile')))
        for stem,x,group,h,expressions in states:
            if not d.valid[group][i] or not np.isfinite(x[i]):continue
            edges=np.flatnonzero(d.edge[group][:i+1]);j=edges-1
            ok=np.isfinite(x[j]);j=j[ok];w=d.dt[edges[ok]]
            suffix=f'_h{h}s' if h else ''
            if not len(j):
                for expr in expressions:put('R04_'+stem+expr+suffix,np.nan,'NO_HISTORY')
                continue
            mean=weighted_mean(x[j],w)
            for expr in expressions:
                if expr=='Difference':value=x[i]-mean
                elif expr=='Ratio':value=self.ratio(x[i],mean)
                elif expr=='LogRatio':value=math.log(x[i]/mean) if mean>0 and x[i]>0 else np.nan
                else:value=weighted_mean((x[j]<x[i])+.5*(x[j]==x[i]),w)
                put('R04_'+stem+expr+suffix,value)

    def recovery(self,i,r):
        d=self.d;window,reason=self.win(i,10,'ofi')
        if window is None:
            for f in self.catalog:
                if f['family_id']=='B09':r[f['factor']+'__status']=reason
            return
        s,_,_=window
        for side,q,price in [('Bid',d.qb,d.b),('Ask',d.qa,d.a)]:
            ratios=[];durations=[];mature=exit_count=immature=gapped=0
            for k in self.events[side]:
                if not s<k<=i:continue
                if d.times[k]+2>d.times[i]+1e-10:immature+=1;continue
                end=int(np.searchsorted(d.times,d.times[k]+2,side='left'))
                if end>i:immature+=1;continue
                if d.times[end]-d.times[k]-2>self.cfg.maximum_label_error_seconds+1e-10 or d.segment['ofi'][end]>k-1:
                    gapped+=1;continue
                mature+=1
                if np.any(price[k-1:end+1]!=price[k-1]):exit_count+=1;continue
                drop=q[k-1]-q[k];recovery=min(1.,max(0.,np.max(q[k:end+1]-q[k]))/drop)
                ratios.append(recovery)
                success=np.flatnonzero(q[k:end+1]>=q[k-1])
                if len(success):durations.append(d.times[k+success[0]]-d.times[k])
            same=len(ratios);prefix='B09_'+side
            for stem,value in [('MatureCount',mature),('SameCount',same),('ExitCount',exit_count),('SuccessCount',len(durations)),('ImmatureCount',immature),('GappedCount',gapped),('UnrecoveredCount',same-len(durations))]:self.put(r,prefix+stem+'_h10s',value)
            for stem,value in [('Recovery',np.mean(ratios) if same>=3 else np.nan),('FullShare',np.mean(np.array(ratios)==1) if same>=3 else np.nan),('SuccessTime',np.mean(durations) if same>=3 and len(durations) else np.nan),('ExitShare',exit_count/mature if mature>=3 else np.nan)]:self.put(r,prefix+stem+'_h10s',value,'NOT_ENOUGH_EVENTS')
        for stem,b,a in [('RecoveryDifference','Recovery','Recovery'),('FullShareDifference','FullShare','FullShare'),('SuccessTimeDifference','SuccessTime','SuccessTime')]:self.put(r,'B09_'+stem+'_h10s',r['B09_Bid'+b+'_h10s']-r['B09_Ask'+a+'_h10s'],'NOT_ENOUGH_EVENTS')

    def lagged_relation(self,i,r):
        d=self.d;window,reason=self.win(i,10,'full');prefix='D09_';suffix='_h10s_lag2s'
        if window is None:
            for stem in ('LagOFICorr','LagPairCount','LagAlignmentMax'):r[prefix+stem+suffix+'__status']=reason
            return
        s,_,_=window;x=[];y=[];w=[];errors=[]
        for k in range(s+1,i+1):
            end=int(np.searchsorted(d.times,d.times[k]+2,side='left'))
            if end>i:continue
            error=d.times[end]-d.times[k]-2
            if error>self.cfg.maximum_label_error_seconds+1e-10 or d.segment['full'][end]>k-1:continue
            x.append(d.ofi[k]/((d.depth[k-1]+d.depth[k])/2*d.dt[k]))
            y.append((d.p[end]-d.p[k])/(d.times[end]-d.times[k]));w.append(d.dt[k]);errors.append(error)
        self.put(r,prefix+'LagPairCount'+suffix,len(x))
        self.put(r,prefix+'LagAlignmentMax'+suffix,max(errors) if errors else np.nan,'NOT_ENOUGH_EVENTS')
        value,why=weighted_corr(np.array(x),np.array(y),np.array(w))
        self.put(r,prefix+'LagOFICorr'+suffix,value,why)
