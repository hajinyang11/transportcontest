"""Annual forward validation, historical-level drift and weather factorial."""
from pathlib import Path
import sys,json,argparse,hashlib,time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from runtime_common import pd,np,HistGradientBoostingRegressor,threadpool_limits
from runtime_common import DISTRICTS
from runtime_common import scores,period
from runtime_common import shift_at_target,residual_scores

ROOT=HERE.parents[1]
OLD=HERE/'multi_region_wait/multiyear_trip_only'
OUT=HERE/'multi_region_wait/annual_alternative_weather';OUT.mkdir(exist_ok=True)
WEATHER=ROOT/'data/raw/openmeteo_previous_runs_2024_2025'
METHODS=['recent_table','uniform_no_weather','uniform_weather','recent_no_weather','recent_weather']
GRIDS=['59_126','60_127']
META=['row_id','year','request','board','decision','date','month','hour','weekday','district','dong','grid','purpose','wait',
      'forecast_wet','forecast_snow','rain_forecast','precipitation_mm_48h','temperature_c_48h','weather_code_48h']
SETTINGS={'max_iter':120,'learning_rate':.08,'min_samples_leaf':80,'l2_regularization':10,'early_stopping':False,'random_state':20261022}

def save(x,name):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def export(d,name):d.to_csv(OUT/name,index=False,encoding='utf-8-sig')

def prepare():
    source=[]
    for year in [2023,2024,2025]:
        d=pd.read_pickle(OLD/f'prepared_{year}.pkl')
        source.append(d[d.board.ge('2023-11-01')].copy())
    allcity=pd.concat(source,ignore_index=True)
    dates=pd.date_range('2023-11-01','2025-12-31')
    a=allcity.assign(board_date=allcity.board.dt.normalize(),over60=allcity.wait.gt(60).astype(int))
    daily=a.groupby(['board_date','district','hour']).agg(n=('wait','size'),total=('wait','sum'),over60=('over60','sum'))
    idx=pd.MultiIndex.from_product([dates,DISTRICTS,range(24)],names=['date','district','hour'])
    a=daily.reindex(idx,fill_value=0).to_numpy().reshape(len(dates),len(DISTRICTS),24,3)
    end=np.maximum(np.arange(len(dates))-1,0);context=pd.DataFrame(index=idx)
    def roll(v,window):
        c=np.concatenate([np.zeros((1,)+v.shape[1:]),np.cumsum(v,axis=0)],axis=0)
        return c[end]-c[np.maximum(end-window,0)]
    for window in [7,28,56]:
        for label,v in [('near_hour',roll(a+np.roll(a,1,axis=2)+np.roll(a,-1,axis=2),window)),
                        ('region',roll(a.sum(axis=2,keepdims=True),window)),('city',roll(a.sum(axis=(1,2),keepdims=True),window))]:
            n,total,high=v[...,0],v[...,1],v[...,2]
            mean=np.divide(total,n,out=np.full_like(total,np.nan),where=n>0)
            frac=np.divide(high,n,out=np.full_like(high,np.nan),where=n>0)
            for var,values in [('count',n),('mean',mean),('over60',frac)]:
                if label=='city' and var=='count':continue
                context[f'{label}_{window}d_{var}']=np.broadcast_to(values,a.shape[:-1]).reshape(-1)
    context=context.reset_index();context.to_pickle(OUT/'context.pkl')
    mapping=pd.read_csv(HERE/'improved_day_ahead/stable_seoul_grid_map.csv')
    mapping=mapping[mapping.grid.isin(GRIDS)][['district','canonical','grid']].drop_duplicates()
    assert not mapping[['district','canonical']].duplicated().any()
    d=allcity[allcity.year.isin([2024,2025])].copy()
    d=d.merge(mapping,on=['district','canonical'],how='inner',validate='many_to_one')
    keys=sorted((mapping.district+'|'+mapping.canonical).unique());ids={k:i for i,k in enumerate(keys)}
    assert len(keys)<255
    d['dong_id']=d.key.map(ids);d['grid_id']=pd.Categorical(d.grid,categories=GRIDS).codes
    d=d.merge(context,on=['date','district','hour'],validate='many_to_one')
    d['valid_kst']=d.request.dt.floor('h')+pd.Timedelta(hours=1)
    w=pd.read_pickle(WEATHER/'forecast_hours.pkl').rename(columns={'legacy_kma_grid':'grid'})
    d=d.merge(w[['grid','valid_kst','precipitation_mm_48h','temperature_c_48h','weather_code_48h','complete','lead_reference_kst']],
              on=['grid','valid_kst'],how='left',validate='many_to_one')
    d['weather_available']=d.complete.fillna(False).astype(bool)
    before=d.groupby('year').agg(candidate_n=('row_id','size'),available_n=('weather_available','sum')).reset_index()
    d=d[d.weather_available].copy()
    d['precip_log']=np.log1p(d.precipitation_mm_48h)
    d['forecast_wet']=d.precipitation_mm_48h.gt(0)
    d['forecast_snow']=d.weather_code_48h.isin([71,73,75,77,85,86])
    d['rain_forecast']=d.forecast_wet&~d.forecast_snow
    d['period']=period(d.hour)
    d=d.sort_values(['request','row_id']).reset_index(drop=True)
    assert (d.decision-d.lead_reference_kst).dt.total_seconds().ge(23*3600).all()
    assert not d[['year','row_id']].duplicated().any()
    d.to_pickle(OUT/'features.pkl')
    export(before,'weather_coverage_by_year.csv')
    export(d.groupby(['year','month']).wait.agg(n='size',mean='mean',median='median',p90=lambda s:s.quantile(.9)).reset_index(),'monthly_wait_levels.csv')
    check=[]
    for date,district,hour in [('2024-01-01','종로구',10),('2025-01-01','서대문구',15),('2025-07-01','종로구',16)]:
        day=pd.Timestamp(date);cut=day-pd.Timedelta(days=1)
        known=allcity[allcity.board.lt(cut)&allcity.board.ge(cut-pd.Timedelta(days=28))&allcity.district.eq(district)&allcity.hour.isin([(hour-1)%24,hour,(hour+1)%24])]
        row=context[context.date.eq(day)&context.district.eq(district)&context.hour.eq(hour)].iloc[0]
        assert len(known)==row.near_hour_28d_count and abs(known.wait.mean()-row.near_hour_28d_mean)<1e-7
        check.append({'date':day,'district':district,'hour':hour,'known_n':len(known),'mean':known.wait.mean()})
    with (OUT/'features.pkl').open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    save({'protocol_sha256':hashlib.sha256((HERE/'ANNUAL_ALTERNATIVE_WEATHER_PROTOCOL.md').read_bytes()).hexdigest(),
          'prepared_features_sha256':sha,'weather_manifest':json.loads((WEATHER/'download_manifest.json').read_text(encoding='utf-8')),
          'records_by_year':d.year.value_counts().sort_index().to_dict(),'dong_keys':keys,
          'continuous_context_direct_checks':check,'future_observed_weather_in_inputs':False},'preparation_audit.json')
    print('prepared annual cohort',d.year.value_counts().sort_index().to_dict(),flush=True)

def features(d,weather):
    cols=['hour_float','weekday','month_sin','month_cos','district_id','dong_id','grid_id']+[c for c in d if c.startswith(('near_hour_','region_','city_'))]
    cats=['district_id','dong_id','grid_id']
    if weather:cols+=['precip_log','temperature_c_48h','weather_code_48h'];cats+=['weather_code_48h']
    return cols,cats

def weights(train,cutoff,recent):
    if not recent:return np.ones(len(train))
    age=(cutoff-train.request).dt.total_seconds().to_numpy()/86400
    assert (age>0).all()
    return np.exp2(-age/180)

def select():
    d=pd.read_pickle(OUT/'features.pkl');rows=[];audit=[]
    for leaves in [15,31]:
        for month in [7,8]:
            start=pd.Timestamp(2024,month,1);cut=start-pd.Timedelta(days=1)
            train=d[d.year.eq(2024)&d.board.lt(cut)];target=d[d.year.eq(2024)&d.month.eq(month)]
            assert train.board.max()<cut<=target.decision.min()
            for weather in [False,True]:
                cols,cats=features(d,weather);w=weights(train,cut,True)
                model=HistGradientBoostingRegressor(loss='quantile',quantile=.5,categorical_features=cats,max_leaf_nodes=leaves,**SETTINGS)
                model.fit(train[cols],train.wait,sample_weight=w);pred=np.maximum(0,model.predict(target[cols]))
                rows.append({'leaves':leaves,'month':month,'weather':weather,'n':len(target),'absolute_error_sum':np.abs(pred-target.wait).sum()})
                audit.append({'leaves':leaves,'month':month,'weather':weather,'cutoff':cut,'latest_board':train.board.max(),
                              'first_decision':target.decision.min(),'last_target_board':target.board.max(),'train_n':len(train)})
                print('2024-only setting check',leaves,month,weather,flush=True)
    s=pd.DataFrame(rows);rank=s.groupby('leaves').agg(n=('n','sum'),absolute_error_sum=('absolute_error_sum','sum')).reset_index()
    rank['mae']=rank.absolute_error_sum/rank.n;rank=rank.sort_values(['mae','leaves'])
    chosen=int(rank.iloc[0].leaves)
    assert max(x['last_target_board'] for x in audit)<pd.Timestamp('2024-10-31')
    export(s,'selection_scores_detail.csv');export(rank,'selection_scores.csv');export(pd.DataFrame(audit),'selection_time_audit.csv')
    save({'selected_leaves':chosen,'config':{**SETTINGS,'max_leaf_nodes':chosen},'selection_year':2024,
          '2025_labels_used':False,'latest_selection_outcome':max(x['last_target_board'] for x in audit)},'selected_2024_model.json')
    print('selected from2024',chosen,rank.to_json(orient='records'),flush=True)

def table(train,target,cutoff):
    train=train[train.request.ge(cutoff-pd.Timedelta(days=90))]
    pred=target[META].copy()
    index=pd.MultiIndex.from_frame(target[['district','weekday','hour']]);bindex=pd.MultiIndex.from_frame(target[['weekday','hour']])
    counts=train.groupby(['district','weekday','hour']).size().reindex(index).fillna(0).to_numpy()
    for q,name in [(.5,'raw_median'),(.9,'raw_upper')]:
        v=train.groupby(['district','weekday','hour']).wait.quantile(q).reindex(index).to_numpy()
        fallback=train.groupby(['weekday','hour']).wait.quantile(q).reindex(bindex).fillna(train.wait.quantile(q)).to_numpy()
        pred[name]=np.maximum(0,np.where((counts<30)|np.isnan(v),fallback,v))
    return pred

def fit():
    d=pd.read_pickle(OUT/'features.pkl');choice=json.loads((OUT/'selected_2024_model.json').read_text(encoding='utf-8'))
    config=choice['config'];audit=[]
    declaration={'protocol_sha256':hashlib.sha256((HERE/'ANNUAL_ALTERNATIVE_WEATHER_PROTOCOL.md').read_bytes()).hexdigest(),
                 'feature_hash':json.loads((OUT/'preparation_audit.json').read_text(encoding='utf-8'))['prepared_features_sha256'],
                 'methods':METHODS,'config':config,'half_life_days':180,'selection':choice,
                 'features_no_weather':features(d,False)[0],'features_weather':features(d,True)[0],
                 'fit_months':'2024Nov-Dec warmup,2025Jan-Dec evaluation','outcome':'receipt-to-boarding completed-only',
                 'evaluation_status':'2025previouslyexplored; present selection and forward fits do not use future outcomes'}
    dp=OUT/'comparison_declaration.json'
    if dp.exists():assert json.loads(dp.read_text(encoding='utf-8'))==declaration
    save(declaration,dp.name)
    months=pd.date_range('2024-11-01','2025-12-01',freq='MS')
    for start in months:
        cut=start-pd.Timedelta(days=1);finish=start+pd.offsets.MonthBegin(1)
        train=d[d.board.lt(cut)];target=d[d.request.ge(start)&d.request.lt(finish)]
        assert train.year.ge(2024).all() and train.board.max()<cut<=target.decision.min()
        for method in METHODS:
            stem=f'{method}_{start:%Y%m}';path=OUT/f'pred_{stem}.pkl';ap=OUT/f'fit_{stem}.json'
            if path.exists() and ap.exists():audit.append(json.loads(ap.read_text(encoding='utf-8')));continue
            t=time.perf_counter();recent=method.startswith('recent_');weather=method.endswith('_weather') and not method.endswith('no_weather')
            w=weights(train,cut,recent)
            if method=='recent_table':pred=table(train,target,cut)
            else:
                cols,cats=features(d,weather);pred=target[META].copy()
                for q,col in [(.5,'raw_median'),(.9,'raw_upper')]:
                    model=HistGradientBoostingRegressor(loss='quantile',quantile=q,categorical_features=cats,**config)
                    model.fit(train[cols],train.wait,sample_weight=w);pred[col]=np.maximum(0,model.predict(target[cols]))
            pred['raw_upper']=np.maximum(pred.raw_median,pred.raw_upper);pred.to_pickle(path)
            used=train[train.request.ge(cut-pd.Timedelta(days=90))] if method=='recent_table' else train
            audit_weights=np.ones(len(used)) if method=='recent_table' else w
            row={'method':method,'month':f'{start:%Y-%m}','cutoff':str(cut),'latest_board':str(used.board.max()),
                 'first_decision':str(target.decision.min()),'train_n':len(used),'target_n':len(target),
                 'train_2024_n':int(used.year.eq(2024).sum()),'train_2025_n':int(used.year.eq(2025).sum()),
                 'weight_min':float(audit_weights.min()),'weight_max':float(audit_weights.max()),
                 'weighted_2024_share':float(audit_weights[used.year.eq(2024)].sum()/audit_weights.sum()),
                 'effective_sample_n':float(audit_weights.sum()**2/np.square(audit_weights).sum()),'seconds':time.perf_counter()-t}
            save(row,ap.name);audit.append(row)
            print('fit',f'{start:%Y-%m}',method,len(used),'seconds',round(row['seconds'],1),flush=True)
    for method in METHODS:
        h=pd.concat([pd.read_pickle(OUT/f'pred_{method}_{s:%Y%m}.pkl') for s in months],ignore_index=True)
        assert not h[['year','row_id']].duplicated().any()
        h.sort_values(['year','row_id']).reset_index(drop=True).to_pickle(OUT/f'history_{method}.pkl')
    export(pd.DataFrame(audit),'fit_audit.csv')

def calibrate(h):
    # Summarize numeric index groups once per day/window instead of repeatedly
    # copying all metadata columns for every district/period cell.
    h=h.copy();h['period']=period(h.hour);audits=[]
    request=h.request.astype('int64').to_numpy();board=h.board.astype('int64').to_numpy()
    decision=h.decision.astype('int64').to_numpy();date=h.date.astype('int64').to_numpy()
    dc=pd.Categorical(h.district,categories=DISTRICTS).codes
    periods=['07-09','10-14','15-17','18-21','22-06']
    pc=pd.Categorical(h.period,categories=periods).codes
    cell=dc.astype('int64')*5+pc
    rawpoint=h.raw_median.to_numpy();rawupper=h.raw_upper.to_numpy();y=h.wait.to_numpy()
    point=np.full(len(h),np.nan);upper=np.full(len(h),np.nan)
    nsday=86400*10**9
    def groups(positions,codes):
        order=np.argsort(codes[positions],kind='stable');pos=positions[order];values=codes[pos]
        cuts=np.flatnonzero(values[1:]!=values[:-1])+1
        return {int(codes[g[0]]):g for g in np.split(pos,cuts) if len(g)}
    def summaries(positions,source,quantile):
        residual=y-source
        def stat(pos):
            r=residual[pos]
            if quantile==.5:shift=float(np.median(r))
            else:
                rank=min(int(np.ceil((len(r)+1)*.9))-1,len(r)-1)
                shift=float(np.partition(r,rank)[rank])
            return (len(pos),len(np.unique(date[pos])),int(board[pos].max()),shift)
        return ({k:stat(v) for k,v in groups(positions,cell).items()},
                {k:stat(v) for k,v in groups(positions,pc).items()},stat(positions))
    for day in pd.date_range('2025-01-01','2025-12-31'):
        cut=day-pd.Timedelta(days=1);cutns=cut.value
        past=np.flatnonzero((board<cutns)&(request>=cutns-56*nsday))
        p28=past[request[past]>=cutns-28*nsday];now=np.flatnonzero(date==day.value)
        assert len(past) and board[past].max()<cutns<=decision[now].min()
        pools={'median':summaries(p28,rawpoint,.5),'upper':summaries(past,rawupper,.9)}
        for key,pos in groups(now,cell).items():
            district=DISTRICTS[key//5];p=periods[key%5]
            values={}
            for label in ['median','upper']:
                local,broad,allstats=pools[label];picked=local.get(key);level='district_period'
                if picked is None or picked[0]<100 or picked[1]<5:
                    picked=broad.get(key%5);level='period'
                    if picked is None or picked[0]<100 or picked[1]<5:picked=allstats;level='all'
                n,days,latest,shift=picked;values[label]=shift
                audits.append({'date':day,'cutoff':cut,'district':district,'period':p,'kind':label,'level':level,
                               'n':n,'dates':days,'latest_board':pd.Timestamp(latest),
                               'first_decision':pd.Timestamp(int(decision[pos].min())),'shift':shift})
            point[pos]=np.maximum(0,rawpoint[pos]+values['median'])
            upper[pos]=np.maximum(point[pos],rawupper[pos]+values['upper'])
    keep=h.year.eq(2025).to_numpy();out=h[keep].copy();out['point']=point[keep];out['upper']=upper[keep]
    assert np.isfinite(out[['point','upper']]).all().all()
    return out.sort_values('row_id').reset_index(drop=True),pd.DataFrame(audits)

def evaluate():
    e=None;audits=[]
    for method in METHODS:
        h=pd.read_pickle(OUT/f'history_{method}.pkl');pred,audit=calibrate(h);audit['method']=method;audits.append(audit)
        if e is None:e=pred[META+['period']].copy()
        assert np.array_equal(e.row_id,pred.row_id)
        for src in ['raw_median','raw_upper','point','upper']:e[method+'_'+src]=pred[src].to_numpy()
        print('calibrated',method,len(pred),flush=True)
    e['request_hour']=e.request.dt.floor('h')
    aws=pd.read_csv(HERE.parent/'calltaxi/aws/district_hour_weather.csv',usecols=['district','time','rain_next'],parse_dates=['time'])
    e=e.merge(aws.rename(columns={'time':'request_hour'}),on=['district','request_hour'],how='left',validate='many_to_one')
    e.to_pickle(OUT/'evaluation_predictions.pkl');export(pd.concat(audits,ignore_index=True),'calibration_audit.csv')
    summer=e.month.isin([7,8]);ae=e.hour.between(15,21);rain=e.rain_forecast
    panels={'all':np.ones(len(e),bool),'JulAug':summer,'JulAug_rain_forecast':summer&rain,
            'JulAug_rain_forecast_afternoon_evening':summer&rain&ae,'rain_forecast':rain,
            'afternoon_evening':ae,'rain_forecast_afternoon_evening':rain&ae,
            'snow_forecast':e.forecast_snow,'medical_explicit':e.purpose.str.contains('치료|재활',na=False),
            'JulAug_actual_district_rain_known':summer&e.rain_next.gt(0),
            'JulAug_actual_district_rain_known_afternoon_evening':summer&e.rain_next.gt(0)&ae}
    rows=[];counts=[];matched=[];shiftrows=[];subgroup=[]
    for panel,mask in panels.items():
        g=e[mask]
        counts.append({'panel':panel,'n':len(g),'days':g.date.nunique(),'wait_mean':g.wait.mean(),
                       'observed_weather_known_n':int(g.rain_next.notna().sum())})
        if not len(g):continue
        for method in METHODS:
            point=g[method+'_point'];upper=g[method+'_upper']
            rows.append({'panel':panel,'method':method,'rule':'raw',**scores(g.wait,g[method+'_raw_median'],g[method+'_raw_upper'])})
            rows.append({'panel':panel,'method':method,'rule':'forward90',**scores(g.wait,point,upper)})
            if len(g)>=50:
                shift=shift_at_target(g.wait,point,upper,.9);u=np.maximum(point,upper+shift)
                rows.append({'panel':panel,'method':method,'rule':'matched90',**scores(g.wait,point,u)})
                shiftrows.append({'panel':panel,'method':method,'shift':shift,'uses_evaluation_labels':True})
    for group in ['month','grid','district']:
        for value,g in e.groupby(group):
            for method in METHODS:
                subgroup.append({'group':group,'value':value,'method':method,'raw_mae':np.abs(g.wait-g[method+'_raw_median']).mean(),
                                 'raw_bias':(g[method+'_raw_median']-g.wait).mean(),**scores(g.wait,g[method+'_point'],g[method+'_upper'])})
    s=pd.DataFrame(rows);export(s,'scores.csv');export(pd.DataFrame(counts),'panel_counts.csv')
    export(pd.DataFrame(shiftrows),'matched_shifts.csv');export(pd.DataFrame(subgroup),'subgroup_scores.csv')
    pairs=[('recent_no_weather','recent_weather'),('uniform_no_weather','uniform_weather'),
           ('uniform_no_weather','recent_no_weather'),('uniform_weather','recent_weather'),('recent_table','recent_no_weather')]
    deltas=[]
    for (panel,rule),g in s.groupby(['panel','rule']):
        for first,second in pairs:
            a=g[g.method.eq(first)].iloc[0];b=g[g.method.eq(second)].iloc[0]
            deltas.append({'panel':panel,'rule':rule,'first':first,'second':second,'n':int(a.n),
                           'mae_saved':a.mae-b.mae,'excess_saved':a.excess_mean-b.excess_mean,
                           'first_coverage':a.coverage,'second_coverage':b.coverage,'upper_saved':a.upper_mean-b.upper_mean,
                           'over20_rate_difference':b.over20_rate-a.over20_rate})
    export(pd.DataFrame(deltas),'paired_differences.csv')
    print(s[s.panel.isin(['all','JulAug','JulAug_rain_forecast','JulAug_rain_forecast_afternoon_evening'])&s.rule.eq('forward90')].to_json(orient='records',indent=2),flush=True)
    boot=[];rng=np.random.default_rng(20261022)
    for panel in ['all','JulAug','JulAug_rain_forecast','JulAug_rain_forecast_afternoon_evening']:
        g=e[panels[panel]].reset_index(drop=True)
        days=pd.date_range('2025-01-01','2025-12-31') if panel=='all' else pd.date_range('2025-07-01','2025-08-31')
        dayid=pd.Index(days).get_indexer(g.date);assert (dayid>=0).all()
        y=g.wait.to_numpy();day_n=np.bincount(dayid,minlength=len(days));arrays={};ranked={};daily={}
        for method in METHODS:
            point=g[method+'_point'].to_numpy();upper=g[method+'_upper'].to_numpy();r=residual_scores(y,point,upper);order=np.argsort(r)
            arrays[method]=(point,upper);ranked[method]=(order,r[order])
            daily[method]=np.stack([np.bincount(dayid,weights=v,minlength=len(days)) for v in
                                    [np.abs(y-point),np.maximum(upper-y,0),(y<=upper).astype(float)]])
        for rep in range(1000):
            starts=rng.integers(0,len(days)-6,size=int(np.ceil(len(days)/7)))
            dw=np.bincount((starts[:,None]+np.arange(7)).ravel()[:len(days)],minlength=len(days));w=dw[dayid]
            total=float(dw@day_n)
            if not total:continue
            ordinary={};matched_values={}
            for method in METHODS:
                point,upper=arrays[method];order,r=ranked[method];cw=np.cumsum(w[order]);shift=r[np.searchsorted(cw,.9*cw[-1],side='left')]
                assert np.isfinite(shift)
                ordinary[method]=daily[method]@dw/total
                matched_values[method]=float((np.maximum(np.maximum(point,upper+shift)-y,0)*w).sum()/total)
            for first,second in pairs:
                a,b=ordinary[first],ordinary[second]
                boot.append({'panel':panel,'rep':rep,'first':first,'second':second,'mae_saved':a[0]-b[0],
                             'forward_excess_saved':a[1]-b[1],'coverage_difference_pp':100*(b[2]-a[2]),
                             'matched_excess_saved':matched_values[first]-matched_values[second]})
        print('bootstrap',panel,len(g),flush=True)
    b=pd.DataFrame(boot);ci=[]
    for (panel,first,second),g in b.groupby(['panel','first','second']):
        for metric in ['mae_saved','forward_excess_saved','coverage_difference_pp','matched_excess_saved']:
            lo,hi=g[metric].quantile([.025,.975]);ci.append({'panel':panel,'first':first,'second':second,'metric':metric,'lo95':lo,'hi95':hi,'reps':len(g)})
    export(b,'bootstrap_draws.csv');export(pd.DataFrame(ci),'bootstrap_summary.csv')
    print('annual evaluation complete',len(e),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--select',action='store_true');p.add_argument('--fit',action='store_true');p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    with threadpool_limits(limits=4):
        if a.prepare:prepare()
        if a.select:select()
        if a.fit:fit()
        if a.evaluate:evaluate()
