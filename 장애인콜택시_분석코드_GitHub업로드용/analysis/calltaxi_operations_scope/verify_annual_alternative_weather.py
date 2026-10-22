"""Independent provenance, common-cohort, timing and metric recomputation."""
from pathlib import Path
import sys,json,hashlib
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from runtime_common import pd,np
from compare_annual_alternative_weather import OUT,OLD,WEATHER,METHODS,save

def main():
    prep=json.loads((OUT/'preparation_audit.json').read_text(encoding='utf-8'))
    dec=json.loads((OUT/'comparison_declaration.json').read_text(encoding='utf-8'))
    assert prep['protocol_sha256']==dec['protocol_sha256']==hashlib.sha256((HERE/'ANNUAL_ALTERNATIVE_WEATHER_PROTOCOL.md').read_bytes()).hexdigest()
    with (OUT/'features.pkl').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==dec['feature_hash']==prep['prepared_features_sha256']
    for source in prep['weather_manifest']['files']:
        with (WEATHER/source['json_file']).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==source['sha256']
    d=pd.read_pickle(OUT/'features.pkl')
    assert set(d.year)=={2024,2025} and not d[['year','row_id']].duplicated().any()
    assert d[['precipitation_mm_48h','temperature_c_48h','weather_code_48h']].notna().all().all()
    assert (d.decision-d.lead_reference_kst).dt.total_seconds().ge(23*3600).all()
    for cols in [dec['features_no_weather'],dec['features_weather']]:
        assert not set(cols)&{'wait','board','purpose','rain_next','year','request','forecast_wet','rain_forecast'}
    assert set(dec['features_weather'])-set(dec['features_no_weather'])=={'precip_log','temperature_c_48h','weather_code_48h'}
    assert np.allclose(d.precip_log,np.log1p(d.precipitation_mm_48h))
    origin_checks=[]
    for year in [2024,2025]:
        source=pd.read_pickle(OLD/f'prepared_{year}.pkl').set_index('row_id');g=d[d.year.eq(year)]
        source=source.loc[g.row_id]
        assert np.array_equal(source.request,g.request) and np.array_equal(source.board,g.board)
        assert np.array_equal(source.wait,g.wait)
        origin_checks.append({'year':year,'records':len(g),'original_trip_values_match':True})
    selection=json.loads((OUT/'selected_2024_model.json').read_text(encoding='utf-8'))
    assert selection['selection_year']==2024 and selection['2025_labels_used'] is False
    sel=pd.read_csv(OUT/'selection_time_audit.csv',parse_dates=['cutoff','latest_board','first_decision','last_target_board'])
    assert len(sel)==8 and (sel.latest_board<sel.cutoff).all() and (sel.cutoff<=sel.first_decision).all()
    assert sel.last_target_board.max()<pd.Timestamp('2024-10-31')
    fits=pd.read_csv(OUT/'fit_audit.csv',parse_dates=['cutoff','latest_board','first_decision'])
    assert len(fits)==70 and (fits.latest_board<fits.cutoff).all() and (fits.cutoff<=fits.first_decision).all()
    fitchecks=[]
    for stamp in ['2025-01','2025-07','2025-12']:
        for method in ['uniform_no_weather','recent_no_weather']:
            row=fits[fits['month'].eq(stamp)&fits.method.eq(method)].iloc[0]
            train=d[d.board.lt(row.cutoff)]
            assert len(train)==row.train_n
            assert train.year.eq(2024).sum()==row.train_2024_n and train.year.eq(2025).sum()==row.train_2025_n
            age=(row.cutoff-train.request).dt.total_seconds().to_numpy()/86400
            w=np.exp2(-age/180) if method.startswith('recent') else np.ones(len(train))
            share=w[train.year.eq(2024)].sum()/w.sum()
            assert abs(share-row.weighted_2024_share)<1e-10
            fitchecks.append({'month':stamp,'method':method,'old_2024_weight_share':share})
    # January context must contain known December2024 calls, not reset to empty.
    source=pd.concat([pd.read_pickle(OLD/f'prepared_{y}.pkl') for y in [2023,2024,2025]],ignore_index=True)
    context=pd.read_pickle(OUT/'context.pkl');contextchecks=[]
    for date,district,hour in [('2025-01-01','서대문구',15),('2025-07-01','종로구',16),('2025-08-01','마포구',10)]:
        day=pd.Timestamp(date);cut=day-pd.Timedelta(days=1)
        known=source[source.board.lt(cut)&source.board.ge(cut-pd.Timedelta(days=28))&source.district.eq(district)&source.hour.isin([(hour-1)%24,hour,(hour+1)%24])]
        row=context[context.date.eq(day)&context.district.eq(district)&context.hour.eq(hour)].iloc[0]
        assert len(known)==row.near_hour_28d_count and abs(known.wait.mean()-row.near_hour_28d_mean)<1e-7
        contextchecks.append({'date':str(day),'known_records':len(known)})
    del source,context
    e=pd.read_pickle(OUT/'evaluation_predictions.pkl')
    target=d[d.year.eq(2025)].sort_values('row_id');assert len(e)==len(target)==113776
    assert np.array_equal(e.row_id,target.row_id) and np.array_equal(e.wait,target.wait)
    assert set(e.month)==set(range(1,13)) and e.date.nunique()==365
    cal=pd.read_csv(OUT/'calibration_audit.csv',parse_dates=['date','cutoff','latest_board','first_decision'])
    assert (cal.latest_board<cal.cutoff).all() and (cal.cutoff<=cal.first_decision).all()
    directcal=[]
    for method in METHODS:
        h=pd.read_pickle(OUT/f'history_{method}.pkl')
        assert set(e.row_id)==set(h[h.year.eq(2025)].row_id)
        for day in ['2025-01-01','2025-07-01','2025-08-01']:
            for kind,sourcecol,window in [('median','raw_median',28),('upper','raw_upper',56)]:
                row=cal[cal.method.eq(method)&cal.date.eq(day)&cal.period.eq('15-17')&cal.kind.eq(kind)].iloc[0]
                pool=h[h.board.lt(row.cutoff)&h.request.ge(row.cutoff-pd.Timedelta(days=window))]
                if row.level!='all':pool=pool[pool.hour.between(15,17)]
                if row.level=='district_period':pool=pool[pool.district.eq(row.district)]
                residual=np.sort((pool.wait-pool[sourcecol]).to_numpy())
                value=np.median(residual) if kind=='median' else residual[min(int(np.ceil((len(residual)+1)*.9))-1,len(residual)-1)]
                assert len(pool)==row.n and abs(value-row['shift'])<1e-8
                directcal.append({'method':method,'date':day,'kind':kind,'n':len(pool)})
    summer=e.month.isin([7,8]);ae=e.hour.between(15,21);rain=e.rain_forecast
    masks={'all':np.ones(len(e),bool),'JulAug':summer,'JulAug_rain_forecast':summer&rain,
           'JulAug_rain_forecast_afternoon_evening':summer&rain&ae,'rain_forecast':rain,
           'afternoon_evening':ae,'rain_forecast_afternoon_evening':rain&ae,'snow_forecast':e.forecast_snow,
           'medical_explicit':e.purpose.str.contains('치료|재활',na=False),
           'JulAug_actual_district_rain_known':summer&e.rain_next.gt(0),
           'JulAug_actual_district_rain_known_afternoon_evening':summer&e.rain_next.gt(0)&ae}
    scores=pd.read_csv(OUT/'scores.csv');shifts=pd.read_csv(OUT/'matched_shifts.csv')
    for row in scores.itertuples():
        g=e[masks[row.panel]];y=g.wait.to_numpy()
        point=g[row.method+('_raw_median' if row.rule=='raw' else '_point')].to_numpy()
        upper=g[row.method+('_raw_upper' if row.rule=='raw' else '_upper')].to_numpy()
        if row.rule=='matched90':
            shift=shifts[shifts.panel.eq(row.panel)&shifts.method.eq(row.method)].iloc[0]['shift']
            upper=np.maximum(point,upper+shift);assert (y<=upper).sum()>=np.ceil(.9*len(g))
        assert len(g)==row.n
        values={'mae':np.mean(np.abs(point-y)),'point_bias':np.mean(point-y),'coverage':np.mean(y<=upper),
                'upper_mean':upper.mean(),'excess_mean':np.maximum(upper-y,0).mean(),'over20_rate':np.mean(y-upper>20)}
        for key,value in values.items():assert abs(value-getattr(row,key))<1e-8,(row.panel,row.method,row.rule,key)
    boot=pd.read_csv(OUT/'bootstrap_draws.csv');cis=pd.read_csv(OUT/'bootstrap_summary.csv');assert len(cis)==80
    for row in cis.itertuples():
        part=boot[boot.panel.eq(row.panel)&boot['first'].eq(row.first)&boot['second'].eq(row.second)]
        lo,hi=part[row.metric].quantile([.025,.975]);assert abs(lo-row.lo95)<1e-8 and abs(hi-row.hi95)<1e-8
        assert len(part)==row.reps>=900
    save({'passed':True,'source_trip_checks':origin_checks,'selection_audits':len(sel),'future_2025_used_for_selection':False,
          'model_fit_cutoffs':len(fits),'sample_learning_weight_checks':fitchecks,'continuous_context_checks':contextchecks,
          'common_evaluation_records':len(e),'evaluation_days':365,'evaluation_months':12,
          'all_calibration_cutoffs':len(cal),'direct_calibration_recomputations':len(directcal),
          'score_rows_recomputed':len(scores),'bootstrap_CI_rows_recomputed':len(cis),
          'future_observed_weather_used_as_input':False},'verification.json')
    print((OUT/'verification.json').read_text(encoding='utf-8'))

if __name__=='__main__':main()
