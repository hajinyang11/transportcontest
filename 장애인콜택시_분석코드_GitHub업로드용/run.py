"""Entry point for source-only competition reproduction. Python 3.12."""
from pathlib import Path
import sys,argparse,json,hashlib,shutil,time

ROOT=Path(__file__).resolve().parent
CODE=ROOT/'analysis/calltaxi_operations_scope'
sys.path.insert(0,str(CODE))


def folders():
    for rel in ['multi_region_wait/multiyear_trip_only','multi_region_wait/annual_alternative_weather/boarding_intervals/short_interval_scope']:
        (CODE/rel).mkdir(parents=True,exist_ok=True)


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(name,value):
    target=ROOT/'validation'/name;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


def check(data_root):
    manifest=json.loads((ROOT/'data_manifest.json').read_text(encoding='utf-8'))
    files=[]
    for expected in manifest['trip_files']:
        matches=list((data_root/'data/raw').glob(expected['target_glob']))
        if len(matches)!=1:raise FileNotFoundError(f"Expected one {expected['target_glob']}; found {len(matches)}")
        matched=digest(matches[0])==expected['sha256']
        if not matched:raise ValueError(f"Source checksum differs: {matches[0].name}")
        files.append({'year':expected['year'],'sha256_match':matched})
    for site in ['ihwa','sinchon']:
        for year in [2024,2025,2026]:
            p=data_root/f'data/raw/openmeteo_previous_runs_2024_2025/jma_msm_{site}_{year}_day2.json'
            if not p.exists():raise FileNotFoundError(p)
    aws=data_root/'analysis/calltaxi/aws/district_hour_weather.csv'
    if not aws.exists():raise FileNotFoundError(aws)
    import pandas as pd
    assert set(manifest['observed_weather_required_columns'])<=set(pd.read_csv(aws,nrows=0).columns)
    result={'passed':True,'trip_source_hash_checks':files,'forecast_json_files':6,'observed_weather_schema_checked':True}
    save('input_check.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


def verify_reference(data_root):
    """Execute the released scope analysis against local reference predictions."""
    folders()
    import pandas as pd
    import numpy as np
    import analyze_short_interval_scope as scope
    import verify_short_interval_scope as verify
    original=data_root/'analysis/calltaxi_operations_scope'
    source=original/'multi_region_wait/annual_alternative_weather/boarding_intervals'
    out=ROOT/'validation/recomputed_scope';out.mkdir(parents=True,exist_ok=True)
    scope.HERE=original;scope.SOURCE=source;scope.OUT=out
    scope.main()
    verify.HERE=original;verify.SOURCE=source;verify.OUT=out;verify.main()
    actual=pd.read_csv(out/'gate_summary.csv')
    expected=pd.read_csv(ROOT/'results/expected/gate_summary.csv')
    key=['cap','phase','method']
    both=actual.merge(expected,on=key,suffixes=('_actual','_expected'),validate='one_to_one')
    assert len(both)==12
    for metric in ['n','denominator','share','coverage','width_mean','city_denominator']:
        np.testing.assert_allclose(both[metric+'_actual'],both[metric+'_expected'],rtol=1e-10,atol=1e-10)
    row=actual[(actual.cap==45)&actual.phase.eq('H2')&actual.method.eq('recent_weather')].iloc[0]
    result={'passed':True,'mode':'published source re-executed on precomputed local intervals',
            'full_training_rerun':False,'gate_rows_matched':12,
            'short45_H2':{k:float(row[k]) for k in ['n','denominator','coverage','share','width_mean']}}
    save('reference_reproduction.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


def smoke_fit(data_root):
    """Fresh fit of three interval quantiles for Jan2025; check original outputs."""
    folders()
    import pandas as pd
    import numpy as np
    import compare_annual_alternative_weather as annual
    source=data_root/'analysis/calltaxi_operations_scope/multi_region_wait/annual_alternative_weather'
    d=pd.read_pickle(source/'features.pkl')
    cut=pd.Timestamp('2024-12-31');train=d[d.board.lt(cut)];target=d[d.year.eq(2025)&d.month.eq(1)]
    expected=pd.read_pickle(source/'boarding_intervals/raw_recent_weather_202501.pkl')
    expected=target[['row_id']].merge(expected,on='row_id',validate='one_to_one')
    config=json.loads((source/'selected_2024_model.json').read_text(encoding='utf-8'))['config']
    cols,cats=annual.features(d,True);weight=annual.weights(train,cut,True);pred={}
    started=time.perf_counter()
    with annual.threadpool_limits(limits=4):
        for q,name in [(.05,'q05'),(.1,'q10'),(.95,'q95')]:
            model=annual.HistGradientBoostingRegressor(loss='quantile',quantile=q,categorical_features=cats,**config)
            model.fit(train[cols],train.wait,sample_weight=weight)
            pred[name]=np.maximum(0,model.predict(target[cols]))
    pred['q10']=np.minimum(pred['q10'],expected.raw_median)
    pred['q05']=np.minimum(pred['q05'],pred['q10'])
    pred['q95']=np.maximum(pred['q95'],expected.q90)
    errors={}
    for name in pred:
        errors[name]=float(np.max(np.abs(np.asarray(pred[name])-expected[name].to_numpy())))
        np.testing.assert_allclose(pred[name],expected[name],rtol=1e-9,atol=1e-9)
    result={'passed':True,'fresh_training_month':'2025-01','fresh_quantiles':[.05,.1,.95],
        'train_n':len(train),'target_n':len(target),'max_absolute_difference':errors,
        'seconds':round(time.perf_counter()-started,2),'full_training_rerun':False}
    save('fresh_fit_check.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


def reproduce(data_root):
    """Raw trip files -> monthly models -> intervals -> scope tables -> checks."""
    check(data_root);folders()
    import prepare_multiyear_trips as trips
    import fetch_alternative_forecasts as weather
    import compare_annual_alternative_weather as annual
    import compare_boarding_intervals as intervals
    import verify_boarding_intervals as verify_intervals
    import analyze_short_interval_scope as scope
    import verify_short_interval_scope as verify_scope
    from runtime_common import threadpool_limits
    trips.RAW=data_root/'data/raw'
    weather.OUT=ROOT/'data/raw/openmeteo_previous_runs_2024_2025';weather.OUT.mkdir(parents=True,exist_ok=True)
    for site in ['ihwa','sinchon']:
        for year in [2024,2025,2026]:
            name=f'jma_msm_{site}_{year}_day2.json'
            source=data_root/'data/raw/openmeteo_previous_runs_2024_2025'/name
            target=weather.OUT/name
            if source.resolve()!=target.resolve():shutil.copy2(source,target)
    aws=ROOT/'analysis/calltaxi/aws/district_hour_weather.csv';aws.parent.mkdir(parents=True,exist_ok=True)
    source=data_root/'analysis/calltaxi/aws/district_hour_weather.csv'
    if source.resolve()!=aws.resolve():shutil.copy2(source,aws)
    with threadpool_limits(limits=4):
        for year in [2023,2024,2025]:
            if not (trips.OUT/f'prepared_{year}.pkl').exists():trips.prepare_year(year)
        weather.main()
        annual.prepare();annual.select();annual.fit();annual.evaluate()
        intervals.fit();intervals.evaluate();verify_intervals.main()
        scope.main();verify_scope.main()
    print('Full pipeline completed. Outputs: analysis/calltaxi_operations_scope/multi_region_wait/')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['check','verify-reference','smoke-fit','reproduce'])
    p.add_argument('--data-root',type=Path,required=True,help='Root containing data/raw and analysis/calltaxi/aws; see README')
    args=p.parse_args();root=args.data_root.resolve()
    {'check':check,'verify-reference':verify_reference,'smoke-fit':smoke_fit,'reproduce':reproduce}[args.command](root)


if __name__=='__main__':main()
