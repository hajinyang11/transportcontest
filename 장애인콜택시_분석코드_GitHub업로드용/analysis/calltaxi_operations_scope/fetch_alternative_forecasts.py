"""Download public archived JMA MSM forecasts; preserve source responses.

Previous Runs day2 is a fixed 48h offset, not observed weather or day0 analysis.
No API key, login, new account, or paid endpoint is used.
"""
from pathlib import Path
import sys,json,hashlib,urllib.request,urllib.parse,csv
from concurrent.futures import ThreadPoolExecutor
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from runtime_common import pd,np

ROOT=HERE.parents[1]
OUT=ROOT/'data/raw/openmeteo_previous_runs_2024_2025'
MODEL='jma_msm'
VARIABLES=['precipitation_previous_day2','temperature_2m_previous_day2','weather_code_previous_day2']
SITES=[{'site':'ihwa','label':'서울 종로구 이화동','legacy_kma_grid':'60_127',
        'latitude':37.5742138888888,'longitude':127.006397222222},
       {'site':'sinchon','label':'서울 서대문구 신촌동','legacy_kma_grid':'59_126',
        'latitude':37.5567388888888,'longitude':126.945586111111}]
SOURCES={
 'previous_runs':'https://open-meteo.com/en/docs/previous-runs-api',
 'jma_models':'https://open-meteo.com/en/docs/jma-api',
 'terms':'https://open-meteo.com/en/terms',
 'coordinate_archive':'transport_contest/analysis/calltaxi_operations_scope/improved_day_ahead/kma_grid_2025q1.zip',
 'coordinate_member':'동네예보지점좌표(위경도)_202504.xlsx',
 'coordinate_columns':'경도(초/100), 위도(초/100); exact Seoul district+dong rows',
}

def fetch(job):
    site,year=job
    start_date=f'{year}-01-01'
    end_date=f'{year}-12-31' if year in [2024,2025] else f'{year}-01-01'
    params={'latitude':str(site['latitude']),'longitude':str(site['longitude']),
            'start_date':start_date,'end_date':end_date,'hourly':','.join(VARIABLES),
            'models':MODEL,'timezone':'Asia/Seoul'}
    url='https://previous-runs-api.open-meteo.com/v1/forecast?'+urllib.parse.urlencode(params)
    path=OUT/f'{MODEL}_{site["site"]}_{year}_day2.json'
    if path.exists():body=path.read_bytes();status='cached public response'
    else:
        with urllib.request.urlopen(url,timeout=45) as response:
            assert response.status==200
            body=response.read();status=response.status
        document=json.loads(body)
        assert not document.get('error'),document
        path.write_bytes(body)
    d=json.loads(body);h=d['hourly']
    frame=pd.DataFrame(h)
    frame['valid_kst']=pd.to_datetime(frame.pop('time'))
    expected=pd.date_range(start_date,pd.Timestamp(end_date)+pd.Timedelta(days=1),freq='h',inclusive='left')
    assert len(frame)==len(expected) and np.array_equal(frame.valid_kst,expected)
    assert frame.valid_kst.is_unique
    for name in VARIABLES:frame[name]=pd.to_numeric(frame[name],errors='raise')
    assert frame.precipitation_previous_day2.dropna().ge(0).all()
    frame=frame.rename(columns={'precipitation_previous_day2':'precipitation_mm_48h',
                               'temperature_2m_previous_day2':'temperature_c_48h',
                               'weather_code_previous_day2':'weather_code_48h'})
    frame['complete']=frame[['precipitation_mm_48h','temperature_c_48h','weather_code_48h']].notna().all(axis=1)
    frame['site']=site['site'];frame['legacy_kma_grid']=site['legacy_kma_grid'];frame['model']=MODEL
    frame['forecast_lead_hours']=48
    # This is an offset reference from API semantics, not a separately supplied
    # original model-run initialisation or publication/receipt timestamp.
    frame['lead_reference_kst']=frame.valid_kst-pd.Timedelta(hours=48)
    frame['requested_latitude']=site['latitude'];frame['requested_longitude']=site['longitude']
    frame['returned_model_latitude']=d['latitude'];frame['returned_model_longitude']=d['longitude']
    frame.to_csv(path.with_suffix('.csv'),index=False,encoding='utf-8-sig')
    complete=frame.complete
    gaps=frame[~complete][['valid_kst','precipitation_mm_48h','temperature_c_48h','weather_code_48h']].copy()
    gaps['site']=site['site'];gaps['year']=year
    meta={**site,'year':year,'start_date':start_date,'end_date':end_date,'boundary_only':year==2026,
          'model':MODEL,'url':url,'status':status,'json_file':path.name,
          'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body),'hours':len(frame),
          'complete_hours':int(complete.sum()),'missing_hours':int((~complete).sum()),
          'nonmissing_by_variable':{c:int(frame[c].notna().sum()) for c in ['precipitation_mm_48h','temperature_c_48h','weather_code_48h']},
          'returned_model_latitude':d['latitude'],'returned_model_longitude':d['longitude'],
          'api_hourly_units':d['hourly_units'],'timezone':d['timezone'],'utc_offset_seconds':d['utc_offset_seconds']}
    assert d['utc_offset_seconds']==32400
    print('downloaded',site['site'],year,'hours',len(frame),'complete',int(complete.sum()),flush=True)
    return frame,gaps,meta

def main():
    OUT.mkdir(exist_ok=True)
    # Jan1 2026 closes the preceding-hour rainfall interval for late calls on
    # Dec31 2025. It is weather context only, not a new2026trip evaluation.
    jobs=[(site,year) for site in SITES for year in [2024,2025,2026]]
    with ThreadPoolExecutor(max_workers=2) as ex:results=list(ex.map(fetch,jobs))
    allhours=pd.concat([r[0] for r in results],ignore_index=True)
    assert not allhours[['site','valid_kst']].duplicated().any()
    gaps=pd.concat([r[1] for r in results],ignore_index=True)
    allhours.to_pickle(OUT/'forecast_hours.pkl')
    allhours.to_csv(OUT/'forecast_hours.csv',index=False,encoding='utf-8-sig')
    gaps.to_csv(OUT/'missing_hours.csv',index=False,encoding='utf-8-sig')
    # Geographic variability is checked; distinct returned cells do not prove
    # neighborhood forecast skill.
    pivot=allhours.pivot(index='valid_kst',columns='site',values='precipitation_mm_48h')
    both=pivot.dropna();different=int((both.ihwa-both.sinchon).abs().gt(1e-8).sum())
    manifest={'retrieved_on_client_date':'2026-10-22','source_urls':SOURCES,'files':[r[2] for r in results],
              'model':MODEL,'forecast_lead_hours':48,'total_site_hours':len(allhours),
              'complete_site_hours':int(allhours.complete.sum()),'missing_site_hours':len(gaps),
              'same_valid_hours_both_sites':len(both),'different_precipitation_hours':different,
              'site_coordinates_from_official_existing_archive':True,
              'historical_forecast_api_stitched_day0_used':False,'observed_or_reanalysis_weather_used':False,
              'missing_policy':'preserved as NaN; not zero-filled or replaced from another model or KMA',
              'precipitation_probability_available_in_this_download':False,
              'weather_code_is_KMA_PTY':False,
              'original_run_initialisation_and_publication_timestamps_supplied':False,
              'lead_reference_definition':'valid_kst-48h by Previous Runs day2 semantics; not a supplied run timestamp',
              'next_join':'valid hour H describes preceding hour: call in H-1..H joins H; 48h offset gives about23h margin before exact call-24h',
              'comparison':'same source/model/offset in both2024and2025; compare matched no-weather vsweather cohort; no2024OpenMeteo/2025KMA splice',
              'KMA_direct_attempt':'public sessionCheck returned code99, public page download action requires login; did not request protected data-generation endpoint or create account'}
    (OUT/'download_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['total_site_hours','complete_site_hours','missing_site_hours','same_valid_hours_both_sites','different_precipitation_hours']},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
