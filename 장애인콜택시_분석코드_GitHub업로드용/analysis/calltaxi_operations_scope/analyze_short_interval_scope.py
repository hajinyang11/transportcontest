"""Prediction-width gates and temporally confirmed district/hour scope."""
from pathlib import Path
import sys, json, hashlib
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / '.analysis_deps'))
import numpy as np
import pandas as pd

SOURCE = HERE / 'multi_region_wait/annual_alternative_weather/boarding_intervals'
OUT = SOURCE / 'short_interval_scope'
OUT.mkdir(exist_ok=True)
METHODS = ['recent_table', 'recent_weather']
CUTOFF = pd.Timestamp('2025-06-30')


def save(obj, name):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def export(d, name):
    d.to_csv(OUT/name, index=False, encoding='utf-8-sig')


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def stat(g, method, denominator):
    lo, hi = g[f'{method}_lower90'], g[f'{method}_upper90']
    width = hi-lo
    return {'n': len(g), 'days': g.date.nunique(), 'denominator': denominator,
        'share': len(g)/denominator if denominator else np.nan,
        'coverage': ((g.wait>=lo)&(g.wait<=hi)).mean(),
        'early_rate': (g.wait<lo).mean(), 'late_rate': (g.wait>hi).mean(),
        'late_over20_rate': (g.wait>hi+20).mean(),
        'width_mean': width.mean(), 'width_median': width.median(), 'width_max': width.max()}


def bootstrap(g, method, start, end, seed):
    if len(g)<100 or g.date.nunique()<20:
        return {'coverage_lo95':np.nan,'coverage_hi95':np.nan,'width_lo95':np.nan,
                'width_hi95':np.nan,'bootstrap_reps':0}
    days=pd.date_range(start,end)
    dayid=pd.Index(days).get_indexer(g.date)
    assert (dayid>=0).all()
    lo,hi=g[f'{method}_lower90'],g[f'{method}_upper90']
    ns=np.bincount(dayid,minlength=len(days))
    agg=np.stack([np.bincount(dayid,weights=np.asarray(v,float),minlength=len(days))
        for v in [g.wait.between(lo,hi),hi-lo]])
    rng=np.random.default_rng(seed);draws=[]
    for _ in range(1000):
        starts=rng.integers(0,len(days)-6,size=int(np.ceil(len(days)/7)))
        w=np.bincount((starts[:,None]+np.arange(7)).ravel()[:len(days)],minlength=len(days))
        n=w@ns
        if n:draws.append(agg@w/n)
    values=np.asarray(draws)
    return {'coverage_lo95':np.quantile(values[:,0],.025),'coverage_hi95':np.quantile(values[:,0],.975),
            'width_lo95':np.quantile(values[:,1],.025),'width_hi95':np.quantile(values[:,1],.975),
            'bootstrap_reps':len(values)}


def main():
    assert json.loads((SOURCE/'verification.json').read_text(encoding='utf-8'))['passed']
    e=pd.read_pickle(SOURCE/'evaluation_predictions.pkl')
    e['phase']=np.where(e.month.le(6),'H1','H2')
    for method in METHODS:
        e[f'{method}_width']=e[f'{method}_upper90']-e[f'{method}_lower90']
    city=pd.read_pickle(HERE/'multi_region_wait/multiyear_trip_only/prepared_2025.pkl')
    assert len(city)==1003567 and city.row_id.is_unique
    matched=city[['row_id','request','board','wait']].merge(e[['row_id','request','board','wait']],
        on='row_id',suffixes=('_city','_eval'),validate='one_to_one')
    assert len(matched)==len(e)
    assert matched.request_city.eq(matched.request_eval).all() and matched.board_city.eq(matched.board_eval).all()
    np.testing.assert_allclose(matched.wait_city,matched.wait_eval,atol=0,rtol=0)
    save({'protocol_sha256':digest(HERE/'SHORT_INTERVAL_SCOPE_PROTOCOL.md'),
          'predictions_sha256':digest(SOURCE/'evaluation_predictions.pkl'),
          'selection_cutoff':CUTOFF,'discovery_end_request_exclusive':'2025-07-01',
          'confirmation_requests':'2025-07-01 <= request < 2026-01-01',
          'candidate_criteria':'n>=100,dates>=20,coverage>=.90 in known H1 data; district x hour and ex ante width gate',
          'caps':[30,45],'independent_untouched_test':False,
          'research_n':len(e),'seoul_comparable_completed_n':len(city),
          'no_new_model_training':True},'declaration.json')
    summary=[];subgroups=[];comparisons=[];members=[];candidate_rows=[];candidate_panels=[];allcells=[]
    for cap in [30,45]:
        for phase in ['all','H1','H2']:
            base=e if phase=='all' else e[e.phase.eq(phase)]
            city_n=len(city) if phase=='all' else int((city.month.le(6) if phase=='H1' else city.month.ge(7)).sum())
            for method in METHODS:
                g=base[base[f'{method}_width'].le(cap+1e-9)]
                start,end=('2025-01-01','2025-12-31') if phase=='all' else ('2025-01-01','2025-06-30') if phase=='H1' else ('2025-07-01','2025-12-31')
                summary.append({'cap':cap,'phase':phase,'method':method,**stat(g,method,len(base)),
                    'city_denominator':city_n,'share_of_city_confirmed_only':len(g)/city_n,
                    **bootstrap(g,method,start,end,20261022)})
                if method=='recent_weather':
                    for group in ['district','hour','month','weekday']:
                        for value,part in g.groupby(group):
                            denom=int(base[group].eq(value).sum())
                            subgroups.append({'cap':cap,'phase':phase,'group':group,'value':value,
                                **stat(part,method,denom)})
            for scope,mask in [('AI_selected',base.recent_weather_width.le(cap+1e-9)),
                               ('both_selected',base.recent_weather_width.le(cap+1e-9)&base.recent_table_width.le(cap+1e-9))]:
                for method in METHODS:
                    comparisons.append({'cap':cap,'phase':phase,'scope':scope,'method':method,
                        **stat(base[mask],method,len(base))})
        # All cells are descriptive and are not used to select H2 winners.
        known=e[e.month.le(6)&e.board.lt(CUTOFF)&e.recent_weather_width.le(cap+1e-9)]
        chosen=[]
        for (district,hour),g in known.groupby(['district','hour']):
            s=stat(g,'recent_weather',len(known))
            selected=s['n']>=100 and s['days']>=20 and s['coverage']>=.9
            allcells.append({'cap':cap,'district':district,'hour':hour,'selected':selected,
                'latest_board':g.board.max(),**s})
            if selected:chosen.append((district,int(hour)))
        chosen=set(chosen)
        allmask=np.array([(d,int(h)) in chosen for d,h in zip(e.district,e.hour)])&e.recent_weather_width.le(cap+1e-9).to_numpy()
        h2=e.month.ge(7)
        eligible=e[allmask&h2.to_numpy()]
        assert CUTOFF<=e.loc[h2,'decision'].min() and known.board.max()<CUTOFF
        for (district,hour) in sorted(chosen):
            k=known[known.district.eq(district)&known.hour.eq(hour)]
            g=eligible[eligible.district.eq(district)&eligible.hour.eq(hour)]
            s=stat(g,'recent_weather',int((e.loc[h2,'district'].eq(district)&e.loc[h2,'hour'].eq(hour)).sum()))
            candidate_rows.append({'cap':cap,'district':district,'hour':hour,
                'selection_n':len(k),'selection_days':k.date.nunique(),
                'selection_coverage':k.wait.between(k.recent_weather_lower90,k.recent_weather_upper90).mean(),
                'selection_latest_board':k.board.max(),'selection_cutoff':CUTOFF,
                'adequate_H2':len(g)>=100 and g.date.nunique()>=20,
                'observed_H2_ge88':s['coverage']>=.88,'observed_H2_ge90':s['coverage']>=.9,
                **s,**bootstrap(g,'recent_weather','2025-07-01','2025-12-31',20261022)})
        for phase,mask in [('all',allmask),('H2',allmask&h2.to_numpy())]:
            g=e[mask];base=e if phase=='all' else e[h2]
            city_n=len(city) if phase=='all' else int(city.month.ge(7).sum())
            for method in METHODS:
                candidate_panels.append({'cap':cap,'phase':phase,'method':method,'panel':'all_frozen_candidates',
                    'candidate_cells':len(chosen),**stat(g,method,len(base)),
                    'city_denominator':city_n,'share_of_city_confirmed_only':len(g)/city_n,
                    **bootstrap(g,method,'2025-01-01' if phase=='all' else '2025-07-01','2025-12-31',20261022)})
        for group in ['month','weekday','rain_forecast']:
            for value,g in eligible.groupby(group):
                candidate_panels.append({'cap':cap,'phase':'H2','method':'recent_weather','panel':group,
                    'value':str(value),**stat(g,'recent_weather',len(e[h2]))})
        for name,mask in [('JulAug_rain_forecast',eligible.month.isin([7,8])&eligible.rain_forecast),
                          ('JulAug_actual_rain_known',eligible.month.isin([7,8])&eligible.rain_next.gt(0))]:
            candidate_panels.append({'cap':cap,'phase':'H2','method':'recent_weather','panel':name,
                                    **stat(eligible[mask],'recent_weather',len(e[h2]))})
        for row_id in e.loc[allmask&h2.to_numpy(),'row_id']:
            members.append({'cap':cap,'row_id':row_id})
    export(pd.DataFrame(summary),'gate_summary.csv')
    export(pd.DataFrame(subgroups),'gate_subgroups.csv')
    export(pd.DataFrame(comparisons),'same_sample_comparison.csv')
    export(pd.DataFrame(allcells),'H1_all_cells.csv')
    export(pd.DataFrame(candidate_rows),'frozen_candidate_cells.csv')
    export(pd.DataFrame(candidate_panels),'frozen_candidate_summary.csv')
    export(pd.DataFrame(members),'H2_frozen_candidate_members.csv')
    print(pd.DataFrame(summary).query('method=="recent_weather"')[
        ['cap','phase','n','denominator','share','coverage','width_mean','coverage_lo95','coverage_hi95']].round(4).to_string(index=False))
    print(pd.DataFrame(candidate_panels).query('panel=="all_frozen_candidates" and phase=="H2"')[
        ['cap','method','candidate_cells','n','share','coverage','width_mean','coverage_lo95','coverage_hi95']].round(4).to_string(index=False))


if __name__=='__main__':main()
