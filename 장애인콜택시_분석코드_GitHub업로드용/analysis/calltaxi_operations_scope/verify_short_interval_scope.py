"""Independently reconstruct width gates, frozen candidates and denominators."""
from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'.analysis_deps'))
import numpy as np
import pandas as pd

SOURCE=HERE/'multi_region_wait/annual_alternative_weather/boarding_intervals'
OUT=SOURCE/'short_interval_scope'


def main():
    e=pd.read_pickle(SOURCE/'evaluation_predictions.pkl')
    city=pd.read_pickle(HERE/'multi_region_wait/multiyear_trip_only/prepared_2025.pkl')
    summary=pd.read_csv(OUT/'gate_summary.csv')
    candidates=pd.read_csv(OUT/'frozen_candidate_cells.csv',parse_dates=['selection_latest_board','selection_cutoff'])
    frozen=pd.read_csv(OUT/'frozen_candidate_summary.csv')
    members=pd.read_csv(OUT/'H2_frozen_candidate_members.csv')
    assert len(e)==113776 and len(city)==1003567
    for _,r in summary.iterrows():
        base=e if r.phase=='all' else e[e.month.le(6) if r.phase=='H1' else e.month.ge(7)]
        w=base[f'{r.method}_upper90']-base[f'{r.method}_lower90']
        selected=base[w.le(r.cap+1e-9)]
        assert len(base)==r.denominator and len(selected)==r.n
        assert np.isclose(len(selected)/len(base),r.share)
        assert (selected[f'{r.method}_upper90']-selected[f'{r.method}_lower90']<=r.cap+1e-9).all()
        coverage=selected.wait.between(selected[f'{r.method}_lower90'],selected[f'{r.method}_upper90']).mean()
        assert np.isclose(coverage,r.coverage)
        cn=len(city) if r.phase=='all' else int((city.month.le(6) if r.phase=='H1' else city.month.ge(7)).sum())
        assert cn==r.city_denominator and np.isclose(r.n/cn,r.share_of_city_confirmed_only)
    cutoff=pd.Timestamp('2025-06-30')
    assert candidates.selection_latest_board.lt(candidates.selection_cutoff).all()
    assert candidates.selection_cutoff.eq(cutoff).all()
    assert cutoff<=e.loc[e.month.ge(7),'decision'].min()
    for cap in [30,45]:
        width=e.recent_weather_upper90-e.recent_weather_lower90
        known=e[e.month.le(6)&e.board.lt(cutoff)&width.le(cap+1e-9)].copy()
        known['hit']=known.wait.between(known.recent_weather_lower90,known.recent_weather_upper90)
        cells=known.groupby(['district','hour']).agg(n=('row_id','size'),days=('date','nunique'),coverage=('hit','mean'))
        expected=set(cells[(cells.n>=100)&(cells.days>=20)&(cells.coverage>=.9)].index)
        actual=set(zip(candidates.loc[candidates.cap.eq(cap),'district'],candidates.loc[candidates.cap.eq(cap),'hour']))
        assert expected==actual
        mask=np.array([(d,int(h)) in expected for d,h in zip(e.district,e.hour)])&e.month.ge(7).to_numpy()&width.le(cap+1e-9).to_numpy()
        g=e[mask]
        assert set(g.row_id)==set(members.loc[members.cap.eq(cap),'row_id'])
        for method in ['recent_weather','recent_table']:
            r=frozen[frozen.cap.eq(cap)&frozen.phase.eq('H2')&frozen.method.eq(method)&frozen.panel.eq('all_frozen_candidates')].iloc[0]
            assert r.n==len(g) and r.candidate_cells==len(expected)
            assert np.isclose(r.share,len(g)/sum(e.month.ge(7)))
            assert np.isclose(r.coverage,g.wait.between(g[f'{method}_lower90'],g[f'{method}_upper90']).mean())
        for _,r in candidates[candidates.cap.eq(cap)].iterrows():
            part=g[g.district.eq(r.district)&g.hour.eq(r.hour)]
            assert len(part)==r.n and part.date.nunique()==r.days
            assert bool(r.adequate_H2)==(len(part)>=100 and part.date.nunique()>=20)
            assert np.isclose(r.coverage,part.wait.between(part.recent_weather_lower90,part.recent_weather_upper90).mean())
    result={'passed':True,'gate_summary_rows_recomputed':len(summary),'frozen_candidates_rebuilt':True,
        'selection_uses_H1_boards_before_2025_06_30':True,'H2_winners_not_reselected':True,
        'H2_members_exact':True,'research_n':len(e),'city_comparable_completed_n':len(city),
        'city_outside_research_scope_not_evaluated':True,'new_independent_test':False}
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
