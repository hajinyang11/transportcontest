"""Independent time-order and numerical checks for the interval comparison."""
from pathlib import Path
import sys, json, hashlib
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / '.analysis_deps'))
import numpy as np
import pandas as pd

SOURCE = HERE / 'multi_region_wait/annual_alternative_weather'
OUT = SOURCE / 'boarding_intervals'


def main():
    e = pd.read_pickle(OUT / 'evaluation_predictions.pkl')
    old = pd.read_pickle(SOURCE / 'evaluation_predictions.pkl').sort_values('row_id').reset_index(drop=True)
    assert len(e) == 113776 and e.row_id.is_unique and e.date.nunique() == 365
    for col in ['row_id', 'request', 'board', 'wait', 'rain_next']:
        assert e[col].equals(old[col]), col
    np.testing.assert_allclose(e.wait, (e.board - e.request).dt.total_seconds() / 60, rtol=0, atol=1e-9)
    fit = pd.read_csv(OUT / 'fit_audit.csv', parse_dates=['cutoff', 'latest_board', 'first_decision'])
    cal = pd.read_csv(OUT / 'calibration_audit.csv', parse_dates=['date', 'cutoff', 'latest_board', 'first_decision'])
    assert len(fit) == 28 and fit.latest_board.lt(fit.cutoff).all() and fit.cutoff.le(fit.first_decision).all()
    assert cal.latest_board.lt(cal.cutoff).all() and cal.cutoff.le(cal.first_decision).all()
    assert (cal.cutoff == cal.date - pd.Timedelta(days=1)).all()
    assert cal.n.ge(100).all() and cal.days.ge(5).all()
    expected = pd.read_csv(OUT / 'scores.csv')
    checked = 0
    for method in ['recent_table', 'recent_weather']:
        np.testing.assert_allclose(e[f'{method}_point'], old[f'{method}_point'], atol=0, rtol=0)
        lo90, lo80 = e[f'{method}_lower90'], e[f'{method}_lower80']
        hi90, hi80 = e[f'{method}_upper90'], e[f'{method}_upper80']
        assert (lo90 <= lo80).all() and (hi90 >= hi80).all()
        assert (lo90 >= 0).all() and (lo80 <= hi80).all()
    for _, row in expected.iterrows():
        if row.panel == 'all':
            mask = np.ones(len(e), bool)
        elif row.panel == 'JulAug':
            mask = e.month.isin([7, 8])
        elif row.panel == 'JulAug_rain_forecast_afternoon_evening':
            mask = e.month.isin([7, 8]) & e.rain_forecast & e.hour.between(15, 21)
        elif row.panel == 'JulAug_actual_rain_afternoon_evening':
            mask = e.month.isin([7, 8]) & e.rain_next.gt(0) & e.hour.between(15, 21)
        elif row.panel == 'JulAug_dry_forecast_afternoon_evening':
            mask = e.month.isin([7, 8]) & ~e.rain_forecast & e.hour.between(15, 21)
        elif row.panel == 'afternoon_evening':
            mask = e.hour.between(15, 21)
        elif row.panel == 'rain_forecast_afternoon_evening':
            mask = e.rain_forecast & e.hour.between(15, 21)
        elif row.panel == 'daytime_07_21':
            mask = e.hour.between(7, 21)
        else:
            raise AssertionError(row.panel)
        g = e[mask]
        if row.rule == 'forward':
            lo = g[f'{row.method}_lower{int(row.target)}'].to_numpy()
            hi = g[f'{row.method}_upper{int(row.target)}'].to_numpy()
            alpha = 1 - row.target / 100
        else:
            cap = int(row.rule.removeprefix('fixed'))
            center = g[f'{row.method}_point'].to_numpy()
            lo, hi = np.maximum(0, center - cap / 2), center + cap / 2
            assert (hi - lo <= cap + 1e-10).all()
            alpha = .1
        y = g.wait.to_numpy()
        assert np.isfinite(lo).all() and np.isfinite(hi).all() and (hi >= lo).all()
        width = hi - lo
        values = {'coverage': ((lo <= y) & (y <= hi)).mean(), 'width_mean': width.mean(),
            'width_median': np.median(width), 'width_p90': np.quantile(width, .9),
            'early_rate': (y < lo).mean(), 'late_rate': (y > hi).mean(),
            'late_over20_rate': (y > hi + 20).mean(),
            'interval_score': (width + 2 / alpha * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))).mean()}
        assert len(g) == row.n and g.date.nunique() == row.days
        for key, value in values.items():
            assert np.isclose(value, row[key], rtol=1e-10, atol=1e-10), (key, value, row[key])
        assert abs(values['coverage'] + values['early_rate'] + values['late_rate'] - 1) < 1e-12
        checked += 1
    # Explicitly rebuild calibration pools/order statistics for three dates,
    # not merely the audit inequalities, to check group selection and boundaries.
    for method in ['recent_table', 'recent_weather']:
        h = pd.read_pickle(OUT / f'history_{method}.pkl')
        def periods(hour):
            return np.select([hour.between(7, 9), hour.between(10, 14), hour.between(15, 17), hour.between(18, 21)],
                             ['07-09', '10-14', '15-17', '18-21'], default='22-06')
        h['period'] = periods(h.hour)
        for day in ['2025-01-01', '2025-07-15', '2025-12-01']:
            rows = cal[cal.method.eq(method) & cal.date.eq(pd.Timestamp(day))]
            for _, a in rows.iterrows():
                pool = h[h.board.lt(a.cutoff) & h.request.ge(a.cutoff - pd.Timedelta(days=56))]
                if a.level == 'district_period':
                    pool = pool[pool.district.eq(a.district) & pool.period.eq(a.period)]
                elif a.level == 'period':
                    pool = pool[pool.period.eq(a.period)]
                assert len(pool) == a.n and pool.date.nunique() == a.days and pool.board.max() == a.latest_board
                q = (1 - a.target / 100) / 2 if a.side == 'lower' else 1 - (1 - a.target / 100) / 2
                # Round away binary representation noise for the intended 5/10/90/95 percentiles.
                q = round(q, 2)
                col = {0.05: 'q05', .1: 'q10', .9: 'q90', .95: 'q95'}[q]
                residual = np.sort((pool.wait - pool[col]).to_numpy())
                rank = int(np.floor((len(pool) + 1) * q) if a.side == 'lower' else np.ceil((len(pool) + 1) * q))
                expected_shift = residual[min(max(rank, 1), len(pool)) - 1]
                assert np.isclose(expected_shift, a['shift'], rtol=1e-11, atol=1e-11)
    declaration = json.loads((OUT / 'declaration.json').read_text(encoding='utf-8'))
    assert not declaration['future_observed_weather_used_as_input']
    result = {'passed': True, 'n': len(e), 'days': 365, 'metric_rows_recomputed': checked,
        'fit_time_checks': len(fit), 'calibration_time_checks': len(cal),
        'direct_calibration_rebuild_dates': ['2025-01-01', '2025-07-15', '2025-12-01'],
        'existing_central_predictions_unchanged': True,
        'all_intervals_nested_nonnegative_finite': True,
        'protocol_sha256': declaration['protocol_sha256'],
        'scope': 'completed trips, two forecast grids, previously explored 2025 internal development'}
    (OUT / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
