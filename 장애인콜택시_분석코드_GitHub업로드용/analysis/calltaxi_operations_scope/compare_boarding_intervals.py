"""Forward, two-sided boarding intervals; fixed protocol, no evaluation tuning."""
from pathlib import Path
import sys, json, hashlib, time, argparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_annual_alternative_weather import (
    pd, np, HistGradientBoostingRegressor, threadpool_limits,
    features, weights, META, DISTRICTS, period,
)

SOURCE = HERE / 'multi_region_wait/annual_alternative_weather'
OUT = SOURCE / 'boarding_intervals'
OUT.mkdir(exist_ok=True)
METHODS = ['recent_table', 'recent_weather']
PERIODS = ['07-09', '10-14', '15-17', '18-21', '22-06']
TARGETS = {80: (.1, .9), 90: (.05, .95)}
COLS = {.05: 'q05', .1: 'q10', .9: 'q90', .95: 'q95'}


def save(obj, name):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                     default=str), encoding='utf-8')


def export(df, name):
    df.to_csv(OUT / name, index=False, encoding='utf-8-sig')


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def fit():
    d = pd.read_pickle(SOURCE / 'features.pkl')
    config = json.loads((SOURCE / 'selected_2024_model.json').read_text(encoding='utf-8'))['config']
    declaration = {
        'protocol_sha256': digest(HERE / 'BOARDING_INTERVAL_PROTOCOL.md'),
        'features_sha256': digest(SOURCE / 'features.pkl'),
        'previous_predictions_sha256': digest(SOURCE / 'evaluation_predictions.pkl'),
        'config': config, 'methods': METHODS, 'targets': TARGETS,
        'new_ai_quantiles': [.05, .1, .95], 'reused_ai_quantiles': [.5, .9],
        'evaluation': '2025 previously explored internal development; forward monthly fit and daily calibration',
        'calibration': 'prior56day district-period n>=100, dates>=5 else period/all; known before previous midnight',
        'fixed_width_caps': [30, 60, 90],
        'future_observed_weather_used_as_input': False,
    }
    dp = OUT / 'declaration.json'
    if dp.exists():
        assert json.loads(dp.read_text(encoding='utf-8')) == json.loads(json.dumps(declaration))
    save(declaration, 'declaration.json')
    audits = []
    for start in pd.date_range('2024-11-01', '2025-12-01', freq='MS'):
        cutoff = start - pd.Timedelta(days=1)
        train = d[d.board.lt(cutoff)]
        target = d[d.request.ge(start) & d.request.lt(start + pd.offsets.MonthBegin(1))]
        assert train.board.max() < cutoff <= target.decision.min()
        for method in METHODS:
            name = f'{method}_{start:%Y%m}'
            path, ap = OUT / f'raw_{name}.pkl', OUT / f'fit_{name}.json'
            if path.exists() and ap.exists():
                audits.append(json.loads(ap.read_text(encoding='utf-8')))
                continue
            started = time.perf_counter()
            old = pd.read_pickle(SOURCE / f'pred_{name}.pkl')
            pred = target[META].merge(old[['year', 'row_id', 'raw_median', 'raw_upper']],
                                      on=['year', 'row_id'], validate='one_to_one')
            assert len(pred) == len(target)
            # Merge preserves target order. Validate features and metadata before prediction.
            assert np.array_equal(pred.row_id, target.row_id)
            if method == 'recent_weather':
                cols, cats = features(d, True)
                w = weights(train, cutoff, True)
                for q in [.05, .1, .95]:
                    model = HistGradientBoostingRegressor(loss='quantile', quantile=q,
                        categorical_features=cats, **config)
                    model.fit(train[cols], train.wait, sample_weight=w)
                    pred[COLS[q]] = np.maximum(0, model.predict(target[cols]))
                used = train
            else:
                used = train[train.request.ge(cutoff - pd.Timedelta(days=90))]
                ix = pd.MultiIndex.from_frame(target[['district', 'weekday', 'hour']])
                bx = pd.MultiIndex.from_frame(target[['weekday', 'hour']])
                groups = used.groupby(['district', 'weekday', 'hour']).wait
                broad = used.groupby(['weekday', 'hour']).wait
                counts = groups.size().reindex(ix).fillna(0).to_numpy()
                for q in [.05, .1, .95]:
                    values = groups.quantile(q).reindex(ix).to_numpy()
                    fallback = broad.quantile(q).reindex(bx).fillna(used.wait.quantile(q)).to_numpy()
                    pred[COLS[q]] = np.maximum(0, np.where((counts < 30) | np.isnan(values), fallback, values))
            pred['q90'] = pred.raw_upper
            crossing = ((pred.q05 > pred.q10) | (pred.q10 > pred.raw_median) |
                        (pred.q95 < pred.q90)).sum()
            pred['q10'] = np.minimum(pred.q10, pred.raw_median)
            pred['q05'] = np.minimum(pred.q05, pred.q10)
            pred['q95'] = np.maximum(pred.q95, pred.q90)
            pred.to_pickle(path)
            audit = {'method': method, 'month': f'{start:%Y-%m}', 'cutoff': cutoff,
                'latest_board': used.board.max(), 'first_decision': target.decision.min(),
                'train_n': len(used), 'target_n': len(target), 'raw_crossing_repaired_n': int(crossing),
                'seconds': round(time.perf_counter() - started, 3)}
            save(audit, ap.name)
            audits.append(audit)
            print('fit', name, 'seconds', audit['seconds'], flush=True)
    export(pd.DataFrame(audits), 'fit_audit.csv')


def calibrate(h, old, method):
    h = h.copy()
    h['period'] = period(h.hour)
    request = h.request.astype('int64').to_numpy()
    board = h.board.astype('int64').to_numpy()
    dates = h.date.astype('int64').to_numpy()
    decision = h.decision.astype('int64').to_numpy()
    district = pd.Categorical(h.district, categories=DISTRICTS).codes.astype('int64')
    pc = pd.Categorical(h.period, categories=PERIODS).codes
    cells = district * 5 + pc
    y = h.wait.to_numpy()
    keep = h.year.eq(2025).to_numpy()
    out = h[keep][META + ['period']].copy()
    aligned = out[['row_id']].merge(old[['row_id', method + '_point']], on='row_id', validate='one_to_one')
    point = np.full(len(h), np.nan)
    point[keep] = aligned[method + '_point'].to_numpy()
    arrays = {(target, side): np.full(len(h), np.nan) for target in TARGETS for side in ['lower', 'upper']}
    audit = []

    def groups(pos, codes):
        order = np.argsort(codes[pos], kind='stable')
        ordered = pos[order]
        split = np.flatnonzero(codes[ordered][1:] != codes[ordered][:-1]) + 1
        return {int(codes[g[0]]): g for g in np.split(ordered, split) if len(g)}

    raw = {q: h[col].to_numpy() for q, col in COLS.items()}
    nsday = 86400 * 10**9
    for day in pd.date_range('2025-01-01', '2025-12-31'):
        cutoff = day - pd.Timedelta(days=1)
        past = np.flatnonzero((board < cutoff.value) & (request >= cutoff.value - 56 * nsday))
        now = np.flatnonzero(dates == day.value)
        assert len(now) and len(past) and board[past].max() < cutoff.value <= decision[now].min()
        local, broad = groups(past, cells), groups(past, pc)
        cache = {}
        for cell, pos in groups(now, cells).items():
            chosen, level = local.get(cell), 'district_period'
            if chosen is None or len(chosen) < 100 or len(np.unique(dates[chosen])) < 5:
                chosen, level = broad.get(cell % 5), 'period'
                if chosen is None or len(chosen) < 100 or len(np.unique(dates[chosen])) < 5:
                    chosen, level = past, 'all'
            groupkey = (level, cell if level == 'district_period' else cell % 5 if level == 'period' else -1)
            n = len(chosen)
            for target, quantiles in TARGETS.items():
                for side, q in zip(['lower', 'upper'], quantiles):
                    key = (groupkey, q)
                    if key not in cache:
                        # One-based lower floor and upper ceiling order statistics.
                        rank = int(np.floor((n + 1) * q) if side == 'lower' else np.ceil((n + 1) * q))
                        rank = min(max(rank, 1), n)
                        residual = y[chosen] - raw[q][chosen]
                        cache[key] = float(np.partition(residual, rank - 1)[rank - 1])
                    shift = cache[key]
                    values = raw[q][pos] + shift
                    if side == 'lower':
                        values = np.maximum(0, np.minimum(point[pos], values))
                    else:
                        values = np.maximum(point[pos], values)
                    arrays[target, side][pos] = values
                    audit.append({'method': method, 'date': day, 'district': DISTRICTS[cell // 5],
                        'period': PERIODS[cell % 5], 'target': target, 'side': side, 'cutoff': cutoff,
                        'first_decision': pd.Timestamp(decision[pos].min()),
                        'latest_board': pd.Timestamp(board[chosen].max()),
                        'level': level, 'n': n, 'days': len(np.unique(dates[chosen])), 'shift': shift})
    for target in TARGETS:
        for side in ['lower', 'upper']:
            out[f'{side}{target}'] = arrays[target, side][keep]
    nested = (out.lower90 > out.lower80) | (out.upper90 < out.upper80)
    out['lower90'] = np.minimum(out.lower90, out.lower80)
    out['upper90'] = np.maximum(out.upper90, out.upper80)
    out['point'] = point[keep]
    out['nested_repaired'] = nested.to_numpy()
    return out.sort_values('row_id').reset_index(drop=True), pd.DataFrame(audit)


def stats(y, lower, upper, alpha):
    y, lower, upper = np.asarray(y), np.asarray(lower), np.asarray(upper)
    width = upper - lower
    early, late = np.maximum(lower - y, 0), np.maximum(y - upper, 0)
    return {'n': len(y), 'coverage': np.mean((y >= lower) & (y <= upper)),
        'early_rate': np.mean(y < lower), 'late_rate': np.mean(y > upper),
        'late_over20_rate': np.mean(y > upper + 20),
        'lower_mean': lower.mean(), 'upper_mean': upper.mean(),
        'width_mean': width.mean(), 'width_median': np.median(width), 'width_p90': np.quantile(width, .9),
        'width_le30': np.mean(width <= 30 + 1e-9), 'width_le60': np.mean(width <= 60 + 1e-9), 'width_le90': np.mean(width <= 90 + 1e-9),
        'interval_score': np.mean(width + 2 / alpha * (early + late))}


def panels(e):
    summer = e.month.isin([7, 8])
    ae = e.hour.between(15, 21)
    rain = e.rain_forecast
    return {'all': np.ones(len(e), bool), 'JulAug': summer,
        'JulAug_rain_forecast_afternoon_evening': summer & rain & ae,
        'JulAug_actual_rain_afternoon_evening': summer & e.rain_next.gt(0) & ae,
        'JulAug_dry_forecast_afternoon_evening': summer & ~rain & ae,
        'afternoon_evening': ae, 'rain_forecast_afternoon_evening': rain & ae,
        'daytime_07_21': e.hour.between(7, 21)}


def evaluate():
    old = pd.read_pickle(SOURCE / 'evaluation_predictions.pkl').sort_values('row_id').reset_index(drop=True)
    e = old[META + ['period', 'rain_next']].copy()
    audits = []
    for method in METHODS:
        h = pd.concat([pd.read_pickle(OUT / f'raw_{method}_{s:%Y%m}.pkl')
            for s in pd.date_range('2024-11-01', '2025-12-01', freq='MS')], ignore_index=True)
        assert not h[['year', 'row_id']].duplicated().any()
        h.to_pickle(OUT / f'history_{method}.pkl')
        pred, audit = calibrate(h, old, method)
        assert np.array_equal(pred.row_id, e.row_id) and np.array_equal(pred.wait, e.wait)
        for col in ['lower80', 'upper80', 'lower90', 'upper90', 'point', 'nested_repaired']:
            e[f'{method}_{col}'] = pred[col].to_numpy()
        audits.append(audit)
        print('calibrated', method, len(e), flush=True)
    assert len(e) == 113776
    e.to_pickle(OUT / 'evaluation_predictions.pkl')
    export(pd.concat(audits, ignore_index=True), 'calibration_audit.csv')
    rows, bins = [], []
    masks = panels(e)
    for panel, mask in masks.items():
        g = e[mask]
        if not len(g):
            continue
        for method in METHODS:
            for target in TARGETS:
                lower, upper = g[f'{method}_lower{target}'], g[f'{method}_upper{target}']
                rows.append({'panel': panel, 'days': g.date.nunique(), 'method': method,
                    'rule': 'forward', 'target': target, **stats(g.wait, lower, upper, 1 - target / 100)})
                width = upper - lower
                for name, selected in [('le30', width <= 30 + 1e-9), ('30to60', (width > 30 + 1e-9) & (width <= 60 + 1e-9)),
                                       ('60to90', (width > 60 + 1e-9) & (width <= 90 + 1e-9)), ('over90', width > 90 + 1e-9)]:
                    if not selected.sum():
                        continue
                    bins.append({'panel': panel, 'method': method, 'target': target, 'width_bin': name,
                        'share_of_panel': selected.mean(), 'days': g.loc[selected, 'date'].nunique(),
                        **stats(g.wait[selected], lower[selected], upper[selected], 1 - target / 100)})
            for cap in [30, 60, 90]:
                point = g[f'{method}_point']
                lower, upper = np.maximum(0, point - cap / 2), point + cap / 2
                rows.append({'panel': panel, 'days': g.date.nunique(), 'method': method,
                    'rule': f'fixed{cap}', 'target': np.nan, **stats(g.wait, lower, upper, .1)})
    scores = pd.DataFrame(rows)
    export(scores, 'scores.csv')
    export(pd.DataFrame(bins), 'width_bin_scores.csv')
    subgroups = []
    for group in ['district', 'grid', 'month', 'period']:
        for value, g in e.groupby(group):
            for method in METHODS:
                for target in TARGETS:
                    subgroups.append({'group': group, 'value': value, 'method': method, 'target': target,
                        'days': g.date.nunique(), **stats(g.wait, g[f'{method}_lower{target}'],
                                                       g[f'{method}_upper{target}'], 1 - target / 100)})
    export(pd.DataFrame(subgroups), 'subgroup_scores.csv')
    rng = np.random.default_rng(20261022)
    boot = []
    for panel in ['all', 'JulAug', 'JulAug_rain_forecast_afternoon_evening', 'JulAug_actual_rain_afternoon_evening']:
        g = e[masks[panel]]
        days = pd.date_range('2025-01-01', '2025-12-31') if panel == 'all' else pd.date_range('2025-07-01', '2025-08-31')
        dayid = pd.Index(days).get_indexer(g.date)
        assert (dayid >= 0).all()
        day_n = np.bincount(dayid, minlength=len(days))
        daily = {}
        for target in TARGETS:
            for method in METHODS:
                lo, hi = g[f'{method}_lower{target}'], g[f'{method}_upper{target}']
                daily[method, target] = np.stack([np.bincount(dayid, weights=np.asarray(v, float), minlength=len(days))
                    for v in [(g.wait >= lo) & (g.wait <= hi), hi - lo, g.wait > hi + 20]])
        for rep in range(1000):
            starts = rng.integers(0, len(days) - 6, size=int(np.ceil(len(days) / 7)))
            dw = np.bincount((starts[:, None] + np.arange(7)).ravel()[:len(days)], minlength=len(days))
            n = dw @ day_n
            if not n:
                continue
            for target in TARGETS:
                for method in METHODS:
                    coverage, width, late20 = daily[method, target] @ dw / n
                    boot.append({'panel': panel, 'rep': rep, 'target': target, 'method': method,
                        'coverage': coverage, 'width_mean': width, 'late_over20_rate': late20})
    b = pd.DataFrame(boot)
    ci = []
    for keys, g in b.groupby(['panel', 'target', 'method']):
        for metric in ['coverage', 'width_mean', 'late_over20_rate']:
            lo, hi = g[metric].quantile([.025, .975])
            ci.append(dict(zip(['panel', 'target', 'method'], keys), metric=metric,
                           lo95=lo, hi95=hi, reps=len(g)))
    export(pd.DataFrame(ci), 'bootstrap_summary.csv')
    diff = []
    for (panel, target), g in b.groupby(['panel', 'target']):
        a = g[g.method.eq('recent_weather')].set_index('rep')
        t = g[g.method.eq('recent_table')].set_index('rep')
        for name, values in [('coverage_difference_pp', 100 * (a.coverage - t.coverage)),
                             ('width_saved_minutes', t.width_mean - a.width_mean)]:
            lo, hi = values.quantile([.025, .975])
            diff.append({'panel': panel, 'target': target, 'metric': name, 'lo95': lo, 'hi95': hi})
    export(pd.DataFrame(diff), 'paired_bootstrap_summary.csv')
    export(pd.DataFrame([{'panel': name, 'n': int(mask.sum()), 'days': e.loc[mask, 'date'].nunique()}
                        for name, mask in masks.items()]), 'panel_counts.csv')
    save({'nested_expansion_n': {m: int(e[f'{m}_nested_repaired'].sum()) for m in METHODS},
          'n': len(e), 'days': e.date.nunique(), 'bootstrap': '1000 fixed-prediction moving7day blocks; no refit uncertainty'},
         'evaluation_audit.json')
    print(scores[scores.panel.isin(['all', 'JulAug_rain_forecast_afternoon_evening',
          'JulAug_actual_rain_afternoon_evening']) & scores.rule.eq('forward')][
          ['panel','method','target','coverage','width_mean']].to_string(index=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fit', action='store_true')
    p.add_argument('--evaluate', action='store_true')
    args = p.parse_args()
    with threadpool_limits(limits=4):
        if args.fit:
            fit()
        if args.evaluate:
            evaluate()
