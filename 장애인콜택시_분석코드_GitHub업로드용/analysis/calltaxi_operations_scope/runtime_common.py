"""Shared dependencies and unchanged numerical helpers for the release."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
DISTRICTS=sorted(['종로구','중구','용산구','성동구','광진구','동대문구','중랑구','성북구','강북구','도봉구','노원구','은평구','서대문구','마포구','양천구','강서구','구로구','금천구','영등포구','동작구','관악구','서초구','강남구','송파구','강동구'])


def period(h):
    return np.select([h.between(7,9),h.between(10,14),h.between(15,17),h.between(18,21)],
                     ['07-09','10-14','15-17','18-21'],default='22-06')

def scores(y,point,upper,w=None):
    y,point,upper=np.asarray(y),np.asarray(point),np.asarray(upper)
    w=np.ones(len(y)) if w is None else np.asarray(w)
    error=point-y;late=np.maximum(y-upper,0);excess=np.maximum(upper-y,0)
    avg=lambda a:float(np.average(a,weights=w))
    return {'n':int(w.sum()),'mae':avg(np.abs(error)),'rmse':np.sqrt(avg(error**2)),
            'point_bias':avg(error),'point_under20_rate':avg(error < -20),
            'coverage':avg(y<=upper),'upper_mean':avg(upper),'excess_mean':avg(excess),
            'shortfall_mean':avg(late),'over20_rate':avg(late>20),'pinball90':avg(.9*late+.1*excess)}

def residual_scores(y, median, base):
    # The floor at the representative value already covers these observations.
    return np.where(y <= median, -np.inf, y - base)

def shift_at_target(y, median, base, target, finite_sample=False):
    scores = residual_scores(y, median, base)
    rank = int(np.ceil((len(scores) + int(finite_sample)) * target)) - 1
    shift = float(np.partition(scores, min(rank, len(scores)-1))[min(rank, len(scores)-1)])
    assert np.isfinite(shift)
    return shift
