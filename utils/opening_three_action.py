"""Frozen probability-to-action rule. No realized execution outcome is an input."""
import numpy as np

THRESHOLDS = (0., .025, .05, .075, .10, .15, .20, .30)
ACTIONS = ('market', 'lastprice', 'passive')


def select_actions(direction, signal, p_down, p_up, threshold):
    """0=market, 1=LastPrice, 2=passive. Flat always uses LastPrice."""
    side, signal, down, up = np.broadcast_arrays(direction, signal, p_down, p_up)
    if not np.isin(side, [-1, 1]).all() or not np.isin(signal, [-1, 0, 1]).all():
        raise ValueError('Invalid side or signal')
    if not np.isfinite(down).all() or not np.isfinite(up).all():
        raise ValueError('Missing probabilities are not neutral signals')
    if ((down < 0) | (up < 0) | (down+up > 1+1e-10)).any() or threshold < 0 or np.isnan(threshold):
        raise ValueError('Invalid probability or threshold')
    favorable = side*(down-up)
    return np.where(side*signal > 0, 0,
                    np.where((side*signal < 0) & (favorable >= threshold), 2, 1))
