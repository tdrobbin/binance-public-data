import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Optional, Dict, Any, Callable


def ensure_freq(df: pd.DataFrame) -> pd.DataFrame:
    """
    takes a df with a datetime index and ensures that the index has a freq.
    if there is no freq then we infer it from the index. returns the same df
    but with a freq.
    
    Args:
        df: df with datetime index
    
    Returns:
        df with datetime index and freq
    """
    


def perf_stats(pnl, freq: Optional[str]=None):
    freq = freq or pd.infer_freq(pnl.index)

    DAYS_PER_YEAR = 365
    PREIODS_PER_DAY = 1440

    if freq == 'T':
        PREIODS_PER_DAY = 1440
    elif freq == 'H':
        PREIODS_PER_DAY = 24
    elif freq == 'D':
        PREIODS_PER_DAY = 1
 
    periods_per_yr = DAYS_PER_YEAR * PREIODS_PER_DAY

    # periods_per_yr = 252 * 1440
    # pnl = (sig.shift(1) * rets).sum(axis=1)
    sharpe = (pnl.mean() / pnl.std()) * np.sqrt(periods_per_yr)
    sortino = (pnl.mean() / pnl[pnl < 0].std()) * np.sqrt(periods_per_yr)
    max_dd = (pnl / pnl.cummax() - 1).min()
    pnl_mean = pnl.mean()
    pnl_std = pnl.std()
    
    return pd.Series({
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'pnl_mean': pnl_mean,
        'pnl_std': pnl_std,
        'pnl_count': len(pnl),
        'pnl_std_err': pnl_std / np.sqrt(len(pnl)),
        'pnl_tstat': pnl_mean / (pnl_std / np.sqrt(len(pnl)))
    })


class Universe:
    bar_col_names = ['secid', 'timestamp', 'open', 'high', 'low', 'close', 'volume']

    def __init__(
            self, 
            bar: pd.DataFrame, 
            universe_selection_metric: str='volumeusd', 
            universe_selection_metric_window: int=90, 
            universe_selection_metric_min_history: int=90,
            universe_selection_secid_count: int=30,
            universe_selection_metric_thresh: Optional[float]=None,
            risk_calculation_window: int=90,
            bar_freq: Optional[str]=None,
        ):
        """
        Args:
            bar: bar data with at minimum columns ['secid', 'timestamp', 'open', 'high', 'low', 'close', 'volume'].
            universe_selection_metric: metric to use for universe selection. Defaults to 'volume'.
            universe_selection_metric_window: window to use for universe selection metric. Defaults to 90.
            universe_selection_metric_min_history: min history to use for universe selection metric. Defaults to 30.
            universe_selection_secid_count: number of secids to select. Defaults to 30. 
                must pass this or universe_selection_mentric_thresh.
            universe_selection_mentric_thresh: threshold to use for universe selection metric. Defaults to None. 
                must pass this or universe_selection_secid_count.
        """
        # make sure bar has the right columns
        assert set(self.bar_col_names).issubset(bar.columns)

        # make sure only one of universe_selection_secid_count or universe_selection_mentric_thresh is passed
        assert (universe_selection_secid_count is None) != (universe_selection_metric_thresh is None)

        self.universe_selection_metric = universe_selection_metric
        self.universe_selection_metric_window = universe_selection_metric_window
        self.universe_selection_metric_min_history = universe_selection_metric_min_history
        self.universe_selection_secid_count = universe_selection_secid_count
        self.universe_selection_metric_thresh = universe_selection_metric_thresh

        bar = bar.copy()
        bar['timestamp'] = pd.to_datetime(bar['timestamp'])
        for col in ['open', 'high', 'low', 'close']:
            bar[col] = bar[col].astype(float)
        
        bar = bar.sort_values(by=['secid', 'timestamp'], ascending=[True, True])
        bar = bar.replace(0, np.nan)

        self.bar = bar

        self.open = bar.pivot(columns='secid', index='timestamp', values='open')
        self.high = bar.pivot(columns='secid', index='timestamp', values='high')
        self.low = bar.pivot(columns='secid', index='timestamp', values='low')
        self.close = bar.pivot(columns='secid', index='timestamp', values='close')
        self.volume = bar.pivot(columns='secid', index='timestamp', values='volume')
        self.volumeusd = bar.pivot(columns='secid', index='timestamp', values='volumeusd')

        self.rets = self.close.pct_change()
        self.log_rets = np.log(self.close).diff()

        self.trading_dates = pd.DataFrame({
            'start': self.close.apply(pd.Series.first_valid_index, axis=0), 
            'end': self.close.apply(pd.Series.last_valid_index, axis=0)
        })

        metric_for_mask_candidates = (
            getattr(self, universe_selection_metric)
            .rolling(universe_selection_metric_window)
            .mean()
            .where((
                (self.close.notnull()) & # is trading today
                (self.close.notnull().rolling(universe_selection_metric_window).sum() >= universe_selection_metric_min_history) # has min history
            ))
        )
        self.metric_for_mask_candidates = metric_for_mask_candidates

        if self.universe_selection_secid_count is not None:
            mask = (
                metric_for_mask_candidates
                .rank(axis=1, ascending=False)
                <= universe_selection_secid_count
            )
        else:
            mask = (
                metric_for_mask_candidates
                >= universe_selection_metric_thresh
            )
        self.mask = mask

        self.bmk_wgts = (
            getattr(self, universe_selection_metric)
            [mask]
            .apply(lambda x: x / x.sum(), axis=1)
        )
        self.bmk_rets = (self.rets * self.bmk_wgts.shift(1)).sum(axis=1)

        # use the rolling window to calculate the betas
        # self.asset_bmk_betas = self.rets.apply(lambda x: x.cov(self.bmk_rets) / self.bmk_rets.var())
        # self.asset_bmk_betas = self.rets.rolling(risk_calculation_window).apply(lambda x: x.cov(self.bmk_rets) / self.bmk_rets.var())
