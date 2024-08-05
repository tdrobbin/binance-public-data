import numpy as np
import pandas as pd
import ibis
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from tqdm.auto import tqdm
import ibis
from ibis import _

from typing import List, Union, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass


def custom_infer_freq(index: pd.DatetimeIndex) -> str:
    """
    infer the frequency of a pandas datetime index. This function is a modified version of the pandas infer_freq function.
    take the difference between each consecutive value in the index and then take the mode of the differences. in seconds.
    if the mode is close to 60 +/- 10 seconds, then the frequency is 'T'. if the mode is close to 3600 +/- 600 seconds, then the frequency is 'H'.
    if the mode is close to 86400 +/- 14400 seconds, then the frequency is 'D'. if the mode is close to 604800 +/- 86400 seconds, then the frequency is 'W'.
    if the mode is close to 2592000 +/- 432000 seconds, then the frequency is 'M'. if the mode is close to 31536000 +/- 5256000 seconds, then the frequency is 'Y'.

    """
    if len(index) < 2:
        return None

    freq = np.nanmean(index.diff().total_seconds())

    if 50 <= freq <= 70:
        return 'T'
    elif 3500 <= freq <= 4100:
        return 'H'
    elif 85000 <= freq <= 93000:
        return 'D'
    elif 590000 <= freq <= 620000:
        return 'W'
    elif 2550000 <= freq <= 2630000:
        return 'M'
    elif 30900000 <= freq <= 32000000:
        return 'Y'
    else:
        return None


def perf_stats(pnl, freq: Optional[str]=None):
    freq = freq or pd.infer_freq(pnl.index)

    freq = freq or custom_infer_freq(pnl.index)

    DAYS_PER_YEAR = 365
    PREIODS_PER_DAY = 1440

    if freq == 'T':
        PREIODS_PER_DAY = 1440
    elif freq == 'H':
        PREIODS_PER_DAY = 24
    elif freq == 'D':
        PREIODS_PER_DAY = 1
    elif freq == 'W':
        PREIODS_PER_DAY = 1 / 7
    elif freq == 'M':
        PREIODS_PER_DAY = 1 / 30
 
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


@dataclass
class Backtest:
    bt_df: ibis.Table
    bt_dly_df: ibis.Table
    bt_perf_stats: pd.Series
    bt_dly_perf_stats: pd.Series


class Universe:
    _bar_columns_required = ['secid', 'close_time', 'open', 'high', 'low', 'close', 'volume']

    def __init__(
        self,
        bar: ibis.Table,
        selection_metric: str='volumeusd',
        selection_secid_count: int=20,
        selection_metric_window: int=7,
        selection_metric_min_thresh: float=None,
        selection_metric_min_hist: int=7,
        risk_calculation_window: int=180,
        weights: ibis.Table=None
    ):
        """
        Args:
            bar: An ibis table with at a minimum the columns defined in self._bar_columns_required.
            selection_secid_count: The number of securities to select for the universe.
            selection_metric: The metric to use for selecting the securities.
            selection_metric_window: The rolling window to use for the selection metric.
            selection_metric_min_thresh: The minimum threshold for the selection metric.
            selection_metric_min_hist: The minimum number of periods of history required for a security to be considered.
            weights: An ibis table with the columns 'secid', 'open_time', 'close_time', 'apply_open_time', 'apply_close_time', 'universe_weight', 'universe_mask'.
                if None, then the weights will be calculated. If not None, then the weights will be used as is.
        """
        for c in self._bar_columns_required:
            assert c in bar.columns, c

        if weights is None:
            self.weights_were_calculated = True
        else:
            self.weights_were_calculated = False

        self.selection_secid_count = selection_secid_count
        self.selection_metric = selection_metric
        self.selection_metric_window = selection_metric_window
        self.selection_metric_min_thresh = selection_metric_min_thresh
        self.selection_metric_min_hist = selection_metric_min_hist
        self.risk_calculation_window = risk_calculation_window
        
        if 'volumeusd' not in bar.columns:
            bar = bar.mutate(volumeusd=bar.volume * bar.close)
        
        if weights is None:
            weights = (
                bar
                # add day rolling avg colum of volumeusd
                .mutate(
                    selection_metric_roll_avg=(
                        _[selection_metric]
                        .mean()
                        .over(
                            ibis.trailing_window(
                                preceding=(selection_metric_window - 1), 
                                group_by=_.secid, 
                                order_by=_.close_time
                        ))
                    )
                )
                .group_by('close_time')
                .mutate(
                    rank_asc=_['selection_metric_roll_avg'].rank(),
                    rank_desc=ibis.rank().over(order_by=ibis.desc(_['selection_metric_roll_avg']))
                )
                .filter(_.rank_desc < (selection_secid_count if selection_secid_count is not None else 1e6))
                .filter(_[selection_metric] >= (selection_metric_min_thresh if selection_metric_min_thresh is not None else 0))
                .group_by('close_time')
                .mutate(
                    universe_weight=_['selection_metric_roll_avg'] / _['selection_metric_roll_avg'].sum(),
                    universe_mask=True
                )
                # need to "shift forward" make columns for apply_close_time and apply_open_time that 
                # are equivalent to the next close_time and open_time for a given secid
                .group_by('secid')
                .order_by('close_time')
                .mutate(
                    apply_open_time=_.open_time.lag(-1),
                    apply_close_time=_.close_time.lag(-1),
                )
                # .select('secid', 'open_time', 'close_time', 'apply_open_time', 'apply_close_time', 'universe_weight', 'universe_mask')
            )
        
        self.weights = weights
        self.universe_secids = weights.alias('weights').sql('select distinct secid from weights').to_pandas().secid.to_list()
        
        # now augment bar data with secid level returns and risk metrics
        bar = (
            bar
            .filter(_.secid.isin(self.universe_secids))
            .left_join(
                weights.select('universe_weight', 'universe_mask', 'secid', 'apply_open_time', 'apply_close_time'), 
                [
                    bar.secid == weights.secid,
                    # bar.open_time.date() == weights.apply_open_time.date(),
                    # bar.close_time.date() == weights.apply_close_time.date(),
                    bar.open_time >= weights.apply_open_time,
                    bar.close_time <= weights.apply_close_time
                ]
            )
            .fillna(dict(
                universe_weight=0,
                universe_mask=False
            ))
            .drop('secid_right', 'apply_open_time', 'apply_close_time')
            .group_by('secid')
            .order_by('close_time')
            .mutate(
                **{'return': ((_.close / _.close.lag()) - 1)},
                # log_ret=_.close.log() - _.close.lag().log()
            )
        )

        universe_return = (
            bar
            .group_by('close_time')
            .aggregate(universe_return=(bar.universe_weight * bar['return']).sum())
            .select('close_time', 'universe_return')
            .mutate(
                universe_volatility=_.universe_return.std(how='pop').over(ibis.trailing_window(preceding=(risk_calculation_window - 1), order_by=_.close_time))
            )
        )
        bar = (
            bar
            .left_join(
                universe_return,
                [bar.close_time == universe_return.close_time]
            )
            .drop('close_time_right')
        )

        bar = (
            bar
            .group_by('secid')
            .order_by('close_time')
            .mutate(
                volatility=_['return'].std(how='pop').over(ibis.trailing_window(preceding=(risk_calculation_window - 1), group_by=_.secid, order_by=_.close_time)),
                correlation=_['return'].corr(_.universe_return, how='pop').over(ibis.trailing_window(preceding=(risk_calculation_window - 1), group_by=_.secid, order_by=_.close_time)),
            )
            .mutate(
                beta=_['correlation'] * (_.volatility / _.universe_volatility)
            )
            .mutate(
                alpha=(
                    _['return'].mean().over(ibis.trailing_window(preceding=(risk_calculation_window - 1), group_by=_.secid, order_by=_.close_time)) \
                    - _.beta * _.universe_return.mean().over(ibis.trailing_window(preceding=(risk_calculation_window - 1), group_by=_.secid, order_by=_.close_time))
                )
            )
            .mutate(
                residual=_['return'] - _.alpha - _.beta * _.universe_return
            )
            .mutate(
                residual_volatility=_.residual.std(how='pop').over(ibis.trailing_window(preceding=(risk_calculation_window - 1), group_by=_.secid, order_by=_.close_time))
            )
        )

        self.bar = bar
    
    def backtest(self, signal: ibis.Table, tcost_model: float = .0006, fwd: int = 1) -> Backtest:
        sig = signal

        
            # .group_by('secid')
            # .order_by('close_time')
            # .mutate(
            #     # feat=_.pct_change.sum().over(ibis.trailing_window(preceding=(bwd-1), group_by=_.secid, order_by=_.close_time)),
            #     wgt_agg=_.wgt.mean().over(ibis.trailing_window(preceding=(fwd-1), group_by=_.secid, order_by=_.close_time)),
            # )
            # .group_by('secid')
            # .order_by('close_time')
            # .mutate(
            #     pnl_wgt_agg=_.wgt_agg.lag()


        pnl = (
            sig
            # -- add wgt_agg
            .group_by('secid')
            .order_by('close_time')
            .mutate(
                # feat=_.pct_change.sum().over(ibis.trailing_window(preceding=(bwd-1), group_by=_.secid, order_by=_.close_time)),
                wgt_agg=(_.wgt if fwd == 1 else _.wgt.mean().over(ibis.trailing_window(preceding=(fwd-1), group_by=_.secid, order_by=_.close_time))),
            )
            .group_by('secid')
            .order_by('close_time')
            .mutate(
                pnl_wgt_agg=_.wgt_agg.lag()
            )
            # -- end add
            .group_by('secid')
            .order_by('close_time')
            .mutate(
                pnl_gross=_.pnl_wgt_agg * _['return'],
                turnover=_.pnl_wgt_agg - _.pnl_wgt_agg.lag(),
                tcost_model=tcost_model
            )
            .mutate(
                tcost=_.turnover.abs() * _.tcost_model
            )
            .mutate(
                pnl_net=_.pnl_gross - _.tcost
            )
        )

        bt = (
            pnl
            .group_by('close_time')
            .aggregate(
                por_ret_gross=_.pnl_gross.sum(),
                por_ret_net=_.pnl_net.sum(),
            )
            .order_by('close_time')
        )

        bt_dly = (
            bt
            .group_by(_.close_time.date(), )
            .aggregate(
                por_ret_gross=_.por_ret_gross.sum(),
                por_ret_net=_.por_ret_net.sum(),
            )
            .mutate(
                close_time='Date(close_time)'
            )
            .order_by('close_time')
        )

        bt_df = bt.select(['close_time', 'por_ret_gross', 'por_ret_net']).to_pandas().set_index('close_time')
        bt_dly_df = bt_dly.select(['close_time', 'por_ret_gross', 'por_ret_net']).to_pandas().set_index('close_time')

        bt = Backtest(
            bt_df=bt_df,
            bt_dly_df=bt_dly_df,
            bt_perf_stats=bt_df.apply(perf_stats).T,
            bt_dly_perf_stats=bt_dly_df.apply(perf_stats).T
        )

        return bt





# feat
# .left_join(
#     unv.select('universe_weight', 'universe_mask', 'secid', 'apply_open_time', 'apply_close_time'), 
#     [
#         feat.secid == unv.secid,
#         feat.open_time.date() == unv.apply_open_time.date(),
#         feat.close_time.date() == unv.apply_close_time.date()
#     ]
# )
# .fillna(dict(
#     universe_weight=0,
#     universe_mask=False
# ))