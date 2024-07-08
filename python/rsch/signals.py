import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from tqdm.auto import tqdm

from typing import List, Union, Tuple, Optional, Dict, Any, Callable


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
            universe_selection_metric_thresh: float=None,
            include_risk_calculations: bool=False,
            risk_calculation_window: int=90,
            bar_freq: str=None,
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

        self.include_risk_calculations = include_risk_calculations
        self.risk_calculation_window = risk_calculation_window

        self.bmk_secids = self.bmk_wgts.dropna(axis=1, how='all').columns

        base_df = pd.DataFrame(
            index=self.bmk_rets.index, 
            columns=self.bmk_secids
        )

        self.beta = base_df.copy()
        self.alpha = base_df.copy()
        self.yhat = base_df.copy()
        self.residual_ret = base_df.copy()
        self.var = base_df.copy()
        self.vol = base_df.copy()
        self.residual_var_rlzd = base_df.copy()
        self.residual_var_pred = base_df.copy()
        self.residual_vol_rlzd = base_df.copy()
        self.residual_vol_pred = base_df.copy()

        if include_risk_calculations:
            assert isinstance(risk_calculation_window, int) and 3 <= risk_calculation_window

            for secid, y in tqdm(self.rets[self.bmk_secids].items()):
                X = self.bmk_rets
                y, X = y.replace(0, np.nan).dropna(), X.replace(0, np.nan).dropna()
                y, X = y.align(X, join='inner')
                X = sm.add_constant(X)

                window = 90
                model = RollingOLS(y, X, window=window)
                try:
                    results = model.fit()
                except IndexError:
                    continue

                coefs = results.params
                beta = coefs.iloc[:, 1]
                alpha = results.params.iloc[:, 0]
                yhat = (X * coefs).sum(axis=1)
                residual_ret = y - yhat
                var = y.rolling(window=window).var()
                vol = y.rolling(window=window).std()
                residual_var_rlzd = residual_ret.rolling(window=window).var()
                residual_var_pred = var - (beta.apply(np.square) * self.bmk_rets.var())
                residual_vol_rlzd = pd.Series(index=residual_var_rlzd.index, data=np.sqrt(residual_var_rlzd))
                residual_vol_pred = pd.Series(index=residual_var_pred.index, data=np.sqrt(residual_var_pred))

                self.beta[secid] = beta
                self.alpha[secid] = alpha
                self.yhat[secid] = yhat
                self.residual_ret[secid] = residual_ret
                self.var[secid] = var
                self.vol[secid] = vol
                self.residual_var_rlzd[secid] = residual_var_rlzd
                self.residual_var_pred[secid] = residual_var_pred
                self.residual_vol_rlzd[secid] = residual_vol_rlzd
                self.residual_vol_pred[secid] = residual_vol_pred


def transform_feature(
    feature: pd.DataFrame,
    tseq: tuple[callable, int, dict[str, Any]],
    mask: pd.DataFrame = None,
    apply_mask_at_stage: int = None
) -> pd.DataFrame:
    """
    Apply a sequence of transformations to a feature DataFrame with an optional mask.

    Args:
        feature: Feature DataFrame, index should be datetime and columns should be secids.
        tseq: Tuple of (transformation, axis, kwargs) where transformation is a callable and 
            axis is 0 or 1. Callables are applied in sequence to the feature DataFrame. kwargs is a dict of kwargs
            to be passed to the callable
        mask: Mask to apply to the feature DataFrame. Defaults to None. 
            If not None, then the mask should have the same index and columns as the feature DataFrame 
            but have boolean values.
        apply_mask_at_stage: Stage at which to apply the mask. Defaults to None. 
            If not None, then must be an int and should be less than the length of tseq.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    if apply_mask_at_stage is not None and (apply_mask_at_stage < 0 or apply_mask_at_stage >= len(tseq)):
        raise ValueError("apply_mask_at_stage must be within the range of tseq length")

    for i, (transformation, axis, kwargs) in enumerate(tseq):
        # Apply mask if specified
        if mask is not None and i == apply_mask_at_stage:
            feature = feature.mask(mask)

        # Apply transformation
        feature = feature.apply(transformation, axis=axis, **kwargs)

    # Apply mask after all transformations if apply_mask_at_stage is not specified
    if mask is not None and apply_mask_at_stage is None:
        feature = feature.mask(mask)

    return feature


def transform_feature(
    feature: pd.DataFrame,
    tseq: Tuple[Tuple[str, Callable, int, dict[str, Any]]],
    mask: pd.DataFrame = None,
    apply_mask_at_stage: str = None
) -> pd.DataFrame:
    """
    Apply a sequence of transformations to a feature DataFrame with an optional mask.

    Args:
        feature: Feature DataFrame, index should be datetime and columns should be secids.
        tseq: Tuple of tuples of (name, transformation, axis, kwargs) where transformation is a callable and 
            axis is 0 or 1. Callables are applied in sequence to the feature DataFrame. kwargs is a 
            dict of kwargs to be passed to the callable.
        mask: Mask to apply to the feature DataFrame. Defaults to None. 
            If not None, then the mask should have the same index and columns as the feature DataFrame 
            but have boolean values.
        apply_mask_at_stage: Stage at which to apply the mask. Defaults to None. 
            If not None, then must be a string and should correspond to the first value
            of at least one of the tuples in tseq. The mask will be applied prior to the 
            transformation of the specified stage.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    metadata = {
        'args': {
            'feature': feature.copy(),
            'tseq': tseq,
            'mask': mask,
            'apply_mask_at_stage': apply_mask_at_stage
        },
        'stages': {}
    }

    if mask is not None and apply_mask_at_stage is not None:
        if not any(name == apply_mask_at_stage for name, _, _, _ in tseq):
            raise ValueError(f"No transformation stage named '{apply_mask_at_stage}' found in tseq.")

    for name, transformation, axis, kwargs in tseq:
        if apply_mask_at_stage is not None and name == apply_mask_at_stage:
            feature = feature[mask]
        
        feature = feature.apply(transformation, axis=axis, **kwargs)

        metadata['stages'][name] = feature.copy()
    
    feature._transform_feature_metadata = metadata

    return feature



class Transformations:

    @staticmethod
    def zscore(x: pd.Series) -> pd.Series:
        return (x - x.mean()) / x.std()
    
    @staticmethod
    def winsorize(x: pd.Series, lower: float=0.01, upper: float=0.99) -> pd.Series:
        return x.clip(lower=x.quantile(lower), upper=x.quantile(upper))
    
    @staticmethod
    def scale_gmv(x: pd.Series, target_gmv: int=1) -> pd.Series:
        return x * target_gmv / x.abs().sum()
    
    @staticmethod
    def rank(x: pd.Series, ascending: bool=True) -> pd.Series:
        return x.rank(ascending=ascending)  


        # use the rolling window to calculate the betas
        # self.asset_bmk_betas = self.rets.apply(lambda x: x.cov(self.bmk_rets) / self.bmk_rets.var())
        # self.asset_bmk_betas = self.rets.rolling(risk_calculation_window).apply(lambda x: x.cov(self.bmk_rets) / self.bmk_rets.var())


def generate_ibis_universe():
    num_secids = 20
    wdw = 7
    rebal_freq = 7
    min_hist = 7

    unv = (
        ls.to_ibis('um.klines.monthly.1d')
        # add a 7 day rolling avg colum of quote_asset_volume
        .mutate(
            rolling_avg_quote_asset_volume=_.quote_asset_volume.mean().over(ibis.trailing_window(preceding=(wdw - 1), group_by=_.secid, order_by=_.close_time))
        )
        .group_by('close_time')
        .mutate(
            rank_asc=_.rolling_avg_quote_asset_volume.rank(),
            rank_desc=ibis.rank().over(order_by=ibis.desc(_.rolling_avg_quote_asset_volume))
        )
        .filter(_.rank_desc < num_secids)
        .group_by('close_time')
        .mutate(
            unv_wgt=_.rolling_avg_quote_asset_volume / _.rolling_avg_quote_asset_volume.sum(),
            unv_mask=True
        )
        # need to "shift forward" make columns for apply_close_time and apply_open_time that 
        # are equivalent to the next close_time and open_time for a given secid
        .group_by('secid')
        .order_by('close_time')
        .mutate(
            apply_open_time=_.open_time.lag(-1),
            apply_close_time=_.close_time.lag(-1),
        )
    )
    # unv

    return unv


def ibis_backtest():
    from ibis import _
    import ibis.selectors as s

    bwd = 1
    apply_universe = True

    feat = (
        t
        .filter(t.close_time < '2020-03-31')
        .order_by('secid', 'close_time')
        .group_by('secid')
        .order_by('close_time')
        .mutate(
            pct_change=((t.close / t.close.lag()) - 1),
            log_diff=(t.close.log() - t.close.lag().log()),
        )
        .mutate(
            feat=_.pct_change.sum().over(ibis.trailing_window(preceding=(bwd-1), group_by=_.secid, order_by=_.close_time)),
        )
        .select(['secid', 'open_time', 'close_time', 'close', 'pct_change', 'log_diff', 'feat'])
    )

    if apply_universe:
        feat = (
            feat
            .left_join(
                unv.select('unv_wgt', 'unv_mask', 'secid', 'apply_open_time', 'apply_close_time'), 
                [
                    feat.secid == unv.secid,
                    feat.open_time.date() == unv.apply_open_time.date(),
                    feat.close_time.date() == unv.apply_close_time.date()
                ]
            )
            .fillna(dict(
                unv_wgt=0,
                unv_mask=False
            ))
            .filter(_.unv_mask)
            .select(['secid', 'open_time', 'close_time', 'close', 'pct_change', 'log_diff', 'feat', 'unv_wgt', 'unv_mask'])
            
        )

    sig = (
        feat
        .group_by('close_time')
        .mutate(
            rank=_.feat.rank(),
        )
        .group_by('close_time')
        .mutate(
            zscore=(_.rank - _.rank.mean()) / _.rank.std()
        )
        .group_by('close_time')
        .mutate(
            wgt=_.zscore / _.zscore.abs().sum()
        )
        .group_by('close_time')
        .mutate(
            wgt=_.wgt * -1
        )
        .group_by('secid')
        .mutate(
            fwd_ret_1p=_.pct_change.lag(-1),
        )
    )

    pnl = (
        sig
        .group_by('secid')
        .mutate(
            pnl_gross=_.wgt * _.fwd_ret_1p,
            turnover=_.wgt - _.wgt.lag(),
            tcost_model=.0005
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
    # shrp = bts.mean() * np.sqrt(1440 * 365) / bts.std()



    # ibis.to_sql(expr)

    pnl


def ibis_xs_sig():
    import ibis
    from ibis import _

    t = ibis.table('bar')

    feat = (
        t
        .filter(t.close_time < '2020-01-31')
        .order_by('secid', 'close_time')
        .group_by('secid')
        .order_by('close_time')
        .mutate(
            pct_change=((t.close / t.close.lag()) - 1),
            log_diff=(t.close.log() - t.close.lag().log()),
        )
        .mutate(
            feat=_.pct_change.sum().over(ibis.trailing_window(preceding=4, group_by=_.secid, order_by=_.close_time)),
        )
        .select(['secid', 'close_time', 'close', 'pct_change', 'log_diff', 'feat'])
    )

    sig = (
        feat
        .group_by('close_time')
        .mutate(
            rank=_.feat.rank(),
        )
        .group_by('close_time')
        .mutate(
            zscore=(_.rank - _.rank.mean()) / _.rank.std()
        )
        .group_by('close_time')
        .mutate(
            wgt=_.zscore / _.zscore.abs().sum()
        )
        .group_by('close_time')
        .mutate(
            wgt=_.wgt * -1
        )
        .group_by('secid')
        .mutate(
            fwd_ret_1p=_.pct_change.lag(-1),
        )
    )

    pnl = (
        sig
        .group_by('secid')
        .mutate(
            pnl_gross=_.wgt * _.fwd_ret_1p,
            turnover=_.wgt - _.wgt.lag(),
            tcost_model=.0005
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

    bt_df = bt.select(['close_time', 'por_ret_gross', 'por_ret_net']).to_pandas().set_index('close_time')

    return dict(
        pnl=pnl,
        bt=bt,
        bt_df=bt_df
    )