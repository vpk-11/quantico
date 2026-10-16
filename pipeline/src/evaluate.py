"""Evaluator: daily IC, Newey-West t-stat, R^2 vs naive, the five gates, extras.

Phase 3 scope. Pure functions over arrays. A `Panel` maps the flat test rows
(one per ticker-date) onto a (dates x tickers) grid so the daily rank
correlation is one vectorized pass, which keeps the 200-run permutation test
cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm


@dataclass
class Panel:
    date_code: np.ndarray  # per flat row: index into `dates`
    tick_code: np.ndarray  # per flat row: index into the ticker axis
    dates: np.ndarray  # unique sorted 'YYYY-MM-DD'
    n_tickers: int
    years: np.ndarray  # calendar year of each entry in `dates`


def make_panel(dates: np.ndarray, tickers: np.ndarray) -> Panel:
    ud, dc = np.unique(dates, return_inverse=True)
    ut, tc = np.unique(tickers, return_inverse=True)
    return Panel(dc, tc, ud, len(ut), np.array([int(d[:4]) for d in ud]))


def _grid(values: np.ndarray, panel: Panel) -> np.ndarray:
    g = np.full((len(panel.dates), panel.n_tickers), np.nan)
    g[panel.date_code, panel.tick_code] = values
    return g


def daily_ic(pred: np.ndarray, target: np.ndarray, panel: Panel, min_tickers: int) -> np.ndarray:
    """Spearman correlation of pred vs target across tickers, one value per date.

    NaN on dates with fewer than `min_tickers` non-null tickers.
    """
    P, T = _grid(pred, panel), _grid(target, panel)
    mask = ~np.isnan(P)
    cnt = mask.sum(axis=1)
    RP = pd.DataFrame(P).rank(axis=1).to_numpy()
    RT = pd.DataFrame(T).rank(axis=1).to_numpy()
    denom = np.maximum(cnt, 1)
    dp = np.where(mask, RP - (np.where(mask, RP, 0).sum(axis=1) / denom)[:, None], 0.0)
    dt = np.where(mask, RT - (np.where(mask, RT, 0).sum(axis=1) / denom)[:, None], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = (dp * dt).sum(axis=1) / np.sqrt((dp**2).sum(axis=1) * (dt**2).sum(axis=1))
    ic[cnt < min_tickers] = np.nan
    return ic


def mean_ic(pred: np.ndarray, target: np.ndarray, panel: Panel, min_tickers: int) -> float:
    return float(np.nanmean(daily_ic(pred, target, panel, min_tickers)))


def newey_west_t(x: np.ndarray, lags: int) -> float:
    """t-stat of the mean of `x` with a Newey-West (Bartlett) standard error."""
    fit = sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.tvalues[0])


def z_crit(alpha: float) -> float:
    """Two-sided normal critical value at significance `alpha`."""
    return float(norm.ppf(1 - alpha / 2))


def r2_vs_naive(y: np.ndarray, pred: np.ndarray, naive: np.ndarray) -> float:
    return 1.0 - float(((y - pred) ** 2).sum() / ((y - naive) ** 2).sum())


def bootstrap_lower(ic: np.ndarray, block: int, resamples: int, alpha: float, seed: int) -> float:
    """Lower bound of the two-sided (1 - alpha) block-bootstrap interval for mean IC."""
    rng = np.random.default_rng(seed)
    n = len(ic)
    n_blocks = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=(resamples, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(resamples, -1)[:, :n]
    return float(np.quantile(ic[idx].mean(axis=1), alpha / 2))


@dataclass
class Score:
    ic: np.ndarray  # per panel date, NaN where below min_tickers
    mean_ic: float
    nw_t: float
    year_ic: dict[int, float]
    r2: float


def score(pred, y, naive, panel: Panel, cfg: dict) -> Score:
    ic = daily_ic(pred, y, panel, cfg["min_tickers_per_day"])
    ok = ~np.isnan(ic)
    v, yrs = ic[ok], panel.years[ok]
    return Score(
        ic=ic,
        mean_ic=float(v.mean()),
        nw_t=newey_west_t(v, cfg["newey_west_lags"]),
        year_ic={int(yr): float(v[yrs == yr].mean()) for yr in np.unique(yrs)},
        r2=r2_vs_naive(y, pred, naive),
    )


@dataclass
class Gate:
    id: str
    name: str
    value: str
    threshold: str
    passed: bool


def gates(sc: Score, null_means: np.ndarray, k: int, cfg: dict) -> list[Gate]:
    alpha = cfg["significance_alpha"] / k
    z = z_crit(alpha)
    ic = sc.ic[~np.isnan(sc.ic)]
    pos = sum(m > 0 for m in sc.year_ic.values())
    p = (1 + int((null_means >= sc.mean_ic).sum())) / (len(null_means) + 1)
    lo = bootstrap_lower(
        ic, cfg["bootstrap_block_days"], cfg["bootstrap_resamples"], alpha, cfg["random_seed"]
    )
    return [
        Gate("G1", "IC significance", f"mean IC {sc.mean_ic:.5f}, NW t {sc.nw_t:.3f}",
             f"mean IC > 0 and t >= {z:.3f}", sc.mean_ic > 0 and sc.nw_t >= z),
        Gate("G2", "Beats naive", f"R2 vs naive {sc.r2:.6f}", "> 0", sc.r2 > 0),
        Gate("G3", "Consistency", f"{pos} of {len(sc.year_ic)} years positive",
             f">= {cfg['min_positive_years']}", pos >= cfg["min_positive_years"]),
        Gate("G4", "Permutation test", f"p {p:.5f} ({len(null_means)} shuffles)",
             f"< {alpha:.5f}", p < alpha),
        Gate("G5", "Block bootstrap", f"lower bound {lo:.5f}",
             f"> 0 at {1 - alpha:.4f} two-sided", lo > 0),
    ]


def null_centered(null_means: np.ndarray) -> tuple[bool, str]:
    """I4: the shuffled-target mean-IC distribution must be centered on zero."""
    m, s, n = float(null_means.mean()), float(null_means.std(ddof=1)), len(null_means)
    bound = 3 * s / np.sqrt(n)
    return abs(m) < bound, f"|mean| {abs(m):.6f} < 3*std/sqrt(n) {bound:.6f}"


def quintile_spread(pred, y, panel: Panel, min_tickers: int) -> np.ndarray:
    """Daily mean realized target of the top fifth of predictions minus the bottom fifth."""
    P, T = _grid(pred, panel), _grid(y, panel)
    n = (~np.isnan(P)).sum(axis=1)
    q = np.maximum(n // 5, 1)
    R = pd.DataFrame(P).rank(axis=1, method="first").to_numpy()
    top, bot = R > (n - q)[:, None], R <= q[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (np.where(top, T, 0).sum(axis=1) - np.where(bot, T, 0).sum(axis=1)) / q
    out[n < min_tickers] = np.nan
    return out


def extras(pred, y, bench_pred, coefs: dict[int, np.ndarray], sc: Score, panel: Panel, cfg: dict) -> dict:
    """Reported-only numbers: no gate depends on these."""
    qs = quintile_spread(pred, y, panel, cfg["min_tickers_per_day"])
    qs = qs[~np.isnan(qs)]
    bic = daily_ic(bench_pred, y, panel, cfg["min_tickers_per_day"])
    bic = bic[~np.isnan(bic)]
    ic = sc.ic[~np.isnan(sc.ic)]
    return {
        "year_ic": sc.year_ic,
        "hit_rate": float((np.sign(pred) == np.sign(y)).mean()),
        "quintile_mean": float(qs.mean()),
        "quintile_t": newey_west_t(qs, cfg["newey_west_lags"]),
        "bench_mean_ic": float(bic.mean()),
        "bench_t": newey_west_t(bic, cfg["newey_west_lags"]),
        "coefs": coefs,
        "ic_quantiles": {q: float(np.quantile(ic, q)) for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)},
        "ic_std": float(ic.std(ddof=1)),
        "n_days": int(len(ic)),
    }
