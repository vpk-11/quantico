"""Phase 6/7: forward paper-trading loop on simulated money, and performance tracking.

Each trading day after 2026-01-02 (config `[phase6] start_date`), in order:
  1. execute yesterday's targets through the C++ simulator (arrival = today's open),
  2. mark the book to today's close,
  3. on a rebalance day, refit the baseline ridge on rows whose target window ended before today,
     predict today, and store long-only top-N targets for tomorrow.
Only data up to the day being processed is used, so replaying 2026 days is the same computation
a live daily run does. Bars after 2025-12-31 live in `live_bars`; Phases 1 to 3 stay frozen.
Positions and cash are derived from `paper_fills`, never stored separately.

Usage (run from pipeline/):
    python src/paper.py run      # pull new bars, process every unprocessed trading day
    python src/paper.py report   # write reports/phase7_report.md
    python src/paper.py reset    # drop the paper tables (live_bars is kept)
"""

from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
import statistics
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import backtest as bt
import baseline as bl
import execution as ex
import features as ft
import models
import research_store as rs
import trials
from evaluate import daily_ic, make_panel, newey_west_t, z_crit
from walkforward import LIQUIDITY, Scaler, build_target

_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = _ROOT / "reports" / "phase7_report.md"
PAPER_TABLES = ["paper_fills", "paper_nav", "paper_predictions", "paper_targets"]
ET = ZoneInfo("America/New_York")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS live_bars (
    ticker TEXT NOT NULL, date TEXT NOT NULL, open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL,
    close REAL NOT NULL, adj_close REAL NOT NULL, volume INTEGER NOT NULL, PRIMARY KEY (ticker, date));
CREATE TABLE IF NOT EXISTS paper_fills (
    date TEXT, ticker TEXT, side INTEGER, qty INTEGER, filled INTEGER, notional REAL, fees REAL, open REAL,
    mode TEXT NOT NULL DEFAULT 'replay');
CREATE TABLE IF NOT EXISTS paper_nav (date TEXT PRIMARY KEY, nav REAL, cash REAL, mode TEXT NOT NULL DEFAULT 'replay');
CREATE TABLE IF NOT EXISTS paper_predictions (
    date TEXT, ticker TEXT, prediction REAL, mode TEXT NOT NULL DEFAULT 'replay', PRIMARY KEY (date, ticker));
CREATE TABLE IF NOT EXISTS paper_targets (
    decided_on TEXT, ticker TEXT, shares INTEGER, executed INTEGER DEFAULT 0, mode TEXT NOT NULL DEFAULT 'replay');
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """Adds `mode` to any paper table created before it existed, backfilling existing rows as
    'replay' (every row stored before this phase's review predates `replay_end_date`). Idempotent."""
    for table in PAPER_TABLES:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
        if cols and "mode" not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN mode TEXT")
            conn.execute(f"UPDATE {table} SET mode = 'replay' WHERE mode IS NULL")
    conn.commit()


def mode_for(d: str, cfg: dict) -> str:
    """'replay' up to and including `[phase6] replay_end_date`, 'forward' after. Derived from the
    date being processed, never from clock time, so it never depends on when the job happens to run."""
    return "replay" if d <= cfg["phase6"]["replay_end_date"] else "forward"


def eligible_dates(days: list[str], now_et: dt.datetime) -> list[str]:
    """Keeps dates strictly before today in US/Eastern, or today if the ET clock reads 17:00 or
    later (so an in-progress trading day's bar is never processed as if it were final)."""
    today_et = now_et.date().isoformat()
    today_ok = now_et.hour >= 17
    return [d for d in days if d < today_et or (d == today_et and today_ok)]


def check_complete(conn: sqlite3.Connection, cfg: dict, d: str) -> bool:
    """Returns True if every basket ticker has a `live_bars` row for `d`. Returns False (skip,
    not halt) if none do, since that means `d` is not a trading day or nothing has been pulled
    for it yet, not a bad partial pull. Halts (raises) if some tickers have a row and others do
    not, which is a real data problem; never forward-fills."""
    have = conn.execute("SELECT COUNT(DISTINCT ticker) FROM live_bars WHERE date = ?", (d,)).fetchone()[0]
    want = len(rs.basket(cfg))
    if have == 0:
        return False
    if have != want:
        raise RuntimeError(f"{d}: {have} of {want} basket tickers have a live_bars row; halting, no forward-fill")
    return True


def sync_live(conn: sqlite3.Connection, cfg: dict, today: str) -> None:
    """Pull bars since `live_pull_from`, check the overlap with daily_bars, store rows after its last date.

    Today's bar is excluded: it may be incomplete until the close.
    """
    last_hist = conn.execute("SELECT MAX(date) FROM daily_bars").fetchone()[0]
    for ticker, _ in rs.basket(cfg):
        df, _ = rs.clean(rs.pull(ticker, cfg["phase6"]["live_pull_from"], today), ticker, cfg["data"]["max_gap_days"])
        df = df[df.index.strftime("%Y-%m-%d") < today]
        overlap = df[df.index.strftime("%Y-%m-%d") <= last_hist]
        for idx, close in overlap["close"].items():
            old = conn.execute("SELECT close FROM daily_bars WHERE ticker=? AND date=?", (ticker, idx.strftime("%Y-%m-%d"))).fetchone()
            if old is None or abs(old[0] / close - 1) > 1e-4:
                raise RuntimeError(f"{ticker} {idx.date()}: live close {close} disagrees with stored {old}")
        new = df[df.index.strftime("%Y-%m-%d") > last_hist]
        conn.executemany(
            "INSERT OR REPLACE INTO live_bars VALUES (?,?,?,?,?,?,?,?)",
            [(ticker, i.strftime("%Y-%m-%d"), r.open, r.high, r.low, r.close, r.adj_close, int(round(r.volume)))
             for i, r in new.iterrows()])
    conn.commit()


def positions(conn: sqlite3.Connection, capital: float) -> tuple[dict[str, int], float]:
    pos, cash = {}, float(capital)
    for t, s, f, n, fee in conn.execute("SELECT ticker, side, filled, notional, fees FROM paper_fills"):
        pos[t] = pos.get(t, 0) + s * f
        cash -= s * n + fee
    return {t: q for t, q in pos.items() if q}, cash


def predict(conn: sqlite3.Connection, cfg: dict, d: str) -> pd.Series:
    cfg3, cols = cfg["phase3"], bl.feature_cols(cfg)
    bars = pd.read_sql_query(
        "SELECT * FROM (SELECT ticker, date, close, volume FROM daily_bars UNION ALL "
        "SELECT ticker, date, close, volume FROM live_bars) WHERE date <= ? ORDER BY ticker, date", conn, params=(d,))
    parts = []
    for t, g in bars.groupby("ticker", sort=False):
        b = g.set_index(pd.to_datetime(g["date"]))
        f = ft.compute_features(b, cfg["features"]["momentum_windows_days"], cfg["features"]["mean_reversion_window_days"])
        parts.append(pd.concat([g.reset_index(drop=True), f.reset_index(drop=True)], axis=1))
    df = build_target(pd.concat(parts, ignore_index=True), cfg3["target_horizon_days"])
    ok = df[cols].notna().all(axis=1)
    train, test = df[ok & df["target"].notna() & (df["target_end"] < d)], df[ok & (df["date"] == d)]
    j = cols.index(LIQUIDITY)

    def x(frame: pd.DataFrame) -> np.ndarray:
        a = frame[cols].to_numpy(float, copy=True)
        a[:, j] = np.log(a[:, j])
        return a

    sc = Scaler().fit(x(train))
    m = models.make("ridge", cfg3).fit(sc.transform(x(train)), train["target"].to_numpy(float))
    return pd.Series(m.predict(sc.transform(x(test))), index=test["ticker"].to_numpy())


def process_day(conn: sqlite3.Connection, cfg: dict, d: str, i: int) -> None:
    p, p6 = ex.cost_params(cfg["phase4"]), cfg["phase6"]
    mode = mode_for(d, cfg)
    bars_d = pd.read_sql_query("SELECT * FROM live_bars WHERE date = ?", conn, index_col="ticker", params=(d,))
    pos, cash = positions(conn, p["capital"])
    pending = conn.execute("SELECT ticker, shares FROM paper_targets WHERE executed = 0 AND decided_on < ?", (d,)).fetchall()
    if pending:
        tgt = dict(pending)
        trades = {t: tgt.get(t, 0) - pos.get(t, 0) for t in set(tgt) | set(pos)}
        fills = bt.execute_day(trades, bars_d, ex.STRATEGIES[p6["paper_execution_strategy"]], p)
        conn.executemany("INSERT INTO paper_fills VALUES (?,?,?,?,?,?,?,?,?)",
                         [(d, f["ticker"], f["side"], f["qty"], f["filled"], f["notional"], f["fees"], f["open"], mode) for f in fills])
        conn.execute("UPDATE paper_targets SET executed = 1 WHERE executed = 0")
        pos, cash = positions(conn, p["capital"])
    nav = cash + sum(q * bars_d.at[t, "close"] for t, q in pos.items())
    conn.execute("INSERT INTO paper_nav VALUES (?,?,?,?)", (d, nav, cash, mode))
    if i % cfg["phase3"]["target_horizon_days"] == 0:
        pred = predict(conn, cfg, d)
        conn.executemany("INSERT INTO paper_predictions VALUES (?,?,?,?)", [(d, t, float(v), mode) for t, v in pred.items()])
        tgt = bt.top_n_targets(pred, bars_d["close"], p)
        conn.executemany("INSERT INTO paper_targets (decided_on, ticker, shares, mode) VALUES (?,?,?,?)",
                         [(d, t, s, mode) for t, s in tgt.items()])
    conn.commit()


def _process_pending(conn: sqlite3.Connection, cfg: dict, days: list[str]) -> tuple[int, int]:
    """Processes every date in `days` not already in `paper_nav`, in order. A date with no bars
    at all is skipped (not a trading day, or nothing pulled yet) and the loop continues; a date
    with bars for only some tickers halts (propagates), so later dates are never processed ahead
    of a genuinely incomplete one. Returns (total days considered, new days processed)."""
    done = {r[0] for r in conn.execute("SELECT date FROM paper_nav")}
    new = 0
    for i, d in enumerate(days):
        if d in done:
            continue
        if not check_complete(conn, cfg, d):
            continue  # skip: no bars at all for this date, not a halt
        process_day(conn, cfg, d, i)
        new += 1
    return len(days), new


def run(cfg: dict, clock=lambda: dt.datetime.now(ET)) -> int:
    conn = sqlite3.connect(_ROOT / cfg["data"]["db_path"])
    conn.executescript(_SCHEMA)
    _migrate(conn)
    try:
        sync_live(conn, cfg, dt.date.today().isoformat())
        all_days = [r[0] for r in conn.execute(
            "SELECT DISTINCT date FROM live_bars WHERE date >= ? ORDER BY date", (cfg["phase6"]["start_date"],))]
        days = eligible_dates(all_days, clock())
        total, new = _process_pending(conn, cfg, days)
        nav = conn.execute("SELECT date, nav FROM paper_nav ORDER BY date DESC LIMIT 1").fetchone()
        print(f"processed through {nav[0]}, NAV {nav[1]:,.0f}, {total} trading days, {new} new")
    finally:
        conn.close()
    return report(cfg)


def edge_rule(
    n_rebalances: int, mean_ic: float, nw_t: float, z: float, strategy_sharpe: float,
    null_sharpes: list[float], cfg7: dict,
) -> tuple[bool, str]:
    """Frozen before any forward data exists. 'Edge supported' only if: at least
    `min_forward_rebalances`, forward mean IC above 0 with Newey-West t at or above the Phase 3
    G1 critical value at the current trial count k, and the strategy Sharpe at or above the
    `null_percentile_threshold` percentile of the random-top-N null. Otherwise 'no edge
    established'. Pure function: every input is already computed, so this is unit-testable
    without touching the database.
    """
    if n_rebalances < cfg7["min_forward_rebalances"]:
        return False, f"insufficient sample: {n_rebalances} forward rebalances, need {cfg7['min_forward_rebalances']}"
    cond_ic = mean_ic > 0 and nw_t >= z
    pct = float((np.asarray(null_sharpes) < strategy_sharpe).mean() * 100) if len(null_sharpes) else 0.0
    cond_null = pct >= cfg7["null_percentile_threshold"]
    if cond_ic and cond_null:
        return True, "edge supported"
    return False, "no edge established"


def _mode_bounds(nav_all: pd.DataFrame, mode: str, capital: float) -> tuple[str, str, float, float, pd.DataFrame] | None:
    """(prior_date, last_date, nav_at_prior_date, nav_at_last_date, this mode's nav rows), sorted
    by date. `nav_at_prior_date` is the carried-in NAV from the trading day right before this
    mode started (starting `capital` if this mode is the very first data), so attribution can
    measure the change *during* the mode without pooling it with the other mode's days."""
    sub = nav_all[nav_all["mode"] == mode].sort_values("date")
    if sub.empty:
        return None
    first_date = sub["date"].iloc[0]
    prior = nav_all[nav_all["date"] < first_date]
    prior_date = prior["date"].iloc[-1] if not prior.empty else first_date
    nav0 = float(prior["nav"].iloc[-1]) if not prior.empty else capital
    return prior_date, sub["date"].iloc[-1], nav0, float(sub["nav"].iloc[-1]), sub


def attribution(mode_fills: pd.DataFrame, ew_nav: pd.Series, nav0: float, nav1: float, prior_date: str, last_date: str) -> dict:
    """Total NAV change = market + selection + costs, each independently computed except
    selection (the residual: what stock picking added or cost beyond just holding the market).
    `costs` is signed negative (it reduces NAV). The three parts sum to `total` exactly:
    frictionless = total + costs_paid; selection = frictionless - market;
    market + selection + costs = market + frictionless - market - costs_paid = total.
    """
    costs_paid = float((mode_fills["side"] * (mode_fills["notional"] - mode_fills["filled"] * mode_fills["open"])
                        + mode_fills["fees"]).sum()) if not mode_fills.empty else 0.0
    ew0 = float(ew_nav.loc[prior_date]) if prior_date in ew_nav.index else float(ew_nav.iloc[0])
    ew1 = float(ew_nav.loc[last_date])
    market = ew1 - ew0
    total = nav1 - nav0
    frictionless = total + costs_paid
    selection = frictionless - market
    return {"total": total, "market": market, "selection": selection, "costs": -costs_paid}


def _position_as_of(fills_all: pd.DataFrame, ticker: str, as_of_date: str) -> int:
    tf = fills_all[(fills_all["ticker"] == ticker) & (fills_all["date"] <= as_of_date)]
    return int((tf["side"] * tf["filled"]).sum())


def sector_pnl(
    fills_all: pd.DataFrame, mode_fills: pd.DataFrame, close: pd.DataFrame, basket: list[tuple[str, str]],
    prior_date: str, last_date: str,
) -> pd.Series:
    """Per-ticker frictionless P&L (position mark-to-market plus frictionless trade cash flow),
    grouped by sector. Sums to the same `selection + market` (the frictionless change) as
    `attribution`'s residual, which is an exact accounting identity, checked in tests."""
    rows = []
    for ticker, sector in basket:
        pos0, pos1 = _position_as_of(fills_all, ticker, prior_date), _position_as_of(fills_all, ticker, last_date)
        price0 = float(close.at[prior_date, ticker]) if prior_date in close.index else float(close[ticker].iloc[0])
        price1 = float(close.at[last_date, ticker])
        tf = mode_fills[mode_fills["ticker"] == ticker]
        frictionless_cash = float((-tf["side"] * tf["filled"] * tf["open"]).sum()) if not tf.empty else 0.0
        rows.append({"sector": sector, "pnl": pos1 * price1 - pos0 * price0 + frictionless_cash})
    return pd.DataFrame(rows).groupby("sector")["pnl"].sum()


def mode_comparators(full_bars: pd.DataFrame, close: pd.DataFrame, mode_preds: pd.DataFrame, p: dict, cfg7: dict) -> tuple[float, list[float]]:
    """Equal-weight-20 Sharpe and the random-top-N null Sharpe distribution, both over exactly
    `mode_preds`'s rebalance dates. `mode_preds` already holds one row set per rebalance (Phase 6
    stores predictions only on rebalance days), so `rebalance_days=1` keeps every one of them."""
    pred_by_day = {d: g.set_index("ticker")["prediction"] for d, g in mode_preds.groupby("date")}
    exec_for = bt.rebalance_schedule(close, mode_preds, rebalance_days=1)
    null_sharpes = bt.random_null_sharpes(full_bars, close, pred_by_day, exec_for, ex.STRATEGIES["immediate"],
                                           p, cfg7["null_draws"], cfg7["null_seed"])
    dates = sorted(exec_for)
    ew = close.loc[dates].div(close.loc[dates[0]]).mean(axis=1) * p["capital"]
    r = ew.pct_change().dropna()
    ew_sharpe = float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 1 and r.std() > 0 else float("nan")
    return ew_sharpe, null_sharpes


def _ic_series(preds: pd.DataFrame, realized: pd.DataFrame, min_tickers: int) -> pd.Series:
    """Daily rank IC indexed by date, for whatever rows `preds` holds (already mode-filtered)."""
    if preds.empty:
        return pd.Series(dtype=float)
    f = preds.merge(realized, on=["ticker", "date"]).dropna(subset=["target"])
    if f.empty:
        return pd.Series(dtype=float)
    panel = make_panel(f["date"].to_numpy(), f["ticker"].to_numpy())
    ic = daily_ic(f["prediction"].to_numpy(), f["target"].to_numpy(), panel, min_tickers)
    s = pd.Series(ic, index=panel.dates)
    return s[~s.isna()]


def render_mode(
    mode: str, nav_all: pd.DataFrame, fills_all: pd.DataFrame, preds_all: pd.DataFrame, realized: pd.DataFrame,
    full_bars: pd.DataFrame, close: pd.DataFrame, ew_nav: pd.Series, by_year: pd.DataFrame, cfg: dict, p: dict,
    k: int, z: float,
) -> list[str]:
    cfg3, cfg7 = cfg["phase3"], cfg["phase7"]
    L = [f"## {mode.capitalize()}", ""]
    bounds = _mode_bounds(nav_all, mode, p["capital"])
    if bounds is None:
        L += [f"No {mode} data yet.", ""]
        return L
    prior_date, last_date, nav0, nav1, nav_rows = bounds
    mode_fills = fills_all[fills_all["mode"] == mode]
    mode_preds = preds_all[preds_all["mode"] == mode][["date", "ticker", "prediction"]]
    n_rebalances = mode_preds["date"].nunique()
    ic_series = _ic_series(mode_preds, realized, cfg3["min_tickers_per_day"])
    mean_ic = float(ic_series.mean()) if len(ic_series) else float("nan")
    nw_t = float(newey_west_t(ic_series.to_numpy(), 1)) if len(ic_series) > 3 else float("nan")

    nav_series = nav_rows.set_index("date")["nav"]
    r = nav_series.pct_change().dropna()
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 1 and r.std() > 0 else float("nan")
    dd = float((nav_series / nav_series.cummax() - 1.0).min())

    L += [f"{nav_rows['date'].iloc[0]} to {last_date} ({len(nav_rows)} trading days, {n_rebalances} rebalances, "
          f"{len(ic_series)} with a matured {cfg3['target_horizon_days']}-day outcome).", "",
          "| metric | value |", "|---|---|",
          f"| NAV start / end | {nav0:,.0f} / {nav1:,.0f} |",
          f"| return | {nav1 / nav0 - 1:.2%} |",
          f"| Sharpe (annualized, daily) | {sharpe:.2f} |",
          f"| max drawdown | {dd:.2%} |",
          f"| forward rank IC | mean {mean_ic:.4f}" + (f", Newey-West t {nw_t:.2f} |" if not np.isnan(nw_t) else " |"), ""]

    if mode == "forward" and n_rebalances < cfg7["min_forward_rebalances"]:
        L += [f"Insufficient sample: {n_rebalances} forward rebalances, need "
              f"{cfg7['min_forward_rebalances']}. No claim on edge or decay.", ""]
        return L

    att = attribution(mode_fills, ew_nav, nav0, nav1, prior_date, last_date)
    L += ["### Attribution (USD)", "",
          "| component | value |", "|---|---|",
          f"| market (equal-weight 20) | {att['market']:,.0f} |",
          f"| selection | {att['selection']:,.0f} |",
          f"| costs | {att['costs']:,.0f} |",
          f"| total NAV change | {att['total']:,.0f} |", ""]

    pnl = sector_pnl(fills_all, mode_fills, close, rs.basket(cfg), prior_date, last_date)
    L += ["### Sector contribution (frictionless P&L, USD)", "", "| sector | pnl |", "|---|---|"]
    for sector, v in pnl.sort_values(ascending=False).items():
        L.append(f"| {sector} | {v:,.0f} |")
    L.append("")

    if len(ic_series) >= 2:
        window = min(cfg7["decay_window_rebalances"], len(ic_series))
        roll = float(ic_series.iloc[-window:].mean())
        roll_t = float(newey_west_t(ic_series.iloc[-window:].to_numpy(), 1)) if window > 3 else float("nan")
        x = np.arange(len(ic_series))
        y = ic_series.to_numpy()
        slope = float(np.polyfit(x, y, 1)[0])
        band_lo, band_hi = float(by_year["mean"].min()), float(by_year["mean"].max())
        inside = band_lo <= mean_ic <= band_hi
        L += ["### Signal decay", "",
              f"Rolling {window}-rebalance IC: {roll:.4f}" + (f", Newey-West t {roll_t:.2f}." if not np.isnan(roll_t) else "."),
              f"Trend (IC vs rebalance index): slope {slope:.6f} ({'rising' if slope > 0 else 'falling'}).",
              f"Phase 3 out-of-sample yearly IC band: {band_lo:.4f} to {band_hi:.4f}. "
              f"{mode.capitalize()} mean IC is {'inside' if inside else 'outside'} that band.", ""]

    ew_sharpe, null_sharpes = mode_comparators(full_bars, close, mode_preds, p, cfg7)
    pct = float((np.asarray(null_sharpes) < sharpe).mean() * 100) if null_sharpes else float("nan")
    L += ["### Comparators", "",
          f"Equal-weight buy-and-hold Sharpe over the same dates: {ew_sharpe:.2f} "
          "(flattered by survivorship bias: all 20 names survived to 2026).",
          f"Random top-{p['top_n']} null, {len(null_sharpes)} seeded draws, same rebalance schedule "
          f"and cost model: Sharpe mean {statistics.mean(null_sharpes):.2f}, range "
          f"{min(null_sharpes):.2f} to {max(null_sharpes):.2f}. Strategy Sharpe sits at the "
          f"{pct:.0f}th percentile.", ""]

    if mode == "forward":
        supported, label = edge_rule(n_rebalances, mean_ic, nw_t, z, sharpe, null_sharpes, cfg7)
        L += [f"**Edge claim: {label}.**", ""]
    return L


def report(cfg: dict) -> int:
    p, conn = ex.cost_params(cfg["phase4"]), sqlite3.connect(_ROOT / cfg["data"]["db_path"])
    _migrate(conn)
    try:
        verdict, label = bt.phase3_label(conn, cfg)
        k = trials.count_k(conn)
        nav_all = pd.read_sql_query("SELECT date, nav, mode FROM paper_nav ORDER BY date", conn)
        fills_all = pd.read_sql_query("SELECT * FROM paper_fills", conn)
        preds_all = pd.read_sql_query("SELECT * FROM paper_predictions", conn)
        base = pd.read_sql_query("SELECT ticker, date, prediction, target FROM predictions WHERE model_id = ?", conn, params=(bt.MODEL_ID,))
        bars_close = pd.read_sql_query("SELECT ticker, date, close FROM daily_bars UNION ALL SELECT ticker, date, close FROM live_bars "
                                       "ORDER BY date, ticker", conn)
        full_bars = pd.read_sql_query(
            "SELECT ticker, date, open, high, low, close, volume FROM daily_bars UNION ALL "
            "SELECT ticker, date, open, high, low, close, volume FROM live_bars ORDER BY date, ticker",
            conn).set_index(["date", "ticker"])
        last_data_date = conn.execute("SELECT MAX(date) FROM paper_nav").fetchone()[0]
    finally:
        conn.close()
    z = z_crit(cfg["phase3"]["significance_alpha"] / k)
    realized = build_target(bars_close, cfg["phase3"]["target_horizon_days"])[["ticker", "date", "target"]]
    close = bars_close.set_index(["date", "ticker"])["close"].unstack()
    min_t = cfg["phase3"]["min_tickers_per_day"]

    if nav_all.empty:
        ew_nav = pd.Series(dtype=float)
    else:
        paper_dates = sorted(nav_all["date"].unique())
        ew_nav = close.loc[paper_dates].div(close.loc[paper_dates[0]]).mean(axis=1) * p["capital"]

    pan_b = make_panel(base["date"].to_numpy(), base["ticker"].to_numpy())
    ic_b = daily_ic(base["prediction"].to_numpy(), base["target"].to_numpy(), pan_b, min_t)
    by_year = pd.Series(ic_b, index=pan_b.years).groupby(level=0).agg(["mean", "count"])

    L = ["# Phase 6/7 report: paper trading loop and performance tracking", "",
         f"Phase 3 verdict: **{verdict}** ({label}). Replay and forward are reported separately and "
         "never pooled; no edge claim is made unless the rule below passes.", "",
         f"Last updated: {last_data_date} (from the data, not the clock). Paper money only. Signal: "
         "baseline ridge refit at each rebalance. Execution: C++ simulator, "
         f"`{cfg['phase6']['paper_execution_strategy']}`. Long-only top {p['top_n']} of 20, fixed "
         f"${p['capital']:,}, rebalance every {cfg['phase3']['target_horizon_days']} trading days. "
         "Intraday slices are synthetic (see phase5_report.md).", "",
         "Limits: survivorship bias (all 20 names survived to 2026), price basis is `close` (no "
         "dividends), cost model is assumed and uncalibrated, the window is short, and dates up to "
         f"`[phase6] replay_end_date` are a pseudo-live replay, not true paper trading.", ""]
    for mode in ("replay", "forward"):
        L += render_mode(mode, nav_all, fills_all, preds_all, realized, full_bars, close, ew_nav, by_year, cfg, p, k, z)

    L += ["## Phase 3 out-of-sample yearly IC (for comparison)", "", "| year | mean IC | days |", "|---|---|---|"]
    for y, row in by_year.iterrows():
        L.append(f"| {y} | {row['mean']:.4f} | {int(row['count'])} |")
    L += ["", "Yearly IC flips sign (positive 2016 to 2018 and 2023 to 2025, negative 2019 to 2022), "
          "which is what an edge that never cleared its significance gates looks like.", ""]
    REPORT_PATH.write_text("\n".join(L))
    print(f"report: {REPORT_PATH}")
    return 0


def reset(cfg: dict, include_forward: bool = False) -> int:
    """Drops the 4 paper tables (`live_bars` is kept). Refuses if any row is `mode='forward'`
    (real paper trades, not replay) unless `include_forward` is set, so a reset can't silently
    discard forward history."""
    conn = sqlite3.connect(_ROOT / cfg["data"]["db_path"])
    try:
        _migrate(conn)
        if not include_forward:
            for t in PAPER_TABLES:
                cols = [r[1] for r in conn.execute(f"PRAGMA table_info({t})")]
                if "mode" not in cols:
                    continue
                n = conn.execute(f"SELECT COUNT(*) FROM {t} WHERE mode = 'forward'").fetchone()[0]
                if n:
                    raise RuntimeError(f"{t} has {n} forward row(s); refusing to reset. "
                                        "Pass --include-forward to drop them anyway.")
        for t in PAPER_TABLES:
            conn.execute(f"DROP TABLE IF EXISTS {t}")
        conn.commit()
    finally:
        conn.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quantico paper trading loop (Phase 6/7)")
    parser.add_argument("cmd", choices=["run", "report", "reset"])
    parser.add_argument("--include-forward", action="store_true",
                        help="reset: allow dropping forward (real paper trade) rows")
    args = parser.parse_args(argv)
    cfg = rs.load_config()
    if args.cmd == "reset":
        return reset(cfg, include_forward=args.include_forward)
    return {"run": run, "report": report}[args.cmd](cfg)


if __name__ == "__main__":
    sys.exit(main())
