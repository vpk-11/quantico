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
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
import baseline as bl
import execution as ex
import features as ft
import models
import research_store as rs
from evaluate import daily_ic, make_panel, newey_west_t
from walkforward import LIQUIDITY, Scaler, build_target

_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = _ROOT / "reports" / "phase7_report.md"
PAPER_TABLES = ["paper_fills", "paper_nav", "paper_predictions", "paper_targets"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS live_bars (
    ticker TEXT NOT NULL, date TEXT NOT NULL, open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL,
    close REAL NOT NULL, adj_close REAL NOT NULL, volume INTEGER NOT NULL, PRIMARY KEY (ticker, date));
CREATE TABLE IF NOT EXISTS paper_fills (
    date TEXT, ticker TEXT, side INTEGER, qty INTEGER, filled INTEGER, notional REAL, fees REAL, open REAL);
CREATE TABLE IF NOT EXISTS paper_nav (date TEXT PRIMARY KEY, nav REAL, cash REAL);
CREATE TABLE IF NOT EXISTS paper_predictions (date TEXT, ticker TEXT, prediction REAL, PRIMARY KEY (date, ticker));
CREATE TABLE IF NOT EXISTS paper_targets (decided_on TEXT, ticker TEXT, shares INTEGER, executed INTEGER DEFAULT 0);
"""


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
    p, p6 = cfg["phase4"], cfg["phase6"]
    bars_d = pd.read_sql_query("SELECT * FROM live_bars WHERE date = ?", conn, index_col="ticker", params=(d,))
    pos, cash = positions(conn, p["capital"])
    pending = conn.execute("SELECT ticker, shares FROM paper_targets WHERE executed = 0 AND decided_on < ?", (d,)).fetchall()
    if pending:
        tgt = dict(pending)
        trades = {t: tgt.get(t, 0) - pos.get(t, 0) for t in set(tgt) | set(pos)}
        fills = bt.execute_day(trades, bars_d, ex.STRATEGIES[p6["strategy"]], p)
        conn.executemany("INSERT INTO paper_fills VALUES (?,?,?,?,?,?,?,?)",
                         [(d, f["ticker"], f["side"], f["qty"], f["filled"], f["notional"], f["fees"], f["open"]) for f in fills])
        conn.execute("UPDATE paper_targets SET executed = 1 WHERE executed = 0")
        pos, cash = positions(conn, p["capital"])
    nav = cash + sum(q * bars_d.at[t, "close"] for t, q in pos.items())
    conn.execute("INSERT INTO paper_nav VALUES (?,?,?)", (d, nav, cash))
    if i % p["rebalance_days"] == 0:
        pred = predict(conn, cfg, d)
        conn.executemany("INSERT INTO paper_predictions VALUES (?,?,?)", [(d, t, float(v)) for t, v in pred.items()])
        tgt = bt.top_n_targets(pred, bars_d["close"], p)
        conn.executemany("INSERT INTO paper_targets (decided_on, ticker, shares) VALUES (?,?,?)", [(d, t, s) for t, s in tgt.items()])
    conn.commit()


def run(cfg: dict) -> int:
    conn = sqlite3.connect(_ROOT / cfg["data"]["db_path"])
    conn.executescript(_SCHEMA)
    try:
        sync_live(conn, cfg, dt.date.today().isoformat())
        days = [r[0] for r in conn.execute(
            "SELECT DISTINCT date FROM live_bars WHERE date >= ? ORDER BY date", (cfg["phase6"]["start_date"],))]
        done = {r[0] for r in conn.execute("SELECT date FROM paper_nav")}
        for i, d in enumerate(days):
            if d not in done:
                process_day(conn, cfg, d, i)
        nav = conn.execute("SELECT date, nav FROM paper_nav ORDER BY date DESC LIMIT 1").fetchone()
        print(f"processed through {nav[0]}, NAV {nav[1]:,.0f}, {len(days)} trading days, {len(days) - len(done)} new")
    finally:
        conn.close()
    return report(cfg)


def report(cfg: dict) -> int:
    p, conn = cfg["phase4"], sqlite3.connect(_ROOT / cfg["data"]["db_path"])
    try:
        nav = pd.read_sql_query("SELECT date, nav FROM paper_nav ORDER BY date", conn, index_col="date")["nav"]
        fills = pd.read_sql_query("SELECT * FROM paper_fills", conn)
        pos, cash = positions(conn, p["capital"])
        fwd = pd.read_sql_query("SELECT * FROM paper_predictions", conn)
        base = pd.read_sql_query("SELECT ticker, date, prediction, target FROM predictions WHERE model_id = ?", conn, params=(bt.MODEL_ID,))
        bars = pd.read_sql_query("SELECT ticker, date, close FROM daily_bars UNION ALL SELECT ticker, date, close FROM live_bars "
                                 "ORDER BY ticker, date", conn)
    finally:
        conn.close()
    h = cfg["phase3"]["target_horizon_days"]
    realized = build_target(bars, h)[["ticker", "date", "target"]]
    fwd = fwd.merge(realized, on=["ticker", "date"]).dropna(subset=["target"])
    min_t = cfg["phase3"]["min_tickers_per_day"]

    def ic_by(frame: pd.DataFrame):
        pan = make_panel(frame["date"].to_numpy(), frame["ticker"].to_numpy())
        return pan, daily_ic(frame["prediction"].to_numpy(), frame["target"].to_numpy(), pan, min_t)

    pan_b, ic_b = ic_by(base)
    by_year = pd.Series(ic_b, index=pan_b.years).groupby(level=0).agg(["mean", "count"])
    pan_f, ic_f = ic_by(fwd)
    ic_f = ic_f[~np.isnan(ic_f)]
    r = nav.pct_change().dropna()
    dd = (nav / nav.cummax() - 1.0).min()

    L = ["# Phase 6/7 report: forward paper run and tracking", "",
         f"Paper money only. Signal: baseline ridge refit at each rebalance (label: no validated edge). Execution: C++ simulator, "
         f"`{cfg['phase6']['strategy']}`. Long-only top {p['top_n']} of 20, fixed ${p['capital']:,}, rebalance every {p['rebalance_days']} "
         f"trading days. Intraday slices are synthetic (see phase5_report.md).", "",
         f"## Forward run: {nav.index[0]} to {nav.index[-1]} ({len(nav)} trading days)", "",
         "| metric | value |", "|---|---|",
         f"| NAV start / end | {p['capital']:,.0f} / {nav.iloc[-1]:,.0f} |",
         f"| return | {nav.iloc[-1] / p['capital'] - 1:.2%} |",
         f"| Sharpe (annualized, daily) | {r.mean() / r.std() * np.sqrt(252):.2f} |",
         f"| max drawdown | {dd:.2%} |",
         f"| daily P&L mean / best / worst (USD) | {nav.diff().mean():,.0f} / {nav.diff().max():,.0f} / {nav.diff().min():,.0f} |",
         f"| fills | {len(fills)} orders, fees USD {fills['fees'].sum():,.0f} |",
         f"| open positions | {', '.join(f'{t} {q}' for t, q in sorted(pos.items())) or 'none'} (cash {cash:,.0f}) |", "",
         "A short window on a signal with no validated edge: return and Sharpe here are descriptive, not evidence of skill.", "",
         "## Signal decay", "",
         f"Forward rank IC on {len(ic_f)} rebalance days with a matured {h}-day outcome: mean "
         f"{(ic_f.mean() if len(ic_f) else float('nan')):.4f}"
         + (f", Newey-West t {newey_west_t(ic_f, 1):.2f}." if len(ic_f) > 3 else "."),
         "", "Out-of-sample IC by year, Phase 3 baseline (daily, all days):", "", "| year | mean IC | days |", "|---|---|---|"]
    for y, row in by_year.iterrows():
        L.append(f"| {y} | {row['mean']:.4f} | {int(row['count'])} |")
    L += ["", "Reading: yearly IC flips sign (positive 2016 to 2018 and 2023 to 2025, negative 2019 to 2022 in Phase 3), which is what "
          "an edge that never cleared its significance gates looks like. The forward mean sits inside that range and, at this sample "
          "size, cannot be told apart from zero. A decay claim needs the forward IC to leave that band for a sustained stretch; "
          "none is asserted here.", ""]
    REPORT_PATH.write_text("\n".join(L))
    print(f"report: {REPORT_PATH}")
    return 0


def reset(cfg: dict) -> int:
    conn = sqlite3.connect(_ROOT / cfg["data"]["db_path"])
    for t in PAPER_TABLES:
        conn.execute(f"DROP TABLE IF EXISTS {t}")
    conn.commit()
    conn.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quantico paper trading loop (Phase 6/7)")
    parser.add_argument("cmd", choices=["run", "report", "reset"])
    args = parser.parse_args(argv)
    cfg = rs.load_config()
    return {"run": run, "report": report, "reset": reset}[args.cmd](cfg)


if __name__ == "__main__":
    sys.exit(main())
