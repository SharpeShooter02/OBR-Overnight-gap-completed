# ORB gap strategy: specification of what the code does today

Snapshot: 2026-09-29, after V49 (prior-session filter measured open-to-close).
This document describes the code, not intent. Where the backtest and live disagree, both are stated and the
disagreement is listed in section 13. Anything marked **UNVERIFIED** was read from code and never observed running.

**Repos.** BT = `BacktestingGaps` (this repo). LV = `orb-live-trading/orb_live`.
**Canonical backtest** = `python analysis/freeze_run.py` (locked V48 config + V49). Not `authoritative_run.py main()`, see 13-K.

**How this was built.** Four agents read the code (candidates; entries/exits; sizing/risk; costs/timeline/plumbing),
each reporting both repos with file:line. I independently checked the constants in section 14 and implemented and
tested V49. Everything else is agent code-reading; line numbers can drift, so search by function name.

---

## 1. The strategy in one paragraph

Each morning, take a fixed universe of 57 leveraged ETFs on 29 underlyings. Keep ETFs that gapped at least
`leverage x 2%` versus the official prior close, measured at the close of the 09:30 bar. Reconstruct the underlying's
gap from the ETF gap. Drop the underlying if its **prior session already moved in the gap direction by more than
0.5 x sigma** (the PS filter). Keep exactly one ETF per underlying. Build the opening range from 09:30-09:59.
From 10:00, enter **in the direction of the ETF's own gap** the first time price touches the range boundary
(gap up: long at ORB high; gap down: short at ORB low), once per symbol per day. Stop is the opposite boundary.
Target is 2 x range from the entry boundary. Otherwise exit at the 15:58 bar. Skip the trade if round-trip cost is
more than 15% of the target. Size by asset class x SPY-gap regime, in units of 10% of equity, limited by a margin budget.

This is continuation of the gap, not a fade. The PS filter exists to skip gaps that follow an already-large move.

---

## 2. Daily timeline (America/New_York)

| Time | Event | Code |
|---|---|---|
| 08:30 | Daemon wakes. Refreshes underlying daily parquet, **aborts the process if any is stale**. Captures start equity. | `runner/main.py _run_daemon`; `session_runner._run_pre_market`; `underlying_data.refresh_and_assert_fresh` |
| 09:30 | Late-start cutoff: a session starting after this is skipped. | `session_runner LATE_START_CUTOFF` |
| 09:31 | **Phase 1 gap scan.** Ref price = close of first 1-min bar in 09:30-09:32. Gap, PS filter, direction filter, sibling selection, regime, weights. | `signals/pre_market.py run_phase1`, `signals/gap_scan.py scan_gaps`, `strategy/v1_strategy.py plan_session` |
| 09:31-10:00 | Opening-range window. No subscriptions. | `_run_orb_window` |
| 10:00 | **Phase 2.** Fetch 09:30-10:05 1-min bars (historical, RTH), compute ORB, cost/reward gate, pre-flight, margin seed/prewarm (IB whatIf), then subscribe to 5-second bars for candidates only. | `pre_market.run_phase2`, `_run_post_orb` |
| 10:00-15:58 | Entries allowed (no cutoff). Exits by exchange bracket orders and bar polling. | `strategy_engine.on_bar`, `position_manager.on_bar` |
| 15:58 bar (~15:59:05) | Market exit of anything open. | `position_manager.py` EOD branch |
| ~15:59:30 | Safety-net `flatten_all("eod_sweep")`, then day_state=closed, reconcile closed trades from IB executions, equity curve, report. | `session_runner._wait_until_eod`, `_run_eod` |

The `session_runner.py` docstring that says the gap scan runs at 08:30 is stale. It runs at 09:31.

---

## 3. Universe

- Source: `master_universe.csv` (184 rows, byte-identical in both repos) plus force-includes.
- Live keeps classes `1_BROAD`, `2_SECTOR_pair`, `4_CRYPTO`, `force_include`, any `binary_event==YES` row and all `FORCE_INCLUDE`
  names, minus `EXPLICIT_DROPS`, restricted to underlyings that have a sigma (`v1_strategy.build_universe`).
- Force-include: AMDL, TSLL, NVDU, KOLD, BOIL, UXRP, XRPT, GUSH.
- Dropped: CWEB, CHAU, BNKU, DFEN, DPST, NAIL, RETL, DRN, MIDU, NRGU, ETH, and in live also BTFX and UBR.
- Result (measured): **57 ETFs**, C1=12, C2=12, C3=33. Live and backtest agree once BTFX/UBR are removed (13-A).
- Leverage, inverse flag and underlying come from the master (live) / `INSTRUMENTS` (backtest). Measured equal for all 57.

| Underlying | ETFs (inv = inverse) |
|---|---|
| AMD | AMDL |
| BTC | BITU, BITX, BTCL, BTCZ(inv), SBIT(inv) |
| DIA | SDOW(inv), UDOW |
| EEM | EDC, EDZ(inv) |
| ETH | ETHD(inv), ETHT, ETHU, ETU |
| EWW / EWY / EWZ / FEZ | MEXX / KORU / BRZU / EURL |
| FXI | YANG(inv), YINN |
| GDX | DUST(inv), GDXU, NUGT |
| GDXJ | JDST(inv), JNUG |
| IBB | LABD(inv), LABU |
| IHE / INDY / MSOS | PILL / INDL / MSOX |
| IWM | SRTY(inv), TNA, TZA(inv), URTY |
| NVDA / TSLA | NVDU / TSLL |
| QQQ | FNGD(inv), FNGU, SQQQ(inv), TQQQ |
| SOL / XRP | SOLT / UXRP, XRPT |
| SOXX | SOXL, SOXS(inv) |
| SPY | SPXL, SPXS(inv), UPRO |
| UNG | BOIL, KOLD(inv) |
| XLC / XLF / XLK / XOP | WEBL,WEBS(inv) / FAS,FAZ(inv) / TECL,TECS(inv) / GUSH |

3x unless: AMDL, TSLL, NVDU, crypto ETFs, NUGT/DUST, JNUG/JDST, BOIL/KOLD, GUSH, MSOX, BRZU are 2x.

**Asset classes (used for sizing).** `classify()` in `v1_strategy.py`, shared by the backtest.
- **C1 (crypto-heavy):** BITX BITU SBIT BTCZ BTCL BTFX ETHU ETHT ETHD ETH ETU SOLT UXRP XRPT
- **C2:** YINN YANG KORU BRZU INDL MEXX EDC EDZ AMDL TSLL NVDU GUSH
- **C3:** everything else (gold miners, UNG, broad/sector, PILL, MSOX, EURL...)

---

## 4. Candidate selection (Phase 1)

### 4.1 Gap
`gap = (ref - prior_close) / prior_close`, on the **ETF's own prices**.
- `ref` = close of the 09:30 1-minute bar (live: first bar returned for 09:30-09:32).
- `prior_close` = official daily close (live: IB daily TRADES bar, RTH; backtest: last 1-min bar replaced by the official IBKR close if within 3%, register V42).
- ETF qualifies if `|gap| >= leverage x 0.02` (`GAP_THRESHOLD = 0.02`).
- Underlying gap is **reconstructed**: `(-etf_gap if inverse else etf_gap) / leverage`. The underlying's own gap is never used.
- Several ETFs on one underlying: the largest `|reconstructed gap|` wins and sets the direction for the whole group.

### 4.2 Prior-session (PS) filter, V49
The trade is **allowed** if `move x effective_direction <= K_SIGMA x sigma`, with `K_SIGMA = 0.50`.
- `effective_direction` = gap sign for normal ETFs, minus that for inverse ETFs (equivalently, the underlying's gap sign).
- **move** (`prior_session_leg`): equities = `close_t-1 / open_t-1 - 1` (the prior session's own open-to-close);
  24h underlyings (BTC, ETH, SOL, XRP, LINK, ADA, XLM) = `close_t-1 / close_t-2 - 1` (UTC daily bar, register V45).
  The overnight gap into the prior session, on a Tuesday that includes the weekend, is **not** part of the move.
- **sigma** = std of `|move|` on the same basis, over all daily bars strictly before the session (expanding, point-in-time),
  at least 250 observations. Below 250, live falls back to the static `sigma_master.csv` value; the backtest allows the trade (13-G).
- **Missing data** (no bars, no open, NaN) **allows** the trade, in every code path.
- The filter is applied **twice** in each system and both must pass (13-H):
  live = `strategy_signals.check_prior_session_filter` (per ETF, in `gap_scan`) then `v1_strategy.passes_ps_filter` (per underlying);
  backtest = `orb_backtester.check_prior_session_filter` (per ETF, decides which trades exist) and
  `scripts/etf_basis._ps_passes` (per underlying, decides which candidates exist).
- Live and backtest implementations of the leg and sigma agree to 3e-15 on 1,218 comparisons (verified).

Why V49 (decision log). Close-to-close counted a Friday-to-Monday weekend gap as "Monday's move", so a flat Monday after a -3% weekend
gap still blocked Tuesday's trade (this is what skipped BOIL/KOLD on 2026-09-29). Full-sample backtest, k unchanged:

| Variant | Trades | P&L | Sharpe | Holdout P&L / Sharpe |
|---|---|---|---|---|
| close-to-close (old) | 1772 | $16,459 | 1.72 | $676 / 1.38 |
| **open-to-close, sigma on same basis (V49)** | 1776 | $16,071 | 1.67 | $1,086 / 2.02 |
| no filter after weekends | 1948 | $14,895 | 1.47 | $555 / 1.03 |

V49 was chosen on principle (the thesis says "prior session"), not on performance: it costs ~2% of P&L, inside the noise. All rows
were run with stale underlying caches for the last ~5 weeks (13-B), so treat them as approximate.

### 4.3 Other filters
- `DIRECTION_FILTERS = {LABU: +1, LABD: -1}`: the ETF's own gap direction must match.
- No day-of-week exclusions, no ORB-size floor, no RTG.

### 4.4 One ETF per underlying
1. Drop siblings whose measured initial-margin rate for the side to be traded exceeds `MAX_MARGIN_RATE = 2.0`, unless that empties the group.
2. Keep the one with the highest prior close ("skip-cheap"; `keep_n = 1`).
- `SHARED_ALLOTMENT = {GDXJ: GDX}`: GDX and GDXJ each pick their own ETF but share one weight allotment (each weight is halved when both fire).

---

## 5. Opening range and entry (Phase 2 and after)

- **ORB** = high and low of 1-minute bars 09:30-09:59 (10:00 bar excluded). Needs 25 bars (10 for BOIL, KOLD). Skip if `high <= low`.
- **Cost/reward gate (10:00).** `reward_bps = 2.0 x (H-L) / boundary x 1e4`. Skip if `(entry_bps + stop_bps) / reward > 0.15`, where the bps come from the
  participation slippage model (section 9) evaluated at notional $2,943 and the ORB-window median dollar volume per minute.
  Live enforces it (`decision="cost_to_reward"` is stored). A skipped candidate is **not** replaced by a sibling, in either system.
- **Pre-flight.** Gate A (explicit exclusions) and B (tradable/shortable) are on; Gate C (ADV/dollar-volume) is **off**. The `gate_c_disabled`
  you see in the `candidates` table means exactly that.
- **Trigger.** Intrabar touch, no close confirmation: long if `bar.high >= ORB high`, short if `bar.low <= ORB low`. Live evaluates every
  5-second bar (the 1-minute path is a fallback). Backtest scans 1-minute bars.
- **Direction** = the ETF's **own** gap sign. An inverse ETF that gaps up is bought long.
- **Once per symbol per day.** No re-entry after a stop, a miss or a rejection.
- **Entry cutoff:** none before the 15:58 exit (`latest_entry_minute = None`).
- **Entry order (live).** Bracket: parent LIMIT at boundary +/- 0.35 x ORB range (TIF DAY), TP1 limit and stop-market children in one OCA group.
  Unfilled after 60 seconds: cancelled, no retry.
- **Entry price (backtest).** The boundary, moved against the trader by the modelled entry slippage. Always fills.

---

## 6. Stop, target, exit

| | Rule | Note |
|---|---|---|
| Stop | `boundary -/+ 1.00 x range` = the **opposite ORB boundary** (`STOP_ORB_DISTANCE = 1.00`) | Anchored to the boundary, not the fill. Stop-market. Live constant is imported by the backtest. |
| Target (TP1) | `entry +/- 2.0 x range` | 100% of the position (`exit_ratio_tp1=1.0`, TP2/TP3 = 0). Live anchors to the boundary; backtest to the slipped entry. |
| Breakeven/trail | dormant | TP1 closes everything. |
| EOD | 15:58 bar, market order | Backtest: bar close, zero EOD slippage. Half days: see 13-J. |

The code has **one** stop distance. Live traded a tighter 0.75 stop before late August (a comment in `strategy_signals.py` says until
2026-08-30); the backtest applies 1.00 to all history. There is no date switch anywhere in the code.

`exit_reason` in the backtest: `STOP`, `TP2` (**this is a TP1 winner**, because the loop does not stop at TP1 and later labels it), `EOD`.
Live: `STOP`, `TP1`/`TP1_ONLY`, `EOD`, plus `EOD_<reason>` from a flatten, and recovery labels.

---

## 7. Sizing

**Unit.** One unit = 10% of equity. Backtest: fixed $1,000 per unit, non-compounding (`freeze_run` also reports a daily-compounded $10k series).
Live: `0.10 x NetLiquidation`, re-read at **every entry**.

**Regime** from SPY's opening gap `|open/prior_close - 1|`: `< 0.4%` quiet, `> 1.0%` flood, otherwise active. Missing gap = **flood** (live).

**Weights (units), locked V48:**

| Class | quiet | active | flood |
|---|---|---|---|
| C1 | 2.0 | 2.0 | 0.0 (never trade) |
| C2 | 0.5 | 0.5 | 0.5 |
| C3 | 1.0 | 1.0 | 1.5 |

**Per-trade multiplier** = `weight x cap_factor / n_candidates_in_allotment_group`.
`cap_factor = min(1, 20 / sum(weights))` over the 09:31 candidate set (`CAP_UNITS = 20`, not read from `LiveConfig`, so an override does nothing).
Shares = `floor(0.10 x equity x multiplier / boundary_price)`; whole shares only; zero shares = skip.

**Margin budget.**
- Backtest (`run_sequential`, time-ordered): budget 10 units. A trade needs `weight x margin_rate`. Full fill if it fits, else a partial fill of
  `room/need` if that is at least 10%, else skipped. Per-trade need is capped at 5 units by clipping the weight. **Margin is never released**
  when a trade exits.
- Live (`position_manager.open_position`): budget = IB `AvailableFunds` snapshot at ~10:00. Same 10% minimum, in whole shares. Margin **is** released when a position closes.
  Rates: IB whatIf measured at the ORB close, else the CSV seed, else 1.0.

---

## 8. Live risk controls (all of them)

| Control | Value | Behaviour |
|---|---|---|
| Duplicate/in-position guard | | one open or entering position per symbol |
| `max_concurrent_positions` | 0 (off) | |
| Kill switch | -3% of start equity, **realized P&L only** | blocks **new entries only**; does not flatten; persists for the day (`day_state.kill_triggered`); a restart resets the realized total but not the flag |
| `max_gross_exposure_pct` | 2.0 x equity | on open notional |
| `max_position_pct` | 0.50 x equity | on notional; cannot bind at current weights |
| Buying-power floor | IB BuyingPower | `insufficient_buying_power` |
| Margin budget | section 7 | partial or reject |
| Zero weight | C1 on flood days | dropped at planning |

A `.kill` file is mentioned in the runbook but no code reads it.

---

## 9. Costs and the backtest fill model

- **Slippage** (`analysis/out/slippage_model.json`, identical copy in LV): `bps = a + b x sqrt(min(notional / V, part_max))`, `V` = median dollar volume per minute over 09:30-09:59.
  entry a=10.98, b=25.31; stop a=3.44, b=84.77; `part_max`=1.0076; `default_notional`=$2,943.1. Calibrated on 71 entry and 23 stop live fills (2026-07-01 to 09-14).
  The engine charges at $2,943; `freeze_run` re-prices per trade at the weighted size (entry, and stop only when stopped). TP1 and EOD exits carry no slippage.
- **Commission:** IBKR fixed schedule, `clip(shares x 0.005, $1, 1% of value)` per leg, two legs. Spread/impact terms are zeroed when the slippage model is on.
- **Borrow, margin interest, failed locates:** zero/unmodelled (everything is flat by 15:58).
- **`USE_AS_TRADED`:** un-adjusts split-adjusted prices, affecting only the sibling price ranking (13-D).

---

## 10. Data

| Data | Live | Backtest |
|---|---|---|
| ETF prior close | IB daily bar (RTH TRADES) | last 1-min bar, replaced by the official IBKR close if within 3% |
| Ref price | close of first 1-min bar in 09:30-09:32 | close of the 09:30 bar |
| Underlying daily bars (with **open**) | IB for equities, yfinance for crypto, parquet in `orb_live/data/underlyings` (36 files), refreshed 08:30 only | `orb_event_study/cache/daily/<UL>.parquet`, refreshed by separate scripts (13-B) |
| ORB bars | IB 1-min historical, RTH | 1-min intraday cache |
| Streaming | IB 5-second real-time bars, then aggregated to 1-min | n/a |

Merge rule in live `update_one`: existing rows win on overlap (`drop_duplicates(keep="first")`), so a partial-day row is never corrected by this path.

---

## 11. Broker and operations (live)

- Only `IBClient` (ib_async). Defaults: host 127.0.0.1, port 4002 (paper) / 4001 (live), **clientId 1**. `--live` needs `IB_LIVE=1` and typing `yes`.
- Reconnect: up to 5 attempts (2, 4, 8, 16 s backoff). Optionally asks IBC to restart the gateway (port 7462) after 300 s of upstream loss or 2 failed attempts.
  In a session, bars are re-subscribed after reconnect.
- Gateway down at 08:30: the daemon retries the same date until 09:30, then skips the day. Process started while the gateway is down: it exits.
- Restart mid-day: `startup_reconcile` **flattens** any position the broker still holds unless started with `--recover` (which adopts positions).
- Alerts: log plus `alert_log` table only. `AlertManager` (webhook) is never instantiated. No push notifications exist.
- Status endpoint `127.0.0.1:8080/status`. `sigma_calibration_age_days = -1` (no rows in a dead table), `underlying_data_freshness = {}` (reads an empty table),
  `last_reconnect_ts = null` (nothing ever sets it): all three are dead fields, not signals.
- Tables in `state/live.db` that are written: `day_state`, `gap_scan`, `ps_filter_result`, `candidates`, `liquidity_metrics`, `breakout_signal`,
  `open_positions` (rows deleted on close), `closed_trades`, `fills` (entry legs only), `equity_curve`, `alert_log`. Others are empty/legacy.

---

## 12. Reproducing the numbers

```
python analysis/freeze_run.py            # canonical backtest, writes analysis/out/freeze_*.parquet/json
python scripts/daily_parity_check.py     # latest live session vs the backtest, exit 1 on divergence
python analysis/ps_variant_run.py base   # variants, write to analysis/out/ps_variants/<name>/
cd ../orb-live-trading && .venv/Scripts/python -m pytest orb_live/tests -q    # 754 pass, 3 stale-path failures
```
The Verdant pipeline "Daily live vs backtest parity" runs the first two after refreshing IB data. It does not refresh the underlying daily caches (13-B).

---

## 13. Where live and backtest differ, or are broken

Severity: **H** can change results or lose a day; **M** worth a decision; **L** cosmetic or small.

| ID | Sev | Finding | Evidence |
|---|---|---|---|
| **B** | **H** | 105 of 106 underlying daily caches in the backtest are stale (equities end ~08-21, crypto 09-13, only SPY is current). The PS filter silently reuses old rows for the last ~5 weeks. Nothing refreshes them; `refresh_crypto_daily.py` exists but is not scheduled; there is no equity equivalent. Also makes the daily parity check unreliable for recent days. | `orb_event_study/cache/daily/*.parquet` |
| **J** | **H** | **Half-days are unsupported live.** `IBClient.get_clock` hardcodes a 16:00 close, so `is_half_day()` is always False. Upcoming early closes not in the holiday set: **2026-11-27, 2026-12-24**. Live would wait for the 15:58 exit on a 13:00 close. UNVERIFIED how IB would behave. | `ib_client.get_clock`, `core/clock.py` |
| **M-gap** | **H** | Live daemon exits (does not retry) if any underlying is stale at 08:30 (`RuntimeError`), which loses the day until someone restarts it. | `main.py _run_daemon`, `refresh_and_assert_fresh` |
| **A** | M | BTFX and UBR are dropped from live but stay in the backtest's **candidate table**. They win the sibling pick on 29 candidate-days and can never fire, crowding out BITX/SBIT/BRZU. The trade engine drops them, so those days lose a tradable sibling. | `etf_basis` vs `run_v1_at_k extra_drops` |
| **E** | M | Entry fill: live is a limit at boundary +/- 0.35 x range, cancelled after 60 s, and can miss or partially fill; the backtest always fills with modelled slippage. Live TP1 is a passive limit (313 `tp1_limit_not_filling` alerts on 3 symbol-days); the backtest fills on touch. | `order_policy.py`, `position_manager.py` |
| **F** | M | Same-bar handling: the backtest never checks exits on the entry bar and books TP1 before the stop when both touch; live has both orders live immediately and racing. | `orb_backtester.py same_bar_fill_risk` |
| **G** | M | Sigma fallback: below 250 observations live uses static sigma, the backtest lets the trade through. MSOS's backtest cache has only 250 rows, so its PIT sigma is empty. | `pit_sigma._series` vs `point_in_time_sigmas` |
| **H** | M | Two PS applications in each system can differ when sibling gap signs disagree (live filters per ETF before aggregating; backtest aggregates then filters). Live's two applications use `cfg.ps_filter_k` and the constant `K_SIGMA`; an override would split them. | `gap_scan.py`, `etf_basis.py` |
| **R** | M | Regime: live uses the first-bar close over the IB prior close; backtest uses SPY's daily open. Missing regime = flood (live) vs skip the day (`freeze_run`) vs a count-based regime (`authoritative_run`). | `pre_market._spy_gap`, `refit_sleeve_weights_spy` |
| **S** | M | Margin release: the backtest never releases margin intraday; live does. Budget: fixed 10 units vs IB AvailableFunds snapshot. Per-trade need cap (5 units) exists only in `freeze_run`, not in live or `authoritative_run`. | `run_sequential`, `position_manager` |
| **T** | M | Live TP1 = boundary + 2R; backtest = slipped entry + 2R (slightly farther). | `strategy_signals.compute_entry` |
| **C** | M | Commissions: live pays IB's real commission with a ~$1 minimum on ~$3k orders and may place several orders per trade; the backtest formula is computed on a $20k engine notional. Size of the gap not computed. Volume units (shares vs lots) in `dollar_per_min` are UNVERIFIED. | `_production_run.apply_trade_costs` |
| **K** | M | `authoritative_run.py main()` is **not** the locked config (static sigma, ADV gate on, no slippage/cost-reward). Its parity table checks 14 constants but not the universe, gap threshold, reference price, sigma mode, `select_by` or `SHARED_ALLOTMENT`, and its k row is tautological. | `authoritative_run.py` |
| **L** | M | Kill switch is realized-only and never flattens; the runbook's `.kill` file is not implemented. | `risk_gate.py` |
| **P** | M | No push alerts of any kind. | `ops/alerts.py` unused |
| **D** | L | Sibling ranking price: live = IB official prior close; backtest = last cached 1-min close (split-adjusted unless `USE_AS_TRADED`). | `etf_basis.py` |
| **I** | L | Prior close and gap guards exist only in the backtest (`MAX_PRIOR_GAP_DAYS=5`, `MAX_PLAUSIBLE_GAP=0.5`, `SAMPLE_END`). | `etf_basis.py` |
| **N** | L | Backtest labels TP1 winners `TP2` with EOD `exit_time`; live says `TP1`. Any join on `exit_reason` needs a mapping. | `orb_backtester.py` |
| **O** | L | Partial-entry granularity: continuous vs whole shares. Live partials lose the OCA bracket and exit TP1 by market order. | `position_manager.py` |
| **Q** | L | Fixture recorder now stores `open`; fixtures recorded before V49 cannot replay an equity PS decision. | `session_fixture.py` |
| **U** | L | Dead/unused settings: `LiveConfig.cap_units`, `fractional_shares`, `paper_trading`. | `live_config.py` |
| **V** | L | Stale comments: "K_SIGMA 1.0", "ps_filter_k 1.25", "skip-cheap-top-2", "$1k-per-unit", gap scan "at 08:30". The code says 0.50 and keep-one. | various |
| **W** | L | My own script `reconcile_live_session.py` switches the stop distance from 0.75 to 1.00 at **2026-08-31**; the code comment says 2026-08-30. One of them is a day off. | `strategy_signals.py:257` |

---

## 14. Constants checked directly against the code (2026-09-29)

`K_SIGMA=0.50`, `MIN_SIGMA_OBS=250`, `MAX_MARGIN_RATE=2.0`, `CAP_UNITS=20`, `SPY_QUIET_MAX=0.004`, `SPY_FLOOD_MIN=0.010`, `REGIME_FALLBACK="flood"`,
`GAP_THRESHOLD=0.02`, `STOP_ORB_DISTANCE=1.00` (imported by the backtest), `tp1_target_multiple=2.0`, `exit_ratio_tp1/2/3=1/0/0`, `base_notional_pct=0.10`,
`entry_buffer_orb_frac=0.35`, `max_gross_exposure_pct=2.0`, `max_position_pct=0.50`, Gate C off, `max_cost_to_reward=0.15`, `session_kill_loss_pct=0.03`,
`eod_flatten_lead_secs=30`, `use_resting_entries=False`.

## 15. Not verified

Half-day behaviour at IB; IB 1-minute volume units; whether the official-close prior price in live and the backtest agree day by day;
the commission gap in dollars; whether entries between 10:00 and the deferred subscription are missed in practice; which path repairs partial-day crypto rows.
