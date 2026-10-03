# Source validation (2026-10-02)

The exported source was checked with Python 3.13.3 and the locally installed
dependencies. No broker session or real orders were started.

- `python verify_snapshot.py`: **755 passed, 4 deselected**. The deselected
  group is `TestRecordedSessionsReplay`, which requires excluded personal
  session fixtures. Synthetic replay and strategy-parity tests ran.
- Canonical `analysis.freeze_run` and its imported configuration load from the
  exported source. Checks confirm 57 live symbols, k=0.50 and stop distance=1.00.
- Export boundary tests: **4 passed**, including secret rejection before any
  artifact is written, secret-free finding messages, allowlist exclusions,
  archive membership and SHA-256 manifest matching.
- The selected source passed checks for known local .env secret values, personal
  Windows paths, email addresses, broker account identifiers, common key/token
  formats, private keys, webhook URLs and literal credential assignments.

The final MANIFEST.json identifies the source files. Trading source hashes were
compared against the tested candidate. Subsequent additions are release
documentation and a chart generated from aggregated archived backtest results.

Chart validation: daily totals reconcile to the archived V49 report for all
three spans (full, in sample and date holdout), with matching trade counts.
The out-of-sample split is 2026-05-21. CSV columns contain only dates, aggregated
simulated P&L and sample labels. Raw fills, symbols and account records are not
included. The chart was visually inspected and its SVG text was scanned along
with the public source. This is not a fresh numerical performance run.

Limits: dependency installation in a fresh environment was not tested. No full
historical performance run was performed for this source-only package. Licensed
market data, original account-session replays, live broker behaviour, data
freshness and the unresolved specification findings are not certified by these
checks. Existing private Git histories were not audited or scrubbed.
