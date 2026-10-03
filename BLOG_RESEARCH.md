# Evidence to collect for the strategy article

Start with the release manifest, version and strategy specification. Explain
gap continuation, leveraged gap reconstruction, prior-session filtering, sibling
selection, the opening range, entry orders, exits, and margin-based sizing.
Distinguish the strategy thesis from the observations used to choose parameters.

Record the following for every result you intend to publish:

1. Source hashes, dependency versions, configuration, execution timestamp and
   data cutoff. Record the vendor, time zone, adjustment basis, split handling,
   official-close treatment, missing bars and underlying-cache freshness.
2. The full development sequence and number of alternatives examined. State how
   the liquidity threshold, class weights and filter threshold were chosen.
   Explain what was fixed before inspecting the holdout and what changed after.
3. Canonical `freeze_run` outputs for the full sample, in-sample and holdout.
   Report fixed-notional and compounded results separately, with the initial
   capital, costs, trade counts and precise Sharpe / drawdown conventions.
4. Parameter stability, subperiods, class / regime attribution and cost stress.
   Keep data coverage and universe constant when attributing an improvement to
   a strategy rule. Account for selection and repeated testing.
5. Synthetic replay and parity test outcomes. Then privately reconcile real
   sessions: candidate disagreements, missed entries, partial fills, rejection
   causes, commissions, exit times and modelled versus observed slippage.
6. Remaining specification findings, their effect on results, and any fixes
   since the snapshot. Re-run affected comparisons before publishing claims.

The specification's V49 comparison is explicitly approximate because its runs
used stale underlying caches. Its numbers and older README figures should not
be treated as the final release's verified results.

Use aggregate or normalized live statistics in the article. Raw account equity,
broker identifiers, fills, database exports, screenshots, webhook addresses and
deployment details remain private. Publish plots and tables only after reviewing
their labels, annotations, file metadata and underlying data.
