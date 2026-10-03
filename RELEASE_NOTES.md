# 0.2.0 V49 source release candidate

The exported runner package is version 0.2.0. This candidate captures the current
V48 sizing / liquidity rules and V49 prior-session definition. It includes
uncommitted strategy updates present at export time; file hashes in MANIFEST.json
identify the exact exported source. It is not a claim of production readiness.

Release-only changes replace personal absolute path literals with paths relative
to the two sibling projects, replace obsolete READMEs, and exclude deployment
instructions, personal operational files and historical outputs. Strategy rules
are preserved. The working trading system is not changed by the export.

The specification's section 13 lists unresolved differences. In particular:

- Historical underlying caches can be stale and distort prior-session filtering.
- Live half-day scheduling is documented as unsupported.
- Live limit fills can miss; the backtest assumes fills at the boundary with
  modelled slippage. Target anchoring and exit labels also differ.
- Margin release, whole-share partial fills and commission sizing differ.
- Low-history sigma fallback and candidate universe selection can differ.
- Stops/targets and the fill model evolved during live testing; do not present
  earlier live sessions as observations of this exact strategy version.

Validation results belong in VALIDATION.md after testing this exact snapshot.
Do not replace failed tests with historical passing counts from the specification.
Personal recorded-session replay fixtures are excluded, so replay tests requiring
those fixtures cannot certify agreement with actual account sessions.

Publication boundary: initialize a new repository from this directory if using
a clean snapshot. Existing repository history was not scrubbed or certified.
The source scanner checks known secret values from local .env files and common
credential / personal-data patterns. It cannot prove the absence of all possible
personal information. Review the manifest and selected source before publishing.
