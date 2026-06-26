# `_superseded/` — archived dead-code surface from `src/`

These files were **never loaded by the package**. They are not reachable from the
include graph rooted at `src/RiskLabAI.jl`, and were moved here out of `src/` on
**2026-06-26** to restore the public/internal boundary that `src/` is meant to
carry (GOVERNANCE §10.7) and to remove the duplicated `camelCase` re-implementations
of wired `snake_case` code (§10.2/§10.6).

This is an **archive, not a delete** (house rules §7; this repo's "shelve on record,
never delete"). The directory layout under `_superseded/src/...` mirrors each file's
former path under `src/...`, so any file can be reversed with a single move back.

## Provenance

- Driver: `reports/ARCH_AUDIT_jl_2026-06-26.md` §2.1 + §4 item 1, and
  `RECONSTRUCTION_PLAN.md` §3.1.
- Method: the include graph was reconstructed by following every `include(...)`
  statement transitively from `src/RiskLabAI.jl` (nine submodules + the two wired
  legacy includes). Every `.jl` file under `src/` NOT in that graph was archived here.

## Explicitly KEPT in `src/` (wired legacy includes — do not confuse with these copies)

- `src/Backtests/ProbabilityOfBacktestOverfitting.jl` — wired at `src/RiskLabAI.jl:135`
- `src/BetSize/BetSizing.jl` — wired at `src/RiskLabAI.jl:136`

## Status

Reference-only. Not compiled, not tested. Several files here have broken `include`
paths (`Controller/`, `Calibration/`) or crash-on-first-call bugs catalogued in
`RECONSTRUCTION_PLAN.md`; they are retained for history/migration salvage, not use.
