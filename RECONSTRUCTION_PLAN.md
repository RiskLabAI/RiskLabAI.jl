# RiskLabAI.jl — Reconstruction Plan (Phase 2)

Status: **proposal, awaiting approval on the architecture/naming decisions in
§6.** No source rewrites yet. Companion to the repo-root `AUDIT.md` (detailed
findings) and `IMPROVEMENT_PLAN.md` (cross-language sequencing).

## 1. Verified current state (not the June-10 audit state)

Phases 0 and 1 already landed on `fix/phase-0-hotfixes`:

- The package **loads**. `src/RiskLabAI.jl` includes 2 files
  (`Backtests/ProbabilityOfBacktestOverfitting.jl`, `BetSize/BetSizing.jl`),
  exports the two **real** symbols (`probabilityOfBacktestOverfitting`,
  `generateSignal`) — the old phantom/case-mismatched exports and the
  `-> ReturnType` parse errors in those two files are fixed.
- `Project.toml` is real: 7 declared deps, compat bounds, `julia = "1.10"`,
  `Test` in `[extras]`/`[targets]`.
- `test/runtests.jl` has genuine smoke tests (`using RiskLabAI`, `isdefined`
  checks, a `discreteSignal` numeric assertion, a `selectRows` assertion).
- CI: modern matrix + JuliaFormatter check + CompatHelper.

So "make it load and prove it in CI" is **done**. What remains is the bulk: the
other **69 of 71** source files are not wired into the module, the heavy
duplication is untouched, and the unwired files still contain the crash-on-call
bugs catalogued in `AUDIT.md §3.2`.

## 2. The goal

`RiskLabAI.jl` becomes a loadable, tested library that covers what its source
tree already contains, mirrors the Python package's public API where sensible,
and has no load-time Python coupling. Parity with `RiskLabAI.py` is a
first-class goal.

## 3. Work, in order

### 3.1 Deduplicate before wiring (delete dead copies first)

Confirmed still present in the tree:

| Duplicate | Keep | Drop / merge |
|---|---|---|
| StrategyRisk ×2 | `Backtests/StrategyRisk.jl` | `Risk/StrategyRisk.jl` |
| Differentiation ×2 | `Data/Differentiation/Differentiation.jl` | top-level `Differentiation/` |
| `clusterKMeansBase` ×3 (Cluster, Features, NCO) | one in `Cluster/Clustering.jl` | de-dup the copies in `Features/Clustering.jl`, `Optimization/NCO.jl` |
| EWMA ×2 | one correct `Utils/ewma.jl` | `Utils/EWMA_merge.jl` — **resolve the `(1-α)^(i-1)` vs `(1-α)^i` discrepancy against the closed-form EWMA, add a numeric test** |
| HP tuning ×2 | reconcile `Calibration/` vs `Models/HyperParameterTuning.jl` | keep one |
| `symmetricCusumFilter` ×2, `*2.jl` versioned copies in `Validation/`, commented-out `Features/DistanceMetric.jl`, empty `Controller/all.jl` | — | archive/delete |

Everything dropped moves to a `_superseded/` archive (never hard-delete), with a
list, mirroring the Python discipline.

### 3.2 Module architecture **[APPROVAL — §6.1]**

Mirror the Python sub-package layout as Julia submodules:
`RiskLabAI.Data`, `RiskLabAI.Backtest`, `RiskLabAI.Features`,
`RiskLabAI.Optimization`, `RiskLabAI.HPC`, `RiskLabAI.Utils`, with explicit
`export`s per submodule. Keep `Data/Structures/types.jl`'s parametric `Metric`
taxonomy (`StandardBars{Dollar}` + dispatch) — the audit's "most elegant design
in either repo".

### 3.3 Wire up + fix crash bugs as each file is included

Fix the include-path wreckage in `Controller/` (`../model/`, `../utils/` don't
exist; double include; hardcoded CSV path). Then fix, each with a test, the
known crash-on-first-call bugs from `AUDIT.md §3.2`, e.g.:
`generateSignal` undefined `prob`; `calculateGaussianBet` ÷0;
`tripleBarrier` indexing `Vector`s as DataFrames; `HRP` `diagm`→`diag`;
`clusterKMeansBase` return-type mismatch; `HPC` partitions + missing
`using Base.Threads`; `numberConcurrentEvents` DimensionMismatch; inverted
`@assert`; remaining `-> ReturnType` parse errors in the unwired files.

### 3.4 Decouple from Python **[APPROVAL — §6.3]**

Remove load-time `pyimport` (`Features/Clustering.jl` `const Metrics =
pyimport("sklearn.metrics")`) and the PyCall/ScikitLearn/SymPy deps — replace
with Clustering.jl / native equivalents. Drop ProfileView from `src`.

### 3.5 Tests

Per-submodule `@testset`s. Reuse the Python suite's numeric reference values
(same inputs → same expected outputs — the cheapest cross-language correctness
check). Target: every exported function called at least once; AFML worked
examples (HRP weights, purge boundaries, bar OHLC) asserted on both sides.

### 3.6 Type-stability & docs (after correctness)

`const`-ify `Utils/constants.jl` globals; concretize struct fields
(`PurgedKFold.times`); replace `Vector{Any}` accumulators. Normalize the four
docstring dialects to Julia markdown; fix the `deploydocs` `<repository url>`
placeholder; README with install + quickstart. Remove the committed
`Manifest.toml` (still tracked despite `.gitignore`).

## 4. Naming canon **[APPROVAL — §6.2]**

The Julia source is uniformly `camelCase` (against Julia convention and against
the Python sibling). Because the package effectively doesn't run today, renaming
to Julia-convention `snake_case`/lowercase now is nearly free (almost nothing
external depends on the current names). Doing it during wire-up avoids a second
breaking pass later. A shared `PARITY.md` maps py↔jl public names; CI fails if a
public export is missing from the table.

## 5. Recommended delivery: vertical slice first

Rather than "dedup everything, then wire everything" (a giant, hard-to-review
big bang), I recommend proving the pattern on one submodule end-to-end first:

- **PR 1 (skeleton + first slice):** establish the submodule architecture
  (§3.2) and fully wire **`Data/Structures`** (bars) — the cleanest, highest-py-
  parity area with the nice parametric types — including dedup touchpoints,
  bug fixes, and per-submodule tests with numeric values shared with the Python
  bar tests. This sets the template (naming, exports, test style, docstrings).
- **PRs 2..n:** one submodule per PR (Utils/HPC → Data/Labeling+Weights →
  Differentiation/Denoise/Distance → Features → Optimization/Cluster →
  Backtest/Validation), each green in CI before the next.
- Defer (py-only, documented in `PARITY.md`): PDE/Deep-BSDE and the
  overfitting-simulation suite.

This keeps every PR small, reviewable, and green — the same discipline we used
on the Python side.

## 6. Decisions needed before coding

1. **Module architecture (§3.2):** mirror the Python sub-packages as Julia
   submodules (recommended) — yes/no?
2. **Naming canon (§4):** convert jl public names to Julia `snake_case` now,
   during wire-up (recommended, since nothing loads today) — or keep `camelCase`
   to match the book snippets?
3. **PyCall-backed features (§3.4):** reimplement natively (recommended) or cut
   the affected features?
4. **Delivery (§5):** vertical-slice-first, one submodule per PR (recommended) —
   or a different grouping?
5. **General registry:** register `RiskLabAI.jl` in Julia's General registry
   once Phase 2 lands and tests are real — yes/no (later)?

Once you decide 1–4, I'll start with PR 1 (skeleton + `Data/Structures`),
implemented on a branch with tests, and give you the step-by-step to run
`Pkg.test()` locally and open the PR.
