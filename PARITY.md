# RiskLabAI — Python ↔ Julia parity ledger

Tracks the public API correspondence between `RiskLabAI.py` and `RiskLabAI.jl`
as the Julia package is reconstructed (Phase 2). The Julia side adopts
Julia-convention `snake_case`; the table is the source of truth for matching
names. Built up one submodule per PR.

## Conventions

- Python: `snake_case` functions, `CapWords` classes, `UPPER_SNAKE` constants.
- Julia: `snake_case` functions, `CapWords` types, `UPPER_SNAKE` constants,
  submodules mirroring the Python sub-packages (`Utils`, `Data`, `Backtest`,
  `Features`, `Optimization`, `HPC`).
- "Deliberate divergence" rows note where the two intentionally differ (e.g.
  Python-only or Julia-only methods).

## Utils  (PR 1 — wired)

| Concept | Python | Julia | Notes |
|---|---|---|---|
| EWMA | `utils.ewma` | `RiskLabAI.Utils.ewma` / `RiskLabAI.ewma` | identical denominator `1 + (1-α) + (1-α)² + …`; numeric parity asserted |
| Struct-field inheritance | — (n/a in Python) | `RiskLabAI.Utils.@field_inherit` | Julia-only; underpins the bar-type hierarchy |
| Column / stat constants | `utils.constants.*` | `RiskLabAI.Utils.*` | ASCII identifiers; `CUMULATIVE_θ`→`CUMULATIVE_THETA` |

Resolved duplicate: `Utils/EWMA_merge.jl` (`calculateEWMA`) used the wrong
weight exponent `(1-α)^i` (skips the first-order term); removed in favour of
the correct `Utils/ewma.jl`.

## Data.Structures (bars)  — PR 2 (standard bars wired)

| Concept | Python | Julia | Notes |
|---|---|---|---|
| Bar type taxonomy | `data.structures` class hierarchy | `Data` + `types.jl` parametric `Metric` dispatch (`Dollar`/`Volume`/`Tick`) | jl keeps the parametric design |
| Abstract base | `AbstractBars` | `RiskLabAI.Data.AbstractBars` | fields/methods → snake_case |
| Standard bars | `StandardBars` (`construct_bars_from_data`) | `RiskLabAI.Data.StandardBars{T}` (`construct_bars_from_data`) | **wired**; numeric parity asserted against `test_standard_bars.py` |
| `update_base_fields` / `tick_rule` / `construct_next_bar` / `bar_construction_condition` | snake_case | snake_case | renamed from camelCase |
| Time bars | `TimeBars` (`construct_bars_from_data`) | `RiskLabAI.Data.TimeBars` | **wired (PR 3)**; logic rewritten to match Python (emit at bucket boundary, then add the new tick); numeric parity asserted against `test_time_bars.py` |
| Imbalance bars | `ExpectedImbalanceBars`, `FixedImbalanceBars` | `RiskLabAI.Data.{Expected,Fixed}ImbalanceBars{T}` | **wired (PR 4)**; bar-count/tick parity asserted vs `test_imbalance_bars.py`. Fixed bugs: strict `>`→`≥`, `trunc`→`Int` EWMA window, post-bar `warm_up`, dropped `ProfileView`/`TimerOutputs` deps |
| Run bars | `ExpectedRunBars`, `FixedRunBars` | `RiskLabAI.Data.{Expected,Fixed}RunBars{T}` | **wired (PR 5)**; bar-count/tick parity asserted vs `test_run_bars.py` (Fixed→1 bar of 7 ticks). Threshold `E[T]·max(P_buy·E[b_buy], (1-P_buy)·E[b_sell])`; same bug fixes as imbalance |

**`Data.Structures` is now complete** — standard, time, imbalance, and run bars
are all wired with Python-parity tests.

## Data.Differentiation  — PR 6 (wired)

| Concept | Python | Julia | Notes |
|---|---|---|---|
| Expanding-window weights | `calculate_weights_std` | `Data.calculate_weights_std` | exact parity (reversed, `w₀` last) |
| FFD weights | `calculate_weights_ffd` | `Data.calculate_weights_ffd` | exact parity |
| Standard frac-diff | `fractional_difference_std` | `Data.fractional_difference_std` | exact parity; `NaN` warm-up before `skip` |
| Fixed-width frac-diff | `fractional_difference_fixed` / `_single` | `Data.fractional_difference_fixed` | exact parity; `NaN` warm-up before `width` |
| Optimal `d` (ADF scan) | `find_optimal_ffd_simple` | `Data.find_optimal_ffd` | ADF via `HypothesisTests.ADFTest`; **behavioural** parity only (ADF impl ≠ statsmodels). Returns columnar `NamedTuple` |
| Stationary log-price | `fractionally_differentiated_log_price` | `Data.fractionally_differentiated_log_price` | same ADF caveat |

**Deliberate divergence:** the Julia frac-diff functions operate on a single
`AbstractVector` (callers `map` over columns) rather than on a `DataFrame`; and
Python's `plot_weights` (matplotlib) is omitted (no plotting dependency).
`HypothesisTests` added to `[deps]` for the ADF-based finders.

## Data.Weights  — PR 7 (wired)

| Concept | Python | Julia | Notes |
|---|---|---|---|
| Label concurrency | `expand_label_for_meta_labeling` | `Data.expand_label_for_meta_labeling` | events as `(event_start, event_end)` vectors; returns `(index, concurrency)` |
| Average uniqueness | `calculate_average_uniqueness` | `Data.calculate_average_uniqueness` | T×N indicator matrix → per-event uniqueness; exact parity |
| Return-attribution weights | `sample_weight_absolute_return_meta_labeling` | `Data.sample_weight_absolute_return_meta_labeling` | `Σ|r_t|/c_t`, normalised to N; NaN first-return skipped (matches pandas `.sum`) |
| Time decay | `calculate_time_decay` | `Data.calculate_time_decay` | linear decay; oldest→`clf_last_weight ∈ [0,1]`; exact parity |

**Deliberate divergence:** Python carries event start times in the pandas
Series *index*; the Julia port passes `event_start`/`event_end` (and the price
index) as parallel sorted vectors. `calculate_time_decay` assumes the weights
are already in chronological order (Python sorts by index). All four reference
outputs are asserted against the Python implementation.

Next submodules: `Data` (denoise, labeling, distance), then `Backtest`,
`Features`, `Optimization`, `HPC`.

Information-driven base: `ewma_expected_imbalance`, `imbalance_at_tick` (metric
dispatch), dynamic threshold `E[T]·|E[b]|` — shared by imbalance and run bars.

Bar row layout (both languages): `[date_time, idx, open, high, low, close,
volume, buy_volume, sell_volume, ticks, dollar, threshold]`.

_(further submodules appended as they are wired)_
