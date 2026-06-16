# Shared string constants (column names and bar-statistic keys).
# Declared `const` for type stability; ASCII identifiers only (the previous
# `CUMULATIVE_Θ`/`CUMULATIVE_θ` identifiers were non-ASCII and inconsistently
# cased). String values are unchanged.

const DATE_TIME = "Date Time"
const TIMESTAMP = "Timestamp"
const TICK_NUMBER = "Tick Number"
const OPEN_PRICE = "Open"
const HIGH_PRICE = "High"
const LOW_PRICE = "Low"
const CLOSE_PRICE = "Close"

const CUMULATIVE_TICKS = "Cumulative Ticks"
const CUMULATIVE_DOLLAR = "Cumulative Dollar Value"

const THRESHOLD = "Threshold"

const CUMULATIVE_VOLUME = "Cumulative Volume"
const CUMULATIVE_BUY_VOLUME = "Cumulative Buy Volume"
const CUMULATIVE_SELL_VOLUME = "Cumulative Sell Volume"

const CUMULATIVE_THETA = "Cumulative theta"
const CUMULATIVE_BUY_THETA = "Cumulative Buy theta"
const CUMULATIVE_SELL_THETA = "Cumulative Sell theta"

const EXPECTED_IMBALANCE = "expected_imbalance"
const EXPECTED_TICKS_NUMBER = "exp_num_ticks"

const EXPECTED_BUY_IMBALANCE = "exp_imbalance_buy"
const EXPECTED_SELL_IMBALANCE = "exp_imbalance_sell"
const EXPECTED_BUY_TICKS_PROPORTION = "exp_buy_ticks_proportion"
const BUY_TICKS_NUMBER = "buy_ticks_num"

const N_TICKS_ON_BAR_FORMATION = "Number of ticks while bar is formed."

const PREVIOUS_TICK_RULE = "Previous tick rule"
const EXPECTED_IMBALANCE_WINDOW = "Expected Imbalance Window"

const PREVIOUS_BARS_N_TICKS_LIST = "List of previous bars number of ticks"
const PREVIOUS_TICK_IMBALANCES_LIST = "List of previous tick imbalances"

const PREVIOUS_TICK_IMBALANCES_BUY_LIST = "List of previous (buy) tick imbalances"
const PREVIOUS_TICK_IMBALANCES_SELL_LIST = "List of previous (sell) tick imbalances"

const PREVIOUS_BARS_BUY_TICKS_PROPORTIONS_LIST = "List of previous bars buy ticks proportion"

const N_PREVIOUS_BARS_FOR_EXPECTED_N_TICKS_ESTIMATION = "Window size for E[T]"
