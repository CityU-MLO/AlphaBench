QLIB_GENERATE_INSTRUCTION = """
**Arithmetic / Logic**
- Add(x,y), Sub(x,y), Mul(x,y), Div(x,y)  
- Power(x,y), Log(x), Sqrt(x), Abs(x), Sign(x), Delta(x,n)  
- And(x,y), Or(x,y), Not(x)  
- Sqrt(x), Tanh(x) 
- Comparators: Greater(x,y), Less(x,y), Gt(x,y), Ge(x,y), Lt(x,y), Le(x,y), Eq(x,y), Ne(x,y)

**Rolling (n is positive integer)**
- Mean(x,n), Std(x,n), Var(x,n), Max(x,n), Min(x,n)  
- Skew(x,n), Kurt(x,n), Sum(x,n), Med(x,n), Mad(x,n), Count(x,n)  | Med is for median and Mad for Mean Absolute Deviation
- EMA(x,n), WMA(x,n), Corr(x,y,n), Cov(x,y,n)  
- Slope(x,n), Rsquare(x,n), Resi(x,n)

**Ranking / Conditional**
- Rank(x,n), Ref(x,n), IdxMax(x,n), IdxMin(x,n), Quantile(x,n,qscore (float number between 0-1)) 
- If(cond,x,y), Mask(cond,x), Clip(x,a,b)

Note: function signatures must be complete.  
- Corr(x,y,n) requires 3 arguments 
- Quantile(x,n,qscore) requires 3 arguments
- Rank(x,n) requires 2 arguments  
- Ref(x,n) requires 2 arguments

Important rules:
a. For arithmetic operations, do NOT use symbols. Instead, use: Add for +, Sub for -, Mul for *, Div for /
b. Parentheses must balance.  
c. Correct arity — no missing arguments.  
d. Rolling windows (n) must be positive integers.  
e. Division safety — always add epsilon:  
   - Div(x, Add(den, 1e-12)) correct
   - Div(x, den) incorrect
   Sqrt safely, ensure no negative inputs.
f. No undefined / banned functions (e.g., SMA, RSI), and above operation is low/upper-case sensitive.
g. Expressions must be plain strings, no comments or backticks.

"""


# Assay-native operator guide. Appended to the generation/search system prompt
# only when the active evaluation engine is "assay" (see ffo.utils.assay_engine).
# The Assay evaluator accepts BOTH Qlib-style (above) and Assay-native syntax;
# this section unlocks Assay's richer operator set for the LLM.
ASSAY_GENERATE_INSTRUCTION = """
**Assay-native syntax (also accepted — the evaluator parses both Qlib-style and Assay-native)**

You MAY write factors in Assay-native syntax to use a richer operator set. Conventions:

- Fields are bare (no `$`): open, high, low, close, volume, vwap
- Time-series ops are `ts_`-prefixed and take a window n (positive integer):
  ts_mean(x,n), ts_std(x,n), ts_sum(x,n), ts_max(x,n), ts_min(x,n),
  ts_delay(x,n) [= Ref], ts_delta(x,n) [= Delta], ts_returns(x,n),
  ts_corr(x,y,n), ts_cov(x,y,n), ts_rank(x,n), ts_argmax(x,n), ts_argmin(x,n),
  ts_ema(x,n), ts_dema(x,n), ts_decay_linear(x,n)
- Cross-sectional ops are `cs_`-prefixed (per-date, no window):
  cs_rank(x), cs_zscore(x), cs_scale(x), cs_demean(x), cs_winsorize(x,p),
  cs_neutralize(x,'sector'), cs_group_rank(x,g), cs_group_mean(x,g)
- Math / element-wise: abs(x), log(x), sign(x), sqrt(x), pow(x,e),
  signed_power(x,e) [= sign(x)*abs(x)^e], elem_min(x,y), elem_max(x,y),
  where(cond,a,b), clip(x,lo,hi), sigmoid(x), safe_div(a,b,fill=0)
- Arithmetic operators +, -, *, / may be written directly,
  e.g. ts_mean(volume,5)/ts_mean(volume,20)
- Macro: adv{n} expands to ts_mean(volume,n)  (e.g. adv20)

Key distinctions (do not confuse):
- cs_rank(x) is cross-sectional rank (1 arg); ts_rank(x,n) is time-series rank (2 args).
- elem_min/elem_max are element-wise; ts_min/ts_max are rolling over a window.
- signed_power keeps the sign; pow does not.

Pick ONE dialect per factor — do not mix Qlib-style ($close, Mean) and Assay-native
(close, ts_mean) inside the same expression. Prefer richer Assay ops (cs_zscore,
ts_ema, ts_rank, signed_power) where they fit the idea.
"""
