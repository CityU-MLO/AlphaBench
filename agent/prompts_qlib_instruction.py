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
