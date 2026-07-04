# FFO Switchable Evaluation Engine (Qlib / Assay)

FFO can evaluate factors with one of two interchangeable engines, selected at
backend startup. Every downstream consumer (searcher, agent, benchmark, web UI,
`ffo.api`, `ffo.client`) is unchanged — they always talk to the same FFO REST
contract on `:19777` (`/factors/eval`, `/factors/check`, `/factors/portfolio`).

| Engine | Value | How it computes |
|--------|-------|-----------------|
| Qlib (default) | `qlib` | In-process Qlib worker pool (needs local `~/.qlib` data) |
| Assay | `assay` | Delegates to the [Assay](https://github.com/chester1uo/Assay) platform's REST API over HTTP |

Assay runs on Python ≥3.10 in its own virtual-env, while FFO runs on the Qlib
(Python 3.9) interpreter, so they cannot share a process — the `assay` engine
talks to Assay over HTTP. Qlib-style expressions (`$close`, `Mean(...)`,
`Corr(...)`) are accepted by Assay's parser directly, so no rewriting is needed.

## Usage

Start the Assay backend (it must already be running, e.g. via
`python -m assay.cli serve-api --port 8000`), then start FFO on the assay engine:

```bash
# from the repo root, in the FFO (qlib) python env
cd ffo
FFO_ENGINE=assay FFO_ASSAY_URL=http://127.0.0.1:8000 python backend_app.py
```

Confirm the engine:

```bash
curl -s http://127.0.0.1:19777/health | python -m json.tool
# -> "engine": "assay", "assay": {"reachable": true, ...}
```

Everything else is identical to normal FFO usage:

```bash
curl -s -X POST http://127.0.0.1:19777/factors/eval \
  -H 'Content-Type: application/json' \
  -d '{"expression":"Mean($volume,5)/Mean($volume,20)","market":"csi300",
       "start":"2023-01-03","end":"2024-01-01","fast":true}'
```

## Configuration

Set in `ffo/config/ffo.yaml` (`engine:` block) or override per run with env vars:

| Key (`ffo.yaml`) | Env var | Default | Meaning |
|------------------|---------|---------|---------|
| `engine.backend` | `FFO_ENGINE` | `qlib` | `qlib` or `assay` |
| `engine.assay_url` | `FFO_ASSAY_URL` | `http://127.0.0.1:8000` | Assay REST base url |
| `engine.assay_timeout` | `FFO_ASSAY_TIMEOUT` | `180` | per-request timeout (s) |
| `engine.assay_execution` | `FFO_ASSAY_EXECUTION` | _(Assay default)_ | fill model, e.g. `close`, `next_open` |
| `engine.assay_adj` | `FFO_ASSAY_ADJ` | _(Assay default)_ | price adjustment, e.g. `split` |

Each market maps to an Assay universe via `markets.<m>.assay_universe`
(`csi300→CSI300`, `csi500→CSI500`, `csi1000→CSI1000`, `sp500→SP500`,
`nasdaq100→NASDAQ100`).

## Metric mapping (Assay `FactorReport` → FFO `metrics`)

| FFO metric | Assay field |
|------------|-------------|
| `ic`, `rank_ic`, `icir`, `rank_icir` | same names |
| `ir` (alias) | `icir` |
| `turnover` | `turnover_1d` |
| `n_dates` | `n_dates` |
| `success` / `error` | `failure_mode is None` / `failure_mode` + `suggestion` |

## Factor generation in Assay format

When the `assay` engine is active, factor *generation* (T1) and *search* (T3) adapt:

- **Prompts are enriched** with the Assay-native operator set (`ts_*`, `cs_*`,
  bare field names, `signed_power`, `cs_zscore`, …) via `ASSAY_GENERATE_INSTRUCTION`
  in `agent/prompts_qlib_instruction.py`. Both Qlib-style and Assay-native syntax
  are accepted by the evaluator; the LLM is told to pick one dialect per factor.
- **Validation defers to Assay.** The T1 generator skips the Qlib-only complexity
  check and relies on `check_factor_via_api` (Assay lint); T3 already validated via
  the API. Engine detection: `ffo.utils.assay_engine.is_assay()` (reads
  `FFO_ENGINE` / `engine.backend`).
- **Point the agent at the Assay backend.** The FFO client now resolves its URL
  from config/env (`FFO_BACKEND_URL`, or `FFO_BACKEND_PORT` via `backend_url`)
  instead of hardcoding `:19777`. So run the benchmark with the same backend the
  Assay engine serves, e.g.:

  ```bash
  FFO_ENGINE=assay FFO_BACKEND_PORT=19778 python example/search/run_t3_search.py \
      --config example/search/configs/search_csi300.yaml --model_name <model>
  ```

  (Or set `engine.backend: assay` + `server.backend.port: 19778` in `ffo/config/ffo.yaml`
  so both the backend and the agent agree from one config.)

To keep the native Qlib path unchanged, none of this activates unless the engine is `assay`.

## Notes / current limits of the `assay` engine

- `/factors/eval` returns IC-family metrics for both `fast=true` and
  `fast=false`. The Qlib-style portfolio backtest (`portfolio_metrics` /
  `portfolio_details` attached when `fast=false`) is **not** produced by this
  engine — use Assay's own `/v1/portfolio/backtest` for portfolio analytics.
- `daily_metrics` is populated only when Assay returns per-date series
  (off by default); consumers read the aggregate `metrics`, so this is a no-op
  for the search/eval paths.
- `/factors/portfolio` reproduces FFO's "z-score normalize + equal-weight
  average" combine by evaluating `(cs_zscore(c1)+…)/N` on Assay.

Implementation: [`ffo/utils/assay_engine.py`](utils/assay_engine.py); wired into
[`ffo/routes/factors.py`](routes/factors.py) and [`ffo/backend_app.py`](backend_app.py).
