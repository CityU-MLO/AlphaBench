"""
LLM client for AlphaBench.

Routing is fully config-driven via config/api_keys.yaml:

  providers.<name>   → per-provider api_key + base_url
  all_in_one         → single-endpoint platform serving many models
  local              → local vLLM server (always wins when local=True)
  models.<name>
    all_in_one: true   → route to all_in_one platform
    provider: <name>   → route to named provider
    (absent)           → defaults to all_in_one

Usage:
    from agent.llm_client import call_llm, batch_call_llm

    result = call_llm("Explain IC in quant finance.", model="gpt-4.1")
    results = batch_call_llm(prompts, model="deepseek-chat", num_workers=8)
"""

from __future__ import annotations

import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests
import yaml
from openai import OpenAI
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "api_keys.yaml")


def _load_config() -> dict:
    path = os.path.abspath(_CONFIG_PATH)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_CFG: dict = _load_config()


# ---------------------------------------------------------------------------
# Environment-variable resolution
# ---------------------------------------------------------------------------

def _resolve_value(value: Optional[str]) -> str:
    """
    Expand environment-variable references in a config string.

    Supports two formats:
      "${MY_VAR}"   → os.environ["MY_VAR"]  (raises if unset and no fallback)
      "sk-abc..."   → returned as-is

    Args:
        value: Raw string from config (may be None).

    Returns:
        Resolved string, or "" if value is None.

    Raises:
        EnvironmentError: If a ${VAR} reference is used but the variable is not set.
    """
    if not value:
        return ""
    val = str(value).strip()
    if val.startswith("${") and val.endswith("}"):
        env_var = val[2:-1]
        resolved = os.environ.get(env_var)
        if not resolved:
            raise EnvironmentError(
                f"Environment variable '{env_var}' is not set.\n"
                f"  Set it with:  export {env_var}=<your-api-key>\n"
                f"  Or replace '${{env_var}}' with a literal key in config/api_keys.yaml."
            )
        return resolved
    return val


# ---------------------------------------------------------------------------
# Client resolver
# ---------------------------------------------------------------------------

def _resolve_client(
    model: str,
    local: bool = False,
    local_port: int = 8000,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 120,
) -> tuple[Optional[OpenAI], bool, Optional[str]]:
    """
    Determine how to connect for a given model.

    Returns:
        (client, is_local, local_url)
        - If is_local is True, client is None and local_url contains the endpoint.
        - Otherwise client is an OpenAI-compatible client and local_url is None.
    """
    # ── 1. Local vLLM always wins ────────────────────────────────────────────
    if local:
        local_cfg = _CFG.get("local", {})
        url_template = local_cfg.get("base_url", "http://localhost:{port}/v1")
        local_url = url_template.format(port=local_port)
        return None, True, local_url

    # ── 2. Explicit api_key + base_url override ──────────────────────────────
    if api_key and base_url:
        return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout), False, None

    # ── 3. Look up per-model routing in config ───────────────────────────────
    model_cfg: dict = _CFG.get("models", {}).get(model, {})

    # 3a. Named provider (e.g. provider: deepseek)
    provider_name = model_cfg.get("provider")
    if provider_name:
        prov = _CFG.get("providers", {}).get(provider_name, {})
        return OpenAI(
            api_key=api_key or _resolve_value(prov.get("api_key")),
            base_url=base_url or prov.get("base_url", ""),
            timeout=timeout,
        ), False, None

    # 3b. All-in-One (explicit tag OR missing entry → default)
    aio = _CFG.get("all_in_one", {})
    return OpenAI(
        api_key=api_key or _resolve_value(aio.get("api_key")),
        base_url=base_url or aio.get("base_url", ""),
        timeout=timeout,
    ), False, None


# ---------------------------------------------------------------------------
# call_llm
# ---------------------------------------------------------------------------

def call_llm(
    prompt: str,
    model: str = "deepseek-chat",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    json_output: bool = False,
    system_prompt: str = "You are a helpful assistant.",
    temperature: float = 1.0,
    local: bool = False,
    local_port: int = 8000,
    return_raw: bool = False,
    max_try: int = 5,
    timeout: int = 120,
    save_raw_dir: Optional[str] = None,
    service_provider: Optional[str] = None,  # deprecated, routing now in config
) -> Any:
    """
    Call an LLM and return the response string (or raw response object).

    Routing is determined by config/api_keys.yaml:
      - local=True     → local vLLM server
      - model routing  → all_in_one or per-provider (from models.<name> in config)
      - api_key+base_url kwargs override config when both are provided

    Args:
        prompt:        User prompt string.
        model:         Model name (e.g. "gpt-4.1", "deepseek-chat").
        api_key:       Override API key (both api_key and base_url must be set to take effect).
        base_url:      Override base URL.
        json_output:   Request JSON response format.
        system_prompt: System-level instruction.
        temperature:   Sampling temperature.
        local:         If True, send request to local vLLM server.
        local_port:    Port of local vLLM server.
        return_raw:    Return the raw OpenAI response object instead of string.
        max_try:       Retry attempts on rate-limit errors.
        timeout:       HTTP timeout in seconds.
        save_raw_dir:  If set, pickle-dump raw responses to this directory.

    Returns:
        Response string, or raw response object if return_raw=True.
    """
    client, is_local, local_url = _resolve_client(
        model, local=local, local_port=local_port,
        api_key=api_key, base_url=base_url, timeout=timeout,
    )

    # ── Local vLLM path ──────────────────────────────────────────────────────
    if is_local:
        payload: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
            "temperature": temperature,
        }
        if json_output:
            payload["response_format"] = {"type": "json_object"}
        try:
            resp = requests.post(local_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Local vLLM server error: {e}") from e

    # ── API path ─────────────────────────────────────────────────────────────
    request_params: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
        "temperature": temperature,
        "stream": False,
    }
    if json_output:
        request_params["response_format"] = {"type": "json_object"}

    for attempt in range(max_try):
        try:
            response = client.chat.completions.create(**request_params)

            if save_raw_dir:
                os.makedirs(save_raw_dir, exist_ok=True)
                fname = time.strftime("%Y%m%d_%H%M%S", time.localtime()) + ".pkl"
                with open(os.path.join(save_raw_dir, fname), "wb") as f:
                    pickle.dump(response, f)

            if return_raw:
                return response
            return response.choices[0].message.content.strip()

        except Exception as e:
            if "Too Many Requests" in str(e) and attempt < max_try - 1:
                time.sleep(0.25)
                print(f"Rate limited. Retrying ({attempt + 1}/{max_try})...")
                continue
            return f"LLM API error: {e}"


# ---------------------------------------------------------------------------
# batch_call_llm
# ---------------------------------------------------------------------------

def batch_call_llm(
    prompts: List[str],
    model: str = "deepseek-chat",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    json_output: bool = False,
    system_prompt: str = "You are a helpful assistant.",
    temperature: float = 1.0,
    local: bool = False,
    local_port: int = 8000,
    latency: Optional[float] = None,
    num_workers: int = 4,
    return_raw: bool = False,
    verbose: bool = False,
    timeout: Optional[int] = None,
    service_provider: Optional[str] = None,  # deprecated, routing now in config
) -> List[Any]:
    """
    Batch-call an LLM in parallel, preserving prompt order.

    Args:
        prompts:      List of user prompt strings.
        latency:      Optional sleep (seconds) before each request — useful for
                      rate-limiting without reducing parallelism.
        num_workers:  Number of parallel threads.
        verbose:      Show tqdm progress bar.
        (other args): Same as call_llm.

    Returns:
        List of response strings in the same order as prompts.
        Entries are None if the underlying call raises an unhandled exception.
    """
    results: List[Any] = [None] * len(prompts)

    _call_kwargs = dict(
        model=model,
        api_key=api_key,
        base_url=base_url,
        json_output=json_output,
        system_prompt=system_prompt,
        temperature=temperature,
        local=local,
        local_port=local_port,
        return_raw=return_raw,
        timeout=timeout or 120,
    )

    def _worker(idx: int, prompt: str):
        try:
            if latency:
                time.sleep(latency)
            return idx, call_llm(prompt, **_call_kwargs)
        except Exception:
            return idx, None

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_worker, i, p) for i, p in enumerate(prompts)]
        iter_ = (
            tqdm(as_completed(futures), total=len(prompts), desc="LLM batch")
            if verbose
            else as_completed(futures)
        )
        for f in iter_:
            idx, res = f.result()
            results[idx] = res

    return results


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

def main():
    prompt = "Write a Python function to calculate the Sharpe ratio and explain each line."

    print("=== Gemini Test ===")
    try:
        print(call_llm(prompt, model="gemini-2.5-flash"))
    except Exception as e:
        print(f"Error: {e}")

    print("\n=== DeepSeek Test ===")
    try:
        print(call_llm(prompt, model="deepseek-chat"))
    except Exception as e:
        print(f"Error: {e}")

    print("\n=== GPT-4.1 Test ===")
    try:
        print(call_llm(prompt, model="gpt-4.1"))
    except Exception as e:
        print(f"Error: {e}")

    print("\n=== Local vLLM Test ===")
    try:
        print(call_llm(prompt, model="Qwen2.5-72B-Instruct", local=True, local_port=8000))
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
