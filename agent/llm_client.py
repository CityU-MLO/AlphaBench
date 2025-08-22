import time
from openai import OpenAI
import requests
import yaml
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

API_KEY_PATH = "./config/api_keys.yaml"
# Load API keys from YAML file
with open(API_KEY_PATH, "r") as file:
    api_keys = yaml.safe_load(file)

# Assign to variables
DEEPSEEK_API_KEY = api_keys.get("DEEPSEEK_API_KEY", "")
OPENAI_API_KEY = api_keys.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = api_keys.get("GEMINI_API_KEY", "")
CLAUDE_API_KEY = api_keys.get("CLAUDE_API_KEY", "")


def call_llm(
    prompt,
    model="deepseek-chat",
    api_key=None,
    base_url=None,
    json_output=False,
    system_prompt="You are a helpful assistant.",
    temperature=1.0,
    local=False,
    local_port=8000,
    return_raw=False,
    max_try=5,
):
    # If local is True, send request to local server
    if local:
        url = f"http://localhost:{local_port}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }
        if json_output:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Local model server error: {e}")

    # Otherwise, use API-based client
    if model.startswith("gpt-"):
        client = OpenAI(
            api_key=api_key or OPENAI_API_KEY,
            base_url=base_url or "https://api.openai-proxy.com/v1",
        )
    elif model.startswith("gemini-"):
        client = OpenAI(
            api_key=api_key or GEMINI_API_KEY,
            base_url=base_url or "https://gemini-openai-proxy.deno.dev/v1",
        )
    # elif model.startswith("claude-"):
    #     client = OpenAI(
    #         api_key=api_key or CLAUDE_API_KEY,
    #         base_url=base_url or "https://claude-code-proxy.suixifa.workers.dev/https/api.anthropic.com/v1/messages",
    #     )
    else:
        client = OpenAI(
            api_key=api_key or DEEPSEEK_API_KEY,
            base_url=base_url or "https://api.deepseek.com",
        )

    request_params = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "stream": False,
    }
    if json_output:
        request_params["response_format"] = {"type": "json_object"}

    for attempt in range(max_try):
        try:
            response = client.chat.completions.create(**request_params)
            if return_raw:
                return response
            else:
                return response.choices[0].message.content.strip()
        except Exception as e:
            if "Too Many Requests" in str(e) and attempt < max_try - 1:
                sleep_time = 0.25
                print(f"Rate limited (API). Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
                continue
            return f"LLM API error: {e}"


def _worker_call_llm(idx, prompt, kwargs):
    """包装 call_llm 的子进程函数"""
    if kwargs.get("latency"):
        time.sleep(kwargs["latency"])
    res = call_llm(
        prompt,
        model=kwargs["model"],
        api_key=kwargs["api_key"],
        base_url=kwargs["base_url"],
        json_output=kwargs["json_output"],
        system_prompt=kwargs["system_prompt"],
        temperature=kwargs["temperature"],
        local=kwargs["local"],
        local_port=kwargs["local_port"],
        return_raw=kwargs["return_raw"],
    )
    return idx, res


def batch_call_llm(
    prompts,
    model="deepseek-chat",
    api_key=None,
    base_url=None,
    json_output=False,
    system_prompt="You are a helpful assistant.",
    temperature=1.0,
    local=False,
    local_port=8000,
    latency=None,
    num_workers=4,
    return_raw=False,
    verbose=False,
    parallel=True,
    timeout: float = 60.0,  # 超时时间（秒）
):
    """
    Batch call LLM with hard timeout (kills process if exceeded).
    """
    results = [None] * len(prompts)
    if not prompts:
        return results

    kwargs = dict(
        model=model,
        api_key=api_key,
        base_url=base_url,
        json_output=json_output,
        system_prompt=system_prompt,
        temperature=temperature,
        local=local,
        local_port=local_port,
        return_raw=return_raw,
        latency=latency,
    )

    max_workers = max(1, num_workers if parallel else 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_worker_call_llm, i, p, kwargs): i
            for i, p in enumerate(prompts)
        }

        futures_iter = (
            tqdm(futures, total=len(prompts), desc="Processing") if verbose else futures
        )

        for fut in futures_iter:
            idx = futures[fut]
            try:
                idx, res = fut.result(timeout=timeout)  # 超时会直接 kill 子进程
                results[idx] = res
            except TimeoutError:
                fut.cancel()
                print(f"[Timeout] prompt {idx} exceeded {timeout}s and was terminated")
                results[idx] = {
                    "success": False,
                    "error_type": "TIMEOUT",
                    "error_message": f"Request exceeded {timeout}s and was terminated",
                }
            except Exception as e:
                results[idx] = None
                if verbose:
                    print(f"[Error] prompt {idx}: {e}")

    return results


# 🧪 Main test function
def main():
    prompt = (
        "Write a Python function to calculate the Sharpe ratio and explain each line."
    )

    # print("=== 🧪 Claude Client Test ===")
    # try:
    #     result_claude = call_llm(
    #         prompt,
    #         model="claude-sonnet-4-20250514",
    #     )
    #     print(result_claude)
    # except Exception as e:
    #     print(f"Claude Client Error: {e}")

    print("=== 🧪 Gemini Client Test ===")
    try:
        result_gm = call_llm(prompt, model="gemini-2.5-pro")
        print(result_gm)
    except Exception as e:
        print(f"DeepSeek API Error: {e}")

    print("=== 🔍 DeepSeek API Test ===")
    try:
        result_ds = call_llm(prompt, model="deepseek-chat")
        print(result_ds)
    except Exception as e:
        print(f"DeepSeek API Error: {e}")

    print("\n=== 🤖 OpenAI GPT-4 Test ===")
    try:
        result_gpt = call_llm(prompt, model="gpt-4")
        print(result_gpt)
    except Exception as e:
        print(f"GPT-4 Error: {e}")

    print("\n=== 🖥️ Local Model Server Test ===")
    try:
        result_local = call_llm(
            prompt,
            model="deepseek-ai/deepseek-llm-7b-chat",
            local=True,
            local_port=8000,  # change if your server uses a different port
        )
        print(result_local)
    except Exception as e:
        print(f"Local Server Error: {e}")


def test_batch_call_llm():
    # 8 meaningful short questions
    prompts = [
        "What is AI?",
        "Define machine learning.",
        "What is deep learning?",
        "Explain overfitting.",
        "What is a neural network?",
        "Define reinforcement learning.",
        "What is natural language processing?",
        "Explain gradient descent.",
    ]

    # Run batch call with 4 workers
    results = batch_call_llm(
        prompts,
        model="gpt-4.1",  # replace with the model you want
        local=False,  # set False if using API
        num_workers=4,
        verbose=True,  # show progress bar
    )

    # Print results
    for i, (prompt, result) in enumerate(zip(prompts, results)):
        print(f"[{i}] Prompt: {prompt}")
        print(f"    Result: {result}\n")


if __name__ == "__main__":
    # test_batch_call_llm()
    main()
