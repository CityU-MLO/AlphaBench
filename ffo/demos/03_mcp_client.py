#!/usr/bin/env python
"""
Demo 06 — FFO MCP Server: Complete Usage Guide
================================================
Comprehensive demos for the FFO MCP (Model Context Protocol) server,
covering all connection methods and integration patterns.

Contents:
  Demo A — Direct tool calls (no MCP transport, fastest way to test)
  Demo B — MCP stdio client (spawn server as subprocess, like Claude Desktop)
  Demo C — MCP SSE client (connect to a running SSE server)
  Demo D — Claude Agent SDK integration (build an autonomous agent)
  Demo E — IDE / client configuration examples

Prerequisites:
  1. Start the FFO backend:   ppo start backend
  2. Install MCP SDK:         pip install mcp anthropic-sdk

Run all demos:
    python demos/06_mcp_demos.py

Run a specific demo:
    python demos/06_mcp_demos.py --demo a
    python demos/06_mcp_demos.py --demo b
    python demos/06_mcp_demos.py --demo c
    python demos/06_mcp_demos.py --demo d
    python demos/06_mcp_demos.py --demo e
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


# ═══════════════════════════════════════════════════════════════════════════════
# Demo A — Direct Tool Calls (no MCP transport)
# ═══════════════════════════════════════════════════════════════════════════════

def demo_a_direct_calls():
    """
    Call MCP tool functions directly — the fastest way to test.

    These are the exact same functions exposed as MCP tools.
    No transport, no server process — just plain Python calls.
    """
    print("=" * 70)
    print("Demo A: Direct Tool Calls (no MCP transport)")
    print("=" * 70)

    from ffo.mcp.server import (
        check_factor_syntax,
        batch_check_factors,
        evaluate_factor,
        batch_evaluate_factors,
        get_server_health,
        get_cache_stats,
    )

    # ── 1. Health check ──────────────────────────────────────────────────────
    print("\n[A.1] Health check")
    health = get_server_health()
    print(f"  Healthy  : {health['is_healthy']}")
    print(f"  Latency  : {health.get('latency_ms', 0):.0f} ms")

    if not health["is_healthy"]:
        print("\n  Backend not running. Start with: ppo start backend")
        return

    # ── 2. Syntax validation ─────────────────────────────────────────────────
    print("\n[A.2] Syntax check")
    for expr in ["Rank($close, 20)", "BadOp($close)"]:
        result = check_factor_syntax(expr)
        status = "VALID" if result["is_valid"] else f"INVALID: {result['error']}"
        print(f"  {expr:30s} → {status}")

    # ── 3. Single factor evaluation ──────────────────────────────────────────
    print("\n[A.3] Single factor evaluation")
    result = evaluate_factor(
        expression="Rank($close, 20)",
        market="csi300",
        start="2023-01-01",
        end="2024-01-01",
        fast=True,
    )
    if result["success"]:
        m = result["metrics"]
        print(f"  Expression : Rank($close, 20)")
        print(f"  IC         : {m['ic']:+.4f}")
        print(f"  Rank IC    : {m['rank_ic']:+.4f}")
        print(f"  ICIR       : {m['icir']:+.4f}")
        print(f"  Cached     : {result.get('cached', False)}")
    else:
        print(f"  Error: {result['error']}")

    # ── 4. Batch evaluation ──────────────────────────────────────────────────
    print("\n[A.4] Batch evaluation (3 factors)")
    candidates = [
        "Rank($close, 5)",
        "Mean($volume, 5) / Mean($volume, 20)",
        "Corr($close, $volume, 10)",
    ]
    results = batch_evaluate_factors(
        expressions=candidates,
        market="csi300",
        fast=True,
        max_workers=4,
    )
    print(f"  {'Expression':<45} {'IC':>7} {'Rank IC':>9}")
    print(f"  {'─' * 45} {'─' * 7} {'─' * 9}")
    for r in results:
        if r["success"]:
            m = r["metrics"]
            print(f"  {r['expression']:<45} {m['ic']:>+7.4f} {m['rank_ic']:>+9.4f}")
        else:
            print(f"  {r['expression']:<45} ERROR: {r['error']}")

    # ── 5. Batch syntax check — speed benchmark (10 factors) ────────────────
    print("\n[A.5] Batch syntax check — speed benchmark (10 factors)")
    import time as _time

    check_factors = [
        # ── Price momentum / reversal
        "Rank($close, 5)",
        "Rank($close, 20)",
        "$close / Delay($close, 5) - 1",
        "$close / Min($close, 20) - 1",
        # ── Volume dynamics
        "Mean($volume, 5) / Mean($volume, 20)",
        "Rank($volume, 10)",
        # ── Volatility
        "Std($close / Delay($close, 1) - 1, 20)",
        # ── Price-volume interaction
        "Corr($close, $volume, 10)",
        # ── High-low range
        "($high - $low) / $close",
        # ── Combined
        "Rank($close, 20) - Rank($volume, 20)",
    ]

    # --- Benchmark A: one-by-one (sequential) ---
    print(f"\n  Strategy A — sequential check_factor_syntax() × {len(check_factors)}")
    t0 = _time.perf_counter()
    seq_results = []
    for expr in check_factors:
        seq_results.append(check_factor_syntax(expr))
    seq_elapsed = _time.perf_counter() - t0
    seq_valid = sum(1 for r in seq_results if r["is_valid"])
    print(f"    Valid : {seq_valid}/{len(check_factors)}")
    print(f"    Time  : {seq_elapsed:.3f} s")

    # --- Benchmark B: single batch request ---
    print(f"\n  Strategy B — batch_check_factors() (single request)")
    t0 = _time.perf_counter()
    batch_results = batch_check_factors(expressions=check_factors)
    batch_elapsed = _time.perf_counter() - t0
    batch_valid = sum(1 for r in batch_results if r["is_valid"])
    print(f"    Valid : {batch_valid}/{len(check_factors)}")
    print(f"    Time  : {batch_elapsed:.3f} s")

    # --- Comparison ---
    speedup = seq_elapsed / batch_elapsed if batch_elapsed > 0 else float("inf")
    print(f"\n  Speedup : {speedup:.1f}×  (batch vs sequential)")

    # --- Per-factor detail ---
    print(f"\n  {'Expression':<48} {'Valid':>6}")
    print(f"  {'─' * 48} {'─' * 6}")
    for r in batch_results:
        expr = r["expression"]
        if len(expr) > 46:
            expr = expr[:43] + "..."
        status = "  ✓" if r["is_valid"] else f"  ✗ {r.get('error', '')[:30]}"
        print(f"  {expr:<48} {status}")

    # ── 6. Cache stats ───────────────────────────────────────────────────────
    print("\n[A.6] Cache statistics")
    stats = get_cache_stats()
    print(f"  Entries : {stats['cache_size']} / {stats['max_cache_size']}")

    print("\nDemo A complete.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Demo B — MCP stdio Client (spawn server as subprocess)
# ═══════════════════════════════════════════════════════════════════════════════

async def demo_b_stdio_client():
    """
    Connect to the FFO MCP server via stdio transport.

    This is the same mechanism Claude Desktop and Cursor use:
    the client spawns the server as a subprocess and communicates
    via stdin/stdout using the MCP JSON-RPC protocol.
    """
    print("=" * 70)
    print("Demo B: MCP stdio Client (subprocess transport)")
    print("=" * 70)

    try:
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client, StdioServerParameters
    except ImportError:
        print("\n  MCP SDK not installed. Install with:")
        print("    pip install mcp")
        print("\nSkipping Demo B.\n")
        return

    # Server command — same as what Claude Desktop would use
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "ffo.mcp.server"],
    )

    print("\n[B.1] Spawning FFO MCP server via stdio...")
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print("  Connected and initialized.")

            # ── List available tools ─────────────────────────────────
            print("\n[B.2] Discovering tools...")
            tools = await session.list_tools()
            print(f"  Found {len(tools.tools)} tools:")
            for tool in tools.tools:
                desc = (tool.description or "").split("\n")[0][:60]
                print(f"    - {tool.name}: {desc}")

            # ── List resources ───────────────────────────────────────
            print("\n[B.3] Discovering resources...")
            resources = await session.list_resources()
            print(f"  Found {len(resources.resources)} resources:")
            for res in resources.resources:
                print(f"    - {res.uri}")

            # ── Read a resource ──────────────────────────────────────
            print("\n[B.4] Reading ffo://operators resource...")
            content = await session.read_resource("ffo://operators")
            text = content.contents[0].text if content.contents else ""
            # Print first 5 lines
            for line in text.strip().split("\n")[:5]:
                print(f"    {line}")
            print("    ...")

            # ── Call get_server_health ────────────────────────────────
            print("\n[B.5] Calling get_server_health()...")
            result = await session.call_tool("get_server_health", arguments={})
            health = json.loads(result.content[0].text)
            print(f"  Healthy : {health['is_healthy']}")
            print(f"  Latency : {health.get('latency_ms', 0):.0f} ms")

            if not health["is_healthy"]:
                print("\n  Backend not running. Start with: ppo start backend")
                print("  (The MCP server is running, but it needs the backend.)")
                print("\nDemo B complete (partial).\n")
                return

            # ── Call check_factor_syntax ──────────────────────────────
            print("\n[B.6] Calling check_factor_syntax('Rank($close, 20)')...")
            result = await session.call_tool(
                "check_factor_syntax",
                arguments={"expression": "Rank($close, 20)"},
            )
            check = json.loads(result.content[0].text)
            print(f"  Valid: {check['is_valid']}")

            # ── Call evaluate_factor ──────────────────────────────────
            print("\n[B.7] Calling evaluate_factor('Rank($close, 20)')...")
            result = await session.call_tool(
                "evaluate_factor",
                arguments={
                    "expression": "Rank($close, 20)",
                    "market": "csi300",
                    "start": "2023-01-01",
                    "end": "2024-01-01",
                    "fast": True,
                },
            )
            eval_result = json.loads(result.content[0].text)
            if eval_result["success"]:
                m = eval_result["metrics"]
                print(f"  IC      : {m['ic']:+.4f}")
                print(f"  Rank IC : {m['rank_ic']:+.4f}")
                print(f"  ICIR    : {m['icir']:+.4f}")
            else:
                print(f"  Error: {eval_result['error']}")

            # ── Call batch_evaluate_factors ────────────────────────────
            print("\n[B.8] Calling batch_evaluate_factors([...])...")
            result = await session.call_tool(
                "batch_evaluate_factors",
                arguments={
                    "expressions": [
                        "Rank($close, 5)",
                        "Rank($close, 20)",
                        "Mean($volume, 5) / Mean($volume, 20)",
                    ],
                    "market": "csi300",
                    "fast": True,
                },
            )
            batch_results = json.loads(result.content[0].text)
            for r in batch_results:
                expr = r["expression"][:40]
                if r["success"]:
                    ic = r["metrics"]["ic"]
                    print(f"  {expr:<42} IC={ic:+.4f}")
                else:
                    print(f"  {expr:<42} ERROR")

    print("\nDemo B complete.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Demo C — MCP SSE Client (connect to running server)
# ═══════════════════════════════════════════════════════════════════════════════

async def demo_c_sse_client():
    """
    Connect to an FFO MCP server running with SSE transport.

    Start the server first:
        python -m ffo.mcp.server --transport sse --port 8765
    or:
        ppo start mcp --transport sse --port 8765
    """
    print("=" * 70)
    print("Demo C: MCP SSE Client (HTTP transport)")
    print("=" * 70)

    try:
        from mcp import ClientSession
        from mcp.client.sse import sse_client
    except ImportError:
        print("\n  MCP SDK not installed. Install with:")
        print("    pip install mcp")
        print("\nSkipping Demo C.\n")
        return

    sse_url = "http://localhost:8765/sse"
    print(f"\n[C.1] Connecting to {sse_url}...")
    print("  (Make sure server is running: python -m ffo.mcp.server --transport sse)")

    try:
        async with sse_client(sse_url) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                print("  Connected!")

                # ── List tools ───────────────────────────────────────
                tools = await session.list_tools()
                print(f"\n[C.2] Available tools: {[t.name for t in tools.tools]}")

                # ── Health check ─────────────────────────────────────
                print("\n[C.3] Health check via SSE...")
                result = await session.call_tool("get_server_health", arguments={})
                health = json.loads(result.content[0].text)
                print(f"  Healthy: {health['is_healthy']}")

                if not health["is_healthy"]:
                    print("  Backend not running. Start with: ppo start backend")
                    print("\nDemo C complete (partial).\n")
                    return

                # ── Evaluate a factor ────────────────────────────────
                print("\n[C.4] Evaluating Rank($close, 20) via SSE...")
                result = await session.call_tool(
                    "evaluate_factor",
                    arguments={
                        "expression": "Rank($close, 20)",
                        "market": "csi300",
                        "fast": True,
                    },
                )
                eval_result = json.loads(result.content[0].text)
                if eval_result["success"]:
                    m = eval_result["metrics"]
                    print(f"  IC      : {m['ic']:+.4f}")
                    print(f"  Rank IC : {m['rank_ic']:+.4f}")

        print("\nDemo C complete.\n")

    except Exception as e:
        print(f"\n  Connection failed: {e}")
        print("  Make sure the SSE server is running:")
        print("    python -m ffo.mcp.server --transport sse --port 8765")
        print("\nDemo C skipped.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Demo D — Claude Agent SDK Integration
# ═══════════════════════════════════════════════════════════════════════════════

def demo_d_agent_sdk():
    """
    Build an autonomous quant research agent using Claude + FFO MCP tools.

    This demo shows how to connect the FFO MCP server to an Anthropic
    Claude agent that can autonomously discover, validate, and rank
    alpha factors.

    Requires: pip install anthropic mcp
    Set ANTHROPIC_API_KEY in your environment.
    """
    print("=" * 70)
    print("Demo D: Claude Agent SDK Integration")
    print("=" * 70)

    # ── D.1: Show the agent code (always works) ──────────────────────────────
    agent_code = '''
import asyncio
import json
import anthropic
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

async def run_quant_agent():
    """
    An autonomous quant research agent that uses FFO MCP tools
    to discover and evaluate alpha factors.
    """
    client = anthropic.Anthropic()

    # Spawn the FFO MCP server as a subprocess
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "ffo.mcp.server"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discover available tools
            tools_response = await session.list_tools()
            tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.inputSchema,
                }
                for t in tools_response.tools
            ]

            # Agent conversation loop
            messages = [
                {
                    "role": "user",
                    "content": (
                        "You are a quantitative researcher. Your task:\\n"
                        "1. Check if the FFO backend is healthy\\n"
                        "2. Evaluate these momentum factors on CSI 300:\\n"
                        "   - Rank($close, 5)\\n"
                        "   - Rank($close, 10)\\n"
                        "   - Rank($close, 20)\\n"
                        "   - $close / Delay($close, 5) - 1\\n"
                        "3. Rank them by absolute Rank IC\\n"
                        "4. Recommend the best one and explain why"
                    ),
                }
            ]

            # Agentic tool-use loop
            while True:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4096,
                    tools=tools,
                    messages=messages,
                )

                # Check if the model wants to use tools
                if response.stop_reason == "tool_use":
                    # Process each tool call
                    assistant_content = response.content
                    tool_results = []

                    for block in assistant_content:
                        if block.type == "tool_use":
                            print(f"  Agent calls: {block.name}({json.dumps(block.input)[:80]})")
                            result = await session.call_tool(block.name, arguments=block.input)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result.content[0].text,
                            })

                    messages.append({"role": "assistant", "content": assistant_content})
                    messages.append({"role": "user", "content": tool_results})
                else:
                    # Model is done — print final response
                    for block in response.content:
                        if hasattr(block, "text"):
                            print(f"\\n  Agent response:\\n{block.text}")
                    break

asyncio.run(run_quant_agent())
'''

    print("\n[D.1] Agent code (copy-paste ready):\n")
    print(agent_code)

    # ── D.2: Try to run the agent ────────────────────────────────────────────
    print("[D.2] Attempting to run the agent...")

    try:
        import anthropic
        anthropic.Anthropic()  # test API key
    except ImportError:
        print("  anthropic SDK not installed. Install with: pip install anthropic")
        print("  Skipping live execution.\n")
        return
    except anthropic.AuthenticationError:
        print("  ANTHROPIC_API_KEY not set. Set it with:")
        print("    export ANTHROPIC_API_KEY='sk-ant-...'")
        print("  Skipping live execution.\n")
        return

    try:
        from mcp import ClientSession  # noqa: F811
    except ImportError:
        print("  MCP SDK not installed. Install with: pip install mcp")
        print("  Skipping live execution.\n")
        return

    print("  Dependencies OK. Running agent...\n")

    # Execute the agent
    exec(agent_code, {"__name__": "__main__"})

    print("\nDemo D complete.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Demo E — IDE / Client Configuration Examples
# ═══════════════════════════════════════════════════════════════════════════════

def demo_e_configurations():
    """
    Print configuration snippets for popular MCP clients.
    """
    print("=" * 70)
    print("Demo E: IDE / Client Configuration Examples")
    print("=" * 70)

    # ── E.1: Claude Desktop ──────────────────────────────────────────────────
    print("""
┌─────────────────────────────────────────────────────────┐
│  E.1  Claude Desktop                                    │
│  File: ~/Library/Application Support/Claude/            │
│        claude_desktop_config.json                       │
└─────────────────────────────────────────────────────────┘

{
  "mcpServers": {
    "ffo": {
      "command": "python",
      "args": ["-m", "ffo.mcp.server"],
      "env": {
        "PYTHONPATH": "/path/to/AlphaBench"
      }
    }
  }
}

Alternative using the `ppo` CLI (if installed as a package):

{
  "mcpServers": {
    "ffo": {
      "command": "ppo",
      "args": ["start", "mcp"]
    }
  }
}
""")

    # ── E.2: Claude Code (CLI) ───────────────────────────────────────────────
    print("""
┌─────────────────────────────────────────────────────────┐
│  E.2  Claude Code (CLI)                                 │
│  File: ~/.claude/claude_code_config.json                │
└─────────────────────────────────────────────────────────┘

{
  "mcpServers": {
    "ffo": {
      "command": "python",
      "args": ["-m", "ffo.mcp.server"],
      "cwd": "/path/to/AlphaBench"
    }
  }
}
""")

    # ── E.3: Cursor ──────────────────────────────────────────────────────────
    print("""
┌─────────────────────────────────────────────────────────┐
│  E.3  Cursor IDE                                        │
│  File: .cursor/mcp.json (project root)                  │
└─────────────────────────────────────────────────────────┘

{
  "mcpServers": {
    "ffo": {
      "command": "python",
      "args": ["-m", "ffo.mcp.server"],
      "env": {
        "PYTHONPATH": "/path/to/AlphaBench"
      }
    }
  }
}
""")

    # ── E.4: VS Code (Copilot MCP) ──────────────────────────────────────────
    print("""
┌─────────────────────────────────────────────────────────┐
│  E.4  VS Code (GitHub Copilot MCP support)              │
│  File: .vscode/mcp.json (project root)                  │
└─────────────────────────────────────────────────────────┘

{
  "servers": {
    "ffo": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "ffo.mcp.server"],
      "env": {
        "PYTHONPATH": "/path/to/AlphaBench"
      }
    }
  }
}
""")

    # ── E.5: SSE transport (web agents, multi-client) ────────────────────────
    print("""
┌─────────────────────────────────────────────────────────┐
│  E.5  SSE Transport (web agents, multi-client)          │
└─────────────────────────────────────────────────────────┘

Start the server with SSE transport:

    python -m ffo.mcp.server --transport sse --port 8765
    # or
    ppo start mcp --transport sse --port 8765

Then connect from any MCP SSE client:

    URL: http://localhost:8765/sse

Python client example:

    from mcp import ClientSession
    from mcp.client.sse import sse_client

    async with sse_client("http://localhost:8765/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "evaluate_factor",
                arguments={"expression": "Rank($close, 20)"}
            )
""")

    # ── E.6: Streamable HTTP transport ───────────────────────────────────────
    print("""
┌─────────────────────────────────────────────────────────┐
│  E.6  Streamable HTTP Transport (latest MCP spec)       │
└─────────────────────────────────────────────────────────┘

Start the server with streamable-http transport:

    python -m ffo.mcp.server --transport streamable-http --port 8765

This is the newest MCP transport, recommended for production
web deployments. It replaces SSE with a more robust HTTP-based
protocol that supports bidirectional streaming.
""")

    # ── E.7: Quick reference of all MCP tools ────────────────────────────────
    print("""
┌─────────────────────────────────────────────────────────┐
│  E.7  Quick Reference: All MCP Tools                    │
└─────────────────────────────────────────────────────────┘

Tool                       Description
─────────────────────────  ──────────────────────────────────────────
evaluate_factor            Evaluate a single alpha factor expression
batch_evaluate_factors     Evaluate multiple factors in parallel
check_factor_syntax        Validate expression syntax (fast, no eval)
get_server_health          Check if FFO backend is running
get_cache_stats            Show evaluation cache statistics
clear_cache                Clear all cached evaluations

Resources:
  ffo://operators          List of all supported Qlib operators
  ffo://markets            Available market universes (csi300/500/1000)
""")

    # ── E.8: Example prompts for LLM agents ─────────────────────────────────
    print("""
┌─────────────────────────────────────────────────────────┐
│  E.8  Example Prompts for LLM Agents                    │
└─────────────────────────────────────────────────────────┘

Once the MCP server is connected, try these prompts in Claude:

  1. "Check if the FFO backend is healthy."

  2. "Evaluate the factor Rank($close, 20) on CSI 300 for 2023."

  3. "Compare these momentum factors and tell me which is strongest:
     Rank($close, 5), Rank($close, 10), Rank($close, 20)"

  4. "Build a volume-price divergence factor and evaluate it."

  5. "Find the best lookback window for a simple momentum factor.
     Test Rank($close, n) for n = 5, 10, 20, 40, 60."

  6. "Create a volatility-adjusted momentum factor and compare
     it against raw momentum on CSI 300 and CSI 500."

  7. "Check if 'Corr($close, $volume, 10)' is a valid expression,
     then evaluate it and interpret the results."
""")

    print("Demo E complete.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FFO MCP Server — Complete Demo Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demos/06_mcp_demos.py           # Run all demos
  python demos/06_mcp_demos.py --demo a  # Direct tool calls only
  python demos/06_mcp_demos.py --demo b  # stdio MCP client
  python demos/06_mcp_demos.py --demo c  # SSE MCP client
  python demos/06_mcp_demos.py --demo d  # Claude Agent SDK
  python demos/06_mcp_demos.py --demo e  # Configuration examples
        """,
    )
    parser.add_argument(
        "--demo",
        choices=["a", "b", "c", "d", "e", "all"],
        default="all",
        help="Which demo to run (default: all)",
    )
    args = parser.parse_args()

    demos = {
        "a": ("Direct Tool Calls", lambda: demo_a_direct_calls()),
        "b": ("MCP stdio Client", lambda: asyncio.run(demo_b_stdio_client())),
        "c": ("MCP SSE Client", lambda: asyncio.run(demo_c_sse_client())),
        "d": ("Claude Agent SDK", lambda: demo_d_agent_sdk()),
        "e": ("Configuration Examples", lambda: demo_e_configurations()),
    }

    if args.demo == "all":
        for key, (name, fn) in demos.items():
            try:
                fn()
            except KeyboardInterrupt:
                print(f"\n  Skipped (Ctrl-C).\n")
            except Exception as e:
                print(f"\n  Demo {key.upper()} failed: {e}\n")
    else:
        name, fn = demos[args.demo]
        fn()

    print("=" * 70)
    print("All demos complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
