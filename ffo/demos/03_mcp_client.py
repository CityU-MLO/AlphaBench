#!/usr/bin/env python
"""
Demo 03 — Using the FFO MCP Server
=====================================
Shows two ways to interact with the FFO MCP server:

  A) Direct function calls (same functions the MCP server wraps)
  B) Via the MCP client protocol using the mcp Python library

The MCP server exposes FFO tools to any MCP-compatible agent
(Claude, GPT-4o with tools, etc.).

Start both the backend and MCP server first:

    ppo start backend
    # In a separate terminal (or use --transport sse for persistent server):
    ppo start mcp --transport sse --port 8765

Or for stdio transport (used by Claude Desktop):
    The server is spawned automatically by the MCP client.

Run this demo:

    python demos/03_mcp_client.py
"""

from __future__ import annotations

# ── Part A: Direct function calls (same logic as MCP tools) ──────────────────

def demo_direct_calls():
    """Call the same functions that the MCP tools wrap."""
    print("=" * 60)
    print("Part A: Direct function calls (MCP tool functions)")
    print("=" * 60)

    # The MCP server wraps these exact functions
    from ffo.mcp.server import (
        evaluate_factor,
        batch_evaluate_factors,
        check_factor_syntax,
        get_server_health,
        get_cache_stats,
    )

    # Health check
    print("\n[1] get_server_health()")
    health = get_server_health()
    print(f"  is_healthy : {health['is_healthy']}")
    print(f"  latency_ms : {health.get('latency_ms', 0):.0f}")

    if not health["is_healthy"]:
        print("  Backend not running. Start with: ppo start backend")
        return

    # Syntax check
    print("\n[2] check_factor_syntax('Rank($close, 20)')")
    chk = check_factor_syntax("Rank($close, 20)")
    print(f"  is_valid : {chk['is_valid']}")

    # Single evaluation
    print("\n[3] evaluate_factor('Rank($close, 20)')")
    result = evaluate_factor(
        expression="Rank($close, 20)",
        market="csi300",
        start="2023-01-01",
        end="2024-01-01",
        fast=True,
    )
    if result["success"]:
        m = result["metrics"]
        print(f"  IC      : {m['ic']:+.4f}")
        print(f"  Rank IC : {m['rank_ic']:+.4f}")
        print(f"  ICIR    : {m['icir']:+.4f}")
    else:
        print(f"  Error   : {result['error']}")

    # Batch evaluation
    print("\n[4] batch_evaluate_factors([...])")
    expressions = [
        "Rank($close, 5)",
        "Rank($close, 20)",
        "Mean($volume, 5) / Mean($volume, 20)",
    ]
    results = batch_evaluate_factors(
        expressions=expressions,
        market="csi300",
        start="2023-01-01",
        end="2024-01-01",
        fast=True,
        max_workers=4,
    )
    for r in results:
        expr = r["expression"][:40]
        if r["success"]:
            ic = r["metrics"]["ic"]
            ric = r["metrics"]["rank_ic"]
            print(f"  {expr:<42} IC={ic:+.4f}  RankIC={ric:+.4f}")
        else:
            print(f"  {expr:<42} ERROR: {r['error']}")

    # Cache stats
    print("\n[5] get_cache_stats()")
    stats = get_cache_stats()
    print(f"  cache_size     : {stats['cache_size']}")
    print(f"  max_cache_size : {stats['max_cache_size']}")


# ── Part B: MCP client protocol (programmatic MCP usage) ─────────────────────

def demo_mcp_protocol():
    """
    Show how an MCP client would discover and call tools.

    In practice this is done by Claude Desktop or the MCP Python SDK.
    This demo just shows the tool schema to understand what an agent sees.
    """
    print("\n" + "=" * 60)
    print("Part B: MCP Tool Schema (what the agent sees)")
    print("=" * 60)

    import asyncio
    from ffo.mcp.server import mcp

    print(f"\nMCP Server Name: {mcp.name}")

    tools = asyncio.run(mcp.list_tools())
    print(f"\nAvailable Tools ({len(tools)}):")
    for tool in tools:
        first_line = (tool.description or "").strip().split('\n')[0][:72]
        print(f"\n  [{tool.name}]")
        print(f"  {first_line}")

    resources = asyncio.run(mcp.list_resources())
    print(f"\nAvailable Resources ({len(resources)}):")
    for res in resources:
        print(f"  {res.uri}")

    print("\nTo use with Claude Desktop, add to ~/.claude/claude_desktop_config.json:")
    print("""  {
    "mcpServers": {
      "ffo": {
        "command": "ppo",
        "args": ["start", "mcp"]
      }
    }
  }""")

    print("\nTo use with the MCP Python SDK (SSE transport):")
    print("""  from mcp import ClientSession
  from mcp.client.sse import sse_client

  async with sse_client("http://localhost:8765/sse") as streams:
      async with ClientSession(*streams) as session:
          await session.initialize()
          result = await session.call_tool(
              "evaluate_factor",
              arguments={"expression": "Rank($close, 20)"}
          )
          print(result.content)""")


# ── Part C: Example LLM agent prompt that uses the MCP tools ─────────────────

def demo_agent_prompt():
    """Show an example of how an LLM agent would use these tools."""
    print("\n" + "=" * 60)
    print("Part C: Example Agent Interaction")
    print("=" * 60)

    prompt = """
You are a quantitative research assistant with access to the FFO factor
evaluation system. You can:

1. Evaluate alpha factors using evaluate_factor()
2. Batch evaluate many factors using batch_evaluate_factors()
3. Check factor syntax using check_factor_syntax()

Task: Find the best momentum factor from this list and explain why:
  - "Rank($close, 5)"
  - "Rank($close, 20)"
  - "$close / Delay($close, 5) - 1"

Step 1: Check server health
Step 2: Validate all expressions
Step 3: Batch evaluate all factors
Step 4: Compare results and recommend the best one

[Agent would call get_server_health(), check_factor_syntax() for each,
then batch_evaluate_factors(), then analyze results...]
"""
    print(prompt)

    # Simulate what the agent would do
    print("\n--- Simulated Agent Execution ---\n")

    from ffo.mcp.server import get_server_health, batch_evaluate_factors

    health = get_server_health()
    print(f"Step 1: Health = {health['is_healthy']}")

    if not health["is_healthy"]:
        print("Backend not running. Start with: ppo start backend")
        return

    candidates = [
        "Rank($close, 5)",
        "Rank($close, 20)",
        "$close / Delay($close, 5) - 1",
    ]

    results = batch_evaluate_factors(
        expressions=candidates,
        market="csi300",
        start="2023-01-01",
        end="2024-01-01",
        fast=True,
    )

    print("\nStep 3: Evaluation results:")
    best = None
    for r in results:
        if r["success"]:
            m = r["metrics"]
            print(f"  {r['expression']:<45} Rank_IC={m['rank_ic']:+.4f}")
            if best is None or abs(m["rank_ic"]) > abs(best["metrics"]["rank_ic"]):
                best = r

    if best:
        print(f"\nStep 4: Best factor = '{best['expression']}'")
        print(f"  Rank IC = {best['metrics']['rank_ic']:+.4f}")
        print(f"  ICIR    = {best['metrics']['icir']:+.4f}")


def main():
    demo_direct_calls()
    demo_mcp_protocol()
    demo_agent_prompt()
    print("\n" + "=" * 60)
    print("Demo complete.")


if __name__ == "__main__":
    main()
