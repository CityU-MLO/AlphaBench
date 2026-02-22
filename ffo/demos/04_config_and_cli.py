#!/usr/bin/env python
"""
Demo 04 — Configuration Management & CLI Overview
==================================================
Shows how the unified configuration system works and documents
all available CLI commands.

Run this demo to see config resolution in action:

    python demos/04_config_and_cli.py
"""

from __future__ import annotations


def demo_config():
    """Demonstrate the configuration management system."""
    print("=" * 60)
    print("Demo 04A: Configuration Management")
    print("=" * 60)

    from ffo.config import get_config

    cfg = get_config()

    print(f"\nConfig loaded from: {cfg._project_config_path}")
    print(f"User overrides    : {cfg._project_config_path.parent.parent / '.ppo' / 'config.yaml'}")

    # Dot-path access
    print("\n[Dot-path access]")
    print(f"  cfg.get('server.backend.port')      = {cfg.get('server.backend.port')}")
    print(f"  cfg.get('evaluation.market')         = {cfg.get('evaluation.market')}")
    print(f"  cfg.get('evaluation.fast')           = {cfg.get('evaluation.fast')}")
    print(f"  cfg.get('cache.max_entries')         = {cfg.get('cache.max_entries')}")
    print(f"  cfg.get('qlib.data_path')            = {cfg.get('qlib.data_path')}")
    print(f"  cfg.get('nonexistent', 'fallback')   = {cfg.get('nonexistent', 'fallback')}")

    # Convenience properties
    print("\n[Convenience properties]")
    print(f"  cfg.backend_url  = {cfg.backend_url}")
    print(f"  cfg.web_url      = {cfg.web_url}")
    print(f"  cfg.cache_path   = {cfg.cache_path}")
    print(f"  cfg.pid_dir      = {cfg.pid_dir}")
    print(f"  cfg.log_dir      = {cfg.log_dir}")

    # Dict-style access
    print("\n[Dict-style access]")
    print(f"  cfg['server']['backend']['port'] = {cfg['server']['backend']['port']}")

    # Environment variable overrides
    print("\n[Environment variable overrides]")
    print("  Set FFO_BACKEND_PORT=19321 to override the backend port")
    print("  Set FFO_EVALUATION_MARKET=csi500 to change default market")
    print("  Set FFO_CACHE_MAX_ENTRIES=100000 to increase cache size")
    print("  All legacy vars also work: PORT=19777, DEFAULT_MARKET=csi300")

    # Show full config
    import yaml
    print("\n[Full effective configuration]")
    print(yaml.dump(cfg.as_dict(), default_flow_style=False, sort_keys=False))


def demo_cli_reference():
    """Print a comprehensive CLI reference."""
    print("=" * 60)
    print("Demo 04B: CLI Command Reference")
    print("=" * 60)

    reference = """
INSTALLATION
─────────────────────────────────────────────────────────────
  # Install the ppo CLI from the project root:
  pip install -e .

  # Verify installation:
  ppo --version
  ppo --help

SERVER CONTROL
─────────────────────────────────────────────────────────────
  ppo start backend              Start the FFO backend API server
  ppo start backend --port 9000  Start on a custom port
  ppo start backend --no-wait    Don't wait for health check
  ppo start web                  Start the web UI (port 19787)
  ppo start mcp                  Start MCP server (stdio transport)
  ppo start mcp --transport sse  Start MCP server (SSE transport)
  ppo start mcp --port 9000      Start MCP with custom SSE port
  ppo start all                  Start backend + web UI

  ppo stop backend               Stop the backend
  ppo stop web                   Stop the web UI
  ppo stop mcp                   Stop the MCP server
  ppo stop all                   Stop all services

  ppo restart backend            Restart the backend
  ppo restart all                Restart everything

  ppo status                     Show status of all services (with health)

SERVICE LOGS
─────────────────────────────────────────────────────────────
  ppo logs backend               Show last 50 lines of backend log
  ppo logs backend -n 100        Show last 100 lines
  ppo logs backend -f            Follow (tail -f) backend log
  ppo logs web -f                Follow web UI log

FACTOR OPERATIONS (quick CLI tests)
─────────────────────────────────────────────────────────────
  ppo eval "Rank($close, 20)"                Evaluate a factor (fast mode)
  ppo eval "Rank($close, 20)" --no-fast      Full mode with portfolio backtest
  ppo eval "Mean($volume, 5)" --market csi500
  ppo eval "Rank($close, 20)" --json         JSON output

  ppo check "Rank($close, 20)"               Validate factor syntax
  ppo check "BadOp($close)"                  Returns error for invalid syntax

CACHE MANAGEMENT
─────────────────────────────────────────────────────────────
  ppo cache stats                Show cache hit count, size, capacity
  ppo cache clear                Clear all cached evaluations (with prompt)

CONFIGURATION
─────────────────────────────────────────────────────────────
  ppo config show                Print effective configuration (YAML)
  ppo config show --section evaluation    Show only evaluation section
  ppo config set evaluation.market csi500  Set default market
  ppo config set server.backend.port 19321
  ppo config set evaluation.fast false
  ppo config init                Create default ~/.ppo/config.yaml
  ppo config init --force        Overwrite existing user config

CONFIGURATION PRIORITY (highest → lowest)
─────────────────────────────────────────────────────────────
  1. Environment variables    FFO_BACKEND_PORT=19321
  2. User config              ~/.ppo/config.yaml
  3. Project config           ./config/ffo.yaml
  4. Built-in defaults

COMMON ENVIRONMENT VARIABLES
─────────────────────────────────────────────────────────────
  FFO_BACKEND_PORT=19777         Backend API port
  FFO_BACKEND_WORKERS=4          Gunicorn workers
  FFO_EVALUATION_MARKET=csi300   Default market
  FFO_EVALUATION_FAST=true       Default fast mode
  FFO_CACHE_PATH=~/.ppo/cache.db Cache SQLite path
  FFO_CACHE_MAX_ENTRIES=50000    Cache capacity
  FFO_QLIB_DATA_PATH=~/.qlib/... Qlib data directory

  # Legacy variables (still supported)
  PORT=19777
  DEFAULT_MARKET=csi300
  TIMEOUT_EVAL_SEC=180

MCP SERVER INTEGRATION
─────────────────────────────────────────────────────────────
  # Claude Desktop — add to ~/.claude/claude_desktop_config.json:
  {
    "mcpServers": {
      "ffo": {
        "command": "ppo",
        "args": ["start", "mcp"]
      }
    }
  }

  # SSE transport (for persistent server):
  ppo start mcp --transport sse --port 8765
  # Connect via: http://localhost:8765/sse

  # Available MCP tools:
  evaluate_factor         Evaluate a single factor
  batch_evaluate_factors  Evaluate many factors in parallel
  check_factor_syntax     Validate expression syntax
  get_server_health       Check backend status
  get_cache_stats         Show cache statistics
  clear_cache             Clear the cache

  # Available MCP resources:
  ffo://operators         List all Qlib operators
  ffo://markets           List supported market universes

PYTHON API (for scripts & agents)
─────────────────────────────────────────────────────────────
  from ffo.api import (
      evaluate_factor,        # Evaluate one factor
      batch_evaluate_factors, # Evaluate many factors
      check_factor,           # Validate syntax
      get_cache_stats,        # Cache stats
      clear_cache,            # Clear cache
      server_health,          # Health check
  )

  result = evaluate_factor("Rank($close, 20)")
  print(result.metrics.ic)          # Information Coefficient
  print(result.metrics.rank_ic)     # Rank IC
  print(result.success)             # True/False

  results = batch_evaluate_factors(
      ["Rank($close, 5)", "Rank($close, 20)"],
      parallel=True,
      fast=True,
  )

TYPICAL WORKFLOW
─────────────────────────────────────────────────────────────
  1. Install:    pip install -e .
  2. Configure:  ppo config init
                 ppo config set qlib.data_path /path/to/qlib/data
  3. Start:      ppo start backend
  4. Verify:     ppo status
  5. Evaluate:   ppo eval "Rank($close, 20)"
  6. Use API:    python demos/01_basic_usage.py
  7. Use MCP:    ppo start mcp --transport sse
  8. Stop:       ppo stop all
"""
    print(reference)


def main():
    demo_config()
    demo_cli_reference()


if __name__ == "__main__":
    main()
