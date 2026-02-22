#!/usr/bin/env python
"""
PPO CLI - Pluggable Portfolio Optimizer command-line interface.

Commands
--------
  ppo start [backend|web|mcp|all]   Start service(s)
  ppo stop  [backend|web|mcp|all]   Stop service(s)
  ppo restart [backend|web|mcp|all] Restart service(s)
  ppo status                        Show running services
  ppo logs [backend|web]            Tail service logs

  ppo eval  EXPRESSION              Evaluate a factor (quick test)
  ppo check EXPRESSION              Validate factor syntax

  ppo cache stats                   Show cache statistics
  ppo cache clear                   Clear the evaluation cache

  ppo config show                   Print current configuration
  ppo config set KEY VALUE          Set a config key (dot-path)
  ppo config init                   Write default ~/.ppo/config.yaml
"""

from __future__ import annotations

import os
import pathlib
import signal
import subprocess
import sys
import time
from typing import Optional

import click
import requests
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_cfg():
    from ffo.config import get_config
    return get_config()


def _ffo_root() -> pathlib.Path:
    """Absolute path to the ffo/ package directory."""
    return pathlib.Path(__file__).parent.parent.resolve()


def _project_root() -> pathlib.Path:
    """Project root (parent of ffo/)."""
    return _ffo_root().parent


def _pid_file(service: str) -> pathlib.Path:
    pid_dir = _get_cfg().pid_dir
    pid_dir.mkdir(parents=True, exist_ok=True)
    return pid_dir / f"{service}.pid"


def _log_file(service: str) -> pathlib.Path:
    log_dir = _get_cfg().log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{service}.log"


def _write_pid(service: str, pid: int) -> None:
    _pid_file(service).write_text(str(pid))


def _read_pid(service: str) -> Optional[int]:
    pf = _pid_file(service)
    if not pf.exists():
        return None
    try:
        return int(pf.read_text().strip())
    except (ValueError, OSError):
        return None


def _remove_pid(service: str) -> None:
    pf = _pid_file(service)
    if pf.exists():
        pf.unlink(missing_ok=True)


def _is_running(service: str) -> bool:
    """Return True if the process for *service* is alive."""
    import psutil
    pid = _read_pid(service)
    if pid is None:
        return False
    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        _remove_pid(service)
        return False


def _kill_service(service: str) -> bool:
    """Send SIGTERM to a service, return True if stopped."""
    import psutil
    pid = _read_pid(service)
    if pid is None:
        return False
    try:
        proc = psutil.Process(pid)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except psutil.TimeoutExpired:
            proc.kill()
        _remove_pid(service)
        return True
    except psutil.NoSuchProcess:
        _remove_pid(service)
        return False


def _health_check(url: str, retries: int = 15, delay: float = 1.0) -> bool:
    """Poll *url*/health until healthy or give up."""
    for _ in range(retries):
        try:
            r = requests.get(f"{url}/health", timeout=3)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(delay)
    return False


# ── Root group ────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(package_name="ppo")
def cli():
    """PPO - Pluggable Portfolio Optimizer CLI.

    Controls the FFO factor evaluation backend, web UI, and MCP server.
    """


# ── start ─────────────────────────────────────────────────────────────────────

@cli.group()
def start():
    """Start service(s)."""


@start.command("backend")
@click.option("--port", default=None, type=int, help="Override backend port")
@click.option("--workers", default=None, type=int, help="Gunicorn worker count")
@click.option("--no-wait", is_flag=True, help="Don't wait for health check")
def start_backend(port, workers, no_wait):
    """Start the FFO backend API server (gunicorn)."""
    cfg = _get_cfg()
    if _is_running("backend"):
        console.print("[yellow]Backend is already running.[/yellow]")
        return

    _port = port or cfg.get("server.backend.port", 19777)
    _workers = workers or cfg.get("server.backend.workers", 2)
    _threads = cfg.get("server.backend.threads", 4)
    _timeout = cfg.get("server.backend.timeout", 900)
    ffo_dir = _ffo_root()
    log_path = _log_file("backend")

    cmd = [
        "gunicorn", "backend_app:app",
        "--bind", f"0.0.0.0:{_port}",
        "--workers", str(_workers),
        "--threads", str(_threads),
        "--timeout", str(_timeout),
        "--chdir", str(ffo_dir),
        "--access-logfile", str(log_path),
        "--error-logfile", str(log_path),
    ]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    _write_pid("backend", proc.pid)
    console.print(f"[green]Backend starting[/green] (PID {proc.pid}, port {_port})")
    console.print(f"  Logs → {log_path}")

    if not no_wait:
        backend_url = f"http://127.0.0.1:{_port}"
        console.print("  Waiting for health check...", end=" ")
        if _health_check(backend_url):
            console.print("[green]OK[/green]")
            console.print(f"  API → {backend_url}")
        else:
            console.print("[yellow]timeout (server may still be starting)[/yellow]")


@start.command("web")
@click.option("--port", default=None, type=int, help="Override web UI port")
@click.option("--no-wait", is_flag=True)
def start_web(port, no_wait):
    """Start the FFO web UI."""
    cfg = _get_cfg()
    if _is_running("web"):
        console.print("[yellow]Web UI is already running.[/yellow]")
        return

    _port = port or cfg.get("server.web.port", 19787)
    ffo_dir = _ffo_root()
    log_path = _log_file("web")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "PYTHONPATH": str(_project_root())}
    proc = subprocess.Popen(
        [sys.executable, str(ffo_dir / "web_app.py")],
        cwd=str(ffo_dir),
        env=env,
        stdout=open(log_path, "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _write_pid("web", proc.pid)
    console.print(f"[green]Web UI starting[/green] (PID {proc.pid}, port {_port})")
    console.print(f"  Logs → {log_path}")

    if not no_wait:
        time.sleep(2)
        web_url = f"http://127.0.0.1:{_port}"
        console.print(f"  UI  → {web_url}")


@start.command("mcp")
@click.option("--transport", default=None, type=click.Choice(["stdio", "sse"]))
@click.option("--port", default=None, type=int, help="SSE port (only for --transport sse)")
def start_mcp(transport, port):
    """Start the MCP (Model Context Protocol) server."""
    cfg = _get_cfg()
    _transport = transport or cfg.get("server.mcp.transport", "stdio")
    _port = port or cfg.get("server.mcp.port", 8765)
    ffo_dir = _ffo_root()
    log_path = _log_file("mcp")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if _is_running("mcp"):
        console.print("[yellow]MCP server is already running.[/yellow]")
        return

    env = {**os.environ, "PYTHONPATH": str(_project_root())}
    cmd = [sys.executable, "-m", "ffo.mcp.server", "--transport", _transport]
    if _transport == "sse":
        cmd += ["--port", str(_port)]

    proc = subprocess.Popen(
        cmd,
        cwd=str(_project_root()),
        env=env,
        stdout=open(log_path, "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _write_pid("mcp", proc.pid)
    console.print(f"[green]MCP server starting[/green] (PID {proc.pid}, transport={_transport})")
    console.print(f"  Logs → {log_path}")
    if _transport == "sse":
        console.print(f"  SSE → http://127.0.0.1:{_port}/sse")


@start.command("all")
@click.pass_context
def start_all(ctx):
    """Start backend + web UI."""
    ctx.invoke(start_backend)
    ctx.invoke(start_web)


# ── stop ──────────────────────────────────────────────────────────────────────

@cli.group()
def stop():
    """Stop service(s)."""


def _stop_service(name: str) -> None:
    if _kill_service(name):
        console.print(f"[green]Stopped[/green] {name}")
    else:
        console.print(f"[dim]{name} was not running[/dim]")


@stop.command("backend")
def stop_backend():
    """Stop the backend API server."""
    _stop_service("backend")


@stop.command("web")
def stop_web():
    """Stop the web UI."""
    _stop_service("web")


@stop.command("mcp")
def stop_mcp():
    """Stop the MCP server."""
    _stop_service("mcp")


@stop.command("all")
def stop_all():
    """Stop all services."""
    for svc in ("backend", "web", "mcp"):
        _stop_service(svc)


# ── restart ───────────────────────────────────────────────────────────────────

@cli.group()
def restart():
    """Restart service(s)."""


@restart.command("backend")
@click.pass_context
def restart_backend(ctx):
    """Restart the backend API server."""
    ctx.invoke(stop_backend)
    time.sleep(1)
    ctx.invoke(start_backend)


@restart.command("web")
@click.pass_context
def restart_web(ctx):
    """Restart the web UI."""
    ctx.invoke(stop_web)
    time.sleep(1)
    ctx.invoke(start_web)


@restart.command("all")
@click.pass_context
def restart_all(ctx):
    """Restart all services."""
    ctx.invoke(stop_all)
    time.sleep(1)
    ctx.invoke(start_all)


# ── status ────────────────────────────────────────────────────────────────────

@cli.command()
def status():
    """Show the status of all services."""
    import psutil

    cfg = _get_cfg()
    services = [
        ("backend", cfg.backend_url),
        ("web",     cfg.web_url),
        ("mcp",     None),
    ]

    table = Table(title="PPO Service Status", show_lines=True)
    table.add_column("Service",  style="bold")
    table.add_column("Status",   style="bold")
    table.add_column("PID",      justify="right")
    table.add_column("URL / Info")

    for name, url in services:
        pid = _read_pid(name)
        running = _is_running(name)

        if running:
            status_text = "[green]running[/green]"
            pid_text = str(pid)
            # Try health check
            if url:
                try:
                    r = requests.get(f"{url}/health", timeout=2)
                    info = r.json().get("status", url) if r.status_code == 200 else url
                except Exception:
                    info = f"{url} [dim](no response)[/dim]"
            else:
                info = f"transport={cfg.get('server.mcp.transport', 'stdio')}"
        else:
            status_text = "[red]stopped[/red]"
            pid_text = "-"
            info = ""

        table.add_row(name, status_text, pid_text, info)

    console.print(table)

    # Also print config summary
    console.print(
        f"\n[dim]Config: {cfg._project_config_path}[/dim]\n"
        f"[dim]Logs:   {cfg.log_dir}[/dim]\n"
        f"[dim]PIDs:   {cfg.pid_dir}[/dim]"
    )


# ── logs ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("service", type=click.Choice(["backend", "web", "mcp"]))
@click.option("-n", "--lines", default=50, help="Number of lines to show")
@click.option("-f", "--follow", is_flag=True, help="Follow log output")
def logs(service, lines, follow):
    """Tail service logs."""
    log_path = _log_file(service)
    if not log_path.exists():
        console.print(f"[yellow]No log file found for {service}[/yellow]")
        console.print(f"  Expected: {log_path}")
        return

    cmd = ["tail", f"-n{lines}"] + (["-f"] if follow else []) + [str(log_path)]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass


# ── eval ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("expression")
@click.option("--market",  default=None, help="Market universe (e.g. csi300)")
@click.option("--start",   default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end",     default=None, help="End date (YYYY-MM-DD)")
@click.option("--fast",    is_flag=True, default=True, help="Fast mode (IC only)")
@click.option("--no-fast", is_flag=True, help="Full mode (IC + portfolio backtest)")
@click.option("--json",    "as_json", is_flag=True, help="Print raw JSON output")
def eval(expression, market, start, end, fast, no_fast, as_json):
    """Evaluate a factor expression and print metrics.

    \b
    Example:
        ppo eval "Rank($close, 20)"
        ppo eval "Mean($volume, 5) / Std($volume, 20)" --market csi500
    """
    import json as _json
    from ffo.api import evaluate_factor

    cfg = _get_cfg()
    _market = market or cfg.get("evaluation.market", "csi300")
    _start  = start  or cfg.get("evaluation.start",  "2023-01-01")
    _end    = end    or cfg.get("evaluation.end",    "2024-01-01")
    _fast   = False if no_fast else fast

    with console.status(f"Evaluating [bold]{expression}[/bold]..."):
        result = evaluate_factor(
            expression=expression,
            market=_market,
            start=_start,
            end=_end,
            fast=_fast,
        )

    if as_json:
        console.print_json(_json.dumps(result.to_dict()))
        return

    if result.success:
        m = result.metrics
        table = Table(title=f"Factor: {expression}", show_lines=True)
        table.add_column("Metric", style="bold")
        table.add_column("Value",  justify="right")
        table.add_row("IC",        f"{m.ic:.4f}")
        table.add_row("Rank IC",   f"{m.rank_ic:.4f}")
        table.add_row("ICIR",      f"{m.icir:.4f}")
        table.add_row("Rank ICIR", f"{m.rank_icir:.4f}")
        table.add_row("Turnover",  f"{m.turnover:.4f}")
        table.add_row("N Dates",   str(m.n_dates))
        if result.cached:
            table.add_row("Cache", "[dim]hit[/dim]")
        console.print(table)
    else:
        console.print(f"[red]Evaluation failed:[/red] {result.error}")
        sys.exit(1)


# ── check ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("expression")
@click.option("--market", default=None)
def check(expression, market):
    """Validate factor expression syntax.

    \b
    Example:
        ppo check "Rank($close, 20)"
        ppo check "InvalidOp($close)"
    """
    from ffo.api import check_factor

    cfg = _get_cfg()
    _market = market or cfg.get("evaluation.market", "csi300")

    with console.status("Checking syntax..."):
        result = check_factor(expression=expression, market=_market)

    if result.is_valid:
        console.print(f"[green]Valid[/green] — {expression}")
    else:
        console.print(f"[red]Invalid[/red] — {result.error}")
        sys.exit(1)


# ── cache ─────────────────────────────────────────────────────────────────────

@cli.group()
def cache():
    """Manage the evaluation cache."""


@cache.command("stats")
def cache_stats():
    """Show cache statistics."""
    from ffo.api import get_cache_stats

    with console.status("Fetching cache stats..."):
        stats = get_cache_stats()

    table = Table(title="Cache Statistics", show_lines=True)
    table.add_column("Field",  style="bold")
    table.add_column("Value",  justify="right")
    for k, v in stats.items():
        table.add_row(str(k), str(v))
    console.print(table)


@cache.command("clear")
@click.confirmation_option(prompt="Clear all cached evaluations?")
def cache_clear():
    """Clear all cached factor evaluations."""
    from ffo.api import clear_cache

    ok = clear_cache()
    if ok:
        console.print("[green]Cache cleared.[/green]")
    else:
        console.print("[red]Failed to clear cache.[/red]")
        sys.exit(1)


# ── config ────────────────────────────────────────────────────────────────────

@cli.group()
def config():
    """Manage PPO configuration."""


@config.command("show")
@click.option("--section", default=None, help="Show only this top-level section")
def config_show(section):
    """Print current effective configuration."""
    import yaml

    cfg = _get_cfg()
    data = cfg.as_dict()
    if section:
        data = data.get(section, {})

    console.print(f"[dim]# Effective config (project: {cfg._project_config_path})[/dim]")
    console.print(yaml.dump(data, default_flow_style=False, sort_keys=False))


@config.command("set")
@click.argument("key")
@click.argument("value")
@click.option("--save", is_flag=True, default=True, help="Persist to ~/.ppo/config.yaml")
def config_set(key, value, save):
    """Set a configuration value (dot-path syntax).

    \b
    Examples:
        ppo config set evaluation.market csi500
        ppo config set server.backend.port 19321
    """
    cfg = _get_cfg()

    # Attempt type inference
    if value.lower() in ("true", "yes"):
        value = True
    elif value.lower() in ("false", "no"):
        value = False
    else:
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass

    cfg.set(key, value)
    console.print(f"Set [bold]{key}[/bold] = {value!r}")

    if save:
        path = cfg.save_user_config()
        console.print(f"[dim]Saved → {path}[/dim]")


@config.command("init")
@click.option("--force", is_flag=True, help="Overwrite existing user config")
def config_init(force):
    """Write default configuration to ~/.ppo/config.yaml."""
    import shutil

    user_cfg = pathlib.Path.home() / ".ppo" / "config.yaml"
    if user_cfg.exists() and not force:
        console.print(f"[yellow]Config already exists:[/yellow] {user_cfg}")
        console.print("Use --force to overwrite.")
        return

    # Copy from bundled package config (ffo/config/ffo.yaml)
    src = _ffo_root() / "config" / "ffo.yaml"
    user_cfg.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy(src, user_cfg)
    else:
        # Fall back: save defaults
        cfg = _get_cfg()
        cfg.save_user_config()

    console.print(f"[green]Config initialized:[/green] {user_cfg}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
