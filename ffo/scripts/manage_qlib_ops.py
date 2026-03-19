#!/usr/bin/env python
"""
Install / Uninstall FFO extended operators into qlib's source code.

Usage:
    python manage_qlib_ops.py install     # inject ops into qlib/data/ops.py
    python manage_qlib_ops.py uninstall   # remove injected ops
    python manage_qlib_ops.py status      # check current state

This hard-patches qlib's ops.py so the operators are available everywhere
(subprocesses, after qlib.init(), etc.) without needing dynamic registration.
"""

import argparse
import re
import shutil
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Markers used to fence the injected block
# ---------------------------------------------------------------------------
BEGIN_MARKER = "# --- FFO EXTENDED OPS BEGIN ---"
END_MARKER = "# --- FFO EXTENDED OPS END ---"
OPSLIST_MARKER = "# FFO_EXT"

# ---------------------------------------------------------------------------
# Operator definitions to inject (placed BEFORE the OpsList definition)
# ---------------------------------------------------------------------------
INJECTED_CODE = textwrap.dedent("""\
    # --- FFO EXTENDED OPS BEGIN ---
    # Auto-injected by ffo/scripts/manage_qlib_ops.py — do not edit manually.

    class Sqrt(NpElemOperator):
        \"\"\"Element-wise square root.\"\"\"
        def __init__(self, feature):
            super(Sqrt, self).__init__(feature, "sqrt")


    class Exp(NpElemOperator):
        \"\"\"Element-wise exponential.\"\"\"
        def __init__(self, feature):
            super(Exp, self).__init__(feature, "exp")


    class Square(NpElemOperator):
        \"\"\"Element-wise square.\"\"\"
        def __init__(self, feature):
            super(Square, self).__init__(feature, "square")


    class Sin(NpElemOperator):
        \"\"\"Element-wise sine.\"\"\"
        def __init__(self, feature):
            super(Sin, self).__init__(feature, "sin")


    class Cos(NpElemOperator):
        \"\"\"Element-wise cosine.\"\"\"
        def __init__(self, feature):
            super(Cos, self).__init__(feature, "cos")


    class Tan(NpElemOperator):
        \"\"\"Element-wise tangent.\"\"\"
        def __init__(self, feature):
            super(Tan, self).__init__(feature, "tan")


    class Tanh(NpElemOperator):
        \"\"\"Element-wise hyperbolic tangent.\"\"\"
        def __init__(self, feature):
            super(Tanh, self).__init__(feature, "tanh")


    class Arcsin(NpElemOperator):
        \"\"\"Element-wise inverse sine.\"\"\"
        def __init__(self, feature):
            super(Arcsin, self).__init__(feature, "arcsin")


    class Arccos(NpElemOperator):
        \"\"\"Element-wise inverse cosine.\"\"\"
        def __init__(self, feature):
            super(Arccos, self).__init__(feature, "arccos")


    class Arctan(NpElemOperator):
        \"\"\"Element-wise inverse tangent.\"\"\"
        def __init__(self, feature):
            super(Arctan, self).__init__(feature, "arctan")


    class Reciprocal(NpElemOperator):
        \"\"\"Element-wise reciprocal (1/x).\"\"\"
        def __init__(self, feature):
            super(Reciprocal, self).__init__(feature, "reciprocal")


    class Clip(ElemOperator):
        \"\"\"Element-wise clipping to [a_min, a_max].\"\"\"

        def __init__(self, feature, a_min=None, a_max=None):
            if a_min is None and a_max is None:
                raise ValueError("At least one of a_min or a_max must be provided.")
            self.feature = feature
            self.a_min = a_min
            self.a_max = a_max

        def __str__(self):
            return "{}({}, a_min={}, a_max={})".format(
                type(self).__name__, self.feature, self.a_min, self.a_max
            )

        def _load_internal(self, instrument, start_index, end_index, *args):
            import numpy as _np
            series = self.feature.load(instrument, start_index, end_index, *args)
            series = series.astype(_np.float32)
            if self.a_min is None:
                return _np.minimum(series, self.a_max)
            if self.a_max is None:
                return _np.maximum(series, self.a_min)
            return _np.clip(series, self.a_min, self.a_max)


    class CSRank(NpElemOperator):
        \"\"\"Cross-sectional rank (percentile) — wraps scipy percentileofscore.\"\"\"

        def __init__(self, feature):
            super(CSRank, self).__init__(feature, "csrank")

    # --- FFO EXTENDED OPS END ---
""")

# Operator names that get appended to OpsList
EXT_OPS_NAMES = [
    "Sqrt", "Exp", "Square", "Sin", "Cos", "Tan", "Tanh",
    "Arcsin", "Arccos", "Arctan", "Reciprocal", "Clip", "CSRank",
]


def _find_ops_py() -> Path:
    """Locate qlib/data/ops.py from the current Python environment."""
    try:
        import qlib.data.ops as _m
        return Path(_m.__file__)
    except ImportError:
        print("ERROR: qlib is not installed in the current Python environment.")
        sys.exit(1)


def _is_installed(text: str) -> bool:
    return BEGIN_MARKER in text


def _backup(path: Path) -> Path:
    bak = path.with_suffix(".py.ffo_backup")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"  Backup saved: {bak}")
    return bak


def _inject_classes(text: str) -> str:
    """Insert operator class definitions before the TOpsList line."""
    # Insert just before 'TOpsList = '
    anchor = "TOpsList = [TResample]"
    if anchor not in text:
        # Try to find it with different whitespace
        anchor_re = re.search(r"^TOpsList\s*=\s*\[TResample\]", text, re.MULTILINE)
        if not anchor_re:
            print("ERROR: Cannot find 'TOpsList = [TResample]' anchor in ops.py")
            sys.exit(1)
        anchor = anchor_re.group(0)

    return text.replace(anchor, INJECTED_CODE + "\n" + anchor)


def _patch_opslist(text: str) -> str:
    """Append extended op names to the OpsList definition."""
    # Match the closing of OpsList: '] + [TResample]'
    pattern = r"(\]\s*\+\s*\[TResample\])"
    suffix = " + [" + ", ".join(EXT_OPS_NAMES) + "]  " + OPSLIST_MARKER

    match = re.search(pattern, text)
    if not match:
        print("ERROR: Cannot find OpsList closing pattern '] + [TResample]'")
        sys.exit(1)

    # Only patch if not already patched
    if OPSLIST_MARKER in text:
        return text

    return text[:match.end()] + suffix + text[match.end():]


def _remove_classes(text: str) -> str:
    """Remove the injected class block."""
    pattern = re.compile(
        r"\n?" + re.escape(BEGIN_MARKER) + r".*?" + re.escape(END_MARKER) + r"\n?",
        re.DOTALL,
    )
    return pattern.sub("\n", text)


def _unpatch_opslist(text: str) -> str:
    """Remove the extended ops from OpsList."""
    # Remove everything from ' + [Sqrt, ..., CSRank]  # FFO_EXT'
    pattern = re.compile(r"\s*\+\s*\[" + r",\s*".join(EXT_OPS_NAMES) + r"\]\s*" + re.escape(OPSLIST_MARKER))
    return pattern.sub("", text)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_install(ops_py: Path, dry_run: bool = False):
    text = ops_py.read_text()

    if _is_installed(text):
        print(f"Already installed in {ops_py}")
        return

    _backup(ops_py)

    text = _inject_classes(text)
    text = _patch_opslist(text)

    if dry_run:
        print("--- DRY RUN (would write): ---")
        # Show just the changed regions
        for name in EXT_OPS_NAMES:
            if name in text:
                print(f"  + class {name}")
        print(f"  + OpsList extended with {len(EXT_OPS_NAMES)} ops")
        return

    ops_py.write_text(text)
    print(f"Installed {len(EXT_OPS_NAMES)} extended operators into {ops_py}")
    print(f"  Operators: {', '.join(EXT_OPS_NAMES)}")


def cmd_uninstall(ops_py: Path, dry_run: bool = False):
    text = ops_py.read_text()

    if not _is_installed(text):
        print(f"Not installed in {ops_py}")
        return

    text = _remove_classes(text)
    text = _unpatch_opslist(text)

    if dry_run:
        print("--- DRY RUN (would remove): ---")
        for name in EXT_OPS_NAMES:
            print(f"  - class {name}")
        return

    ops_py.write_text(text)
    print(f"Uninstalled extended operators from {ops_py}")

    # Check if backup exists
    bak = ops_py.with_suffix(".py.ffo_backup")
    if bak.exists():
        print(f"  Original backup at: {bak}")


def cmd_status(ops_py: Path):
    text = ops_py.read_text()
    print(f"qlib ops.py: {ops_py}")

    if _is_installed(text):
        found = [n for n in EXT_OPS_NAMES if f"class {n}(" in text]
        print(f"  Status: INSTALLED ({len(found)}/{len(EXT_OPS_NAMES)} operators)")
        for n in EXT_OPS_NAMES:
            marker = "+" if f"class {n}(" in text else "-"
            print(f"    [{marker}] {n}")
        in_list = OPSLIST_MARKER in text
        print(f"  OpsList patched: {'yes' if in_list else 'NO (need re-install)'}")
    else:
        print("  Status: NOT INSTALLED")

    bak = ops_py.with_suffix(".py.ffo_backup")
    if bak.exists():
        print(f"  Backup: {bak}")


def main():
    parser = argparse.ArgumentParser(
        description="Install/uninstall FFO extended operators into qlib's ops.py"
    )
    parser.add_argument(
        "command",
        choices=["install", "uninstall", "status"],
        help="Action to perform",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing",
    )
    parser.add_argument(
        "--ops-py",
        type=Path,
        default=None,
        help="Path to qlib/data/ops.py (auto-detected if omitted)",
    )
    args = parser.parse_args()

    ops_py = args.ops_py or _find_ops_py()

    if not ops_py.exists():
        print(f"ERROR: {ops_py} does not exist")
        sys.exit(1)

    if args.command == "install":
        cmd_install(ops_py, dry_run=args.dry_run)
    elif args.command == "uninstall":
        cmd_uninstall(ops_py, dry_run=args.dry_run)
    elif args.command == "status":
        cmd_status(ops_py)


if __name__ == "__main__":
    main()
