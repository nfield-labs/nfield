"""
Patch remaining backend files to add:
  1. supports_logit_bias property (returns False) after accuracy_loss_baseline
  2. logit_bias: dict[int, float] | None = None, after stop parameter in generate()
"""

from __future__ import annotations

import re
from pathlib import Path

BACKENDS_DIR = Path(__file__).parent.parent / "src" / "formatshield" / "backends"

# Files already patched — skip them
ALREADY_PATCHED = {
    "protocol.py",
    "openai_backend.py",
    "vllm_backend.py",
    "dryrun_backend.py",
}

# Files to skip entirely
SKIP = {"__init__.py"}

# Regex: find `stop: list[str] | str | None = None,` (with any surrounding whitespace/newlines)
# Capture leading indentation so we can preserve it.
STOP_PARAM_RE = re.compile(
    r"(?P<indent>[ \t]*)stop: list\[str\] \| str \| None = None,",
)

# Regex: find the return statement that is the body of `accuracy_loss_baseline`.
# We look for the @property decorator line, then def accuracy_loss_baseline, then its return.
# Strategy: find the whole accuracy_loss_baseline property block and locate the last `return ...`
# line so we can insert right after it.
ACCURACY_LOSS_PROP_RE = re.compile(
    r"""
    # Match the accuracy_loss_baseline property block ending with a return statement
    (?P<block>
        @property\n
        (?:[ \t]+def\ accuracy_loss_baseline[^\n]*\n)  # def line
        (?:[ \t]+\"\"\"[^\n]*\"\"\"\n)?                 # optional one-line docstring
        (?:[ \t]+\"\"\"(?:.|\n)*?\"\"\"[ \t]*\n)?       # optional multi-line docstring
        (?P<return_line>[ \t]+return[^\n]*\n)            # the return statement
    )
    """,
    re.VERBOSE,
)

SUPPORTS_LOGIT_BIAS_PROP = """\
    @property
    def supports_logit_bias(self) -> bool:
        \"\"\"This backend does not support token-level logit biasing.\"\"\"
        return False
"""


def patch_file(path: Path) -> tuple[bool, list[str]]:
    """Patch a single backend file.  Returns (was_modified, list_of_changes)."""
    original = path.read_text(encoding="utf-8")
    changes: list[str] = []

    # ------------------------------------------------------------------ #
    # Guard: skip if already has supports_logit_bias                       #
    # ------------------------------------------------------------------ #
    if "supports_logit_bias" in original:
        return False, ["SKIP — supports_logit_bias already present"]

    # ------------------------------------------------------------------ #
    # Guard: only patch if has accuracy_loss_baseline                      #
    # ------------------------------------------------------------------ #
    has_accuracy_loss = "accuracy_loss_baseline" in original

    text = original

    # ------------------------------------------------------------------ #
    # 1. Insert logit_bias param after every `stop: list[str] | str | None = None,`
    # ------------------------------------------------------------------ #
    stop_param_hits = list(STOP_PARAM_RE.finditer(text))
    if stop_param_hits:
        # Replace from last to first to preserve offsets
        for m in reversed(stop_param_hits):
            indent = m.group("indent")
            replacement = (
                m.group(0) + "\n" + indent + "logit_bias: dict[int, float] | None = None,"
            )
            text = text[: m.start()] + replacement + text[m.end() :]
        changes.append(f"  + Added logit_bias param after stop in {len(stop_param_hits)} location(s)")
    else:
        changes.append("  ~ No stop param found — logit_bias param NOT added")

    # ------------------------------------------------------------------ #
    # 2. Insert supports_logit_bias property after accuracy_loss_baseline  #
    # ------------------------------------------------------------------ #
    if has_accuracy_loss:
        m = ACCURACY_LOSS_PROP_RE.search(text)
        if m:
            insert_pos = m.end()
            text = text[:insert_pos] + "\n" + SUPPORTS_LOGIT_BIAS_PROP + text[insert_pos:]
            changes.append("  + Added supports_logit_bias property after accuracy_loss_baseline")
        else:
            changes.append("  ~ accuracy_loss_baseline found but regex did not match — property NOT added")
    else:
        changes.append("  ~ No accuracy_loss_baseline property — supports_logit_bias NOT added")

    if text == original:
        return False, changes + ["  (no net change)"]

    path.write_text(text, encoding="utf-8")
    return True, changes


def main() -> None:
    py_files = sorted(BACKENDS_DIR.glob("*.py"))
    print(f"Found {len(py_files)} .py files in {BACKENDS_DIR}\n")

    modified_count = 0
    skipped_count = 0

    for path in py_files:
        name = path.name

        if name in SKIP:
            print(f"[SKIP]    {name}  (excluded)")
            skipped_count += 1
            continue

        if name in ALREADY_PATCHED:
            print(f"[DONE]    {name}  (already patched)")
            skipped_count += 1
            continue

        was_modified, changes = patch_file(path)
        status = "[PATCHED] " if was_modified else "[NO-CHANGE]"
        print(f"{status} {name}")
        for c in changes:
            print(c)
        if was_modified:
            modified_count += 1

    print(f"\nSummary: {modified_count} file(s) patched, {skipped_count} skipped.")


if __name__ == "__main__":
    main()
