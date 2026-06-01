"""Single source of version truth.

hatch-vcs writes the real version at build time from git tags.
At development time, falls back to "0.0.0+unknown".
"""

from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("formatshield")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"
except ImportError:
    __version__ = "0.0.0+unknown"
