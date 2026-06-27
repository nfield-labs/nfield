"""Guard the public API surface against silent drift.

The root package exposes its names lazily (PEP 562 ``__getattr__`` over a
``_dynamic_imports`` map). These tests fail if ``__all__``, the lazy map, and the
names that actually resolve ever disagree - so a rename or a forgotten export is
caught here instead of in a downstream user's import.
"""

from __future__ import annotations

import importlib

import pytest

import nfield

# Every public name except the eagerly bound version is resolved lazily.
_LAZY_NAMES = frozenset(nfield.__all__) - {"__version__"}


class TestPublicSurface:
    def test_all_is_sorted_and_unique(self) -> None:
        assert nfield.__all__ == sorted(nfield.__all__)
        assert len(nfield.__all__) == len(set(nfield.__all__))

    def test_all_matches_lazy_import_map(self) -> None:
        # The lazy map and the public surface must list exactly the same names,
        # so neither can grow or shrink without the other.
        assert set(nfield._dynamic_imports) == _LAZY_NAMES

    def test_version_is_exported_and_a_string(self) -> None:
        assert "__version__" in nfield.__all__
        assert isinstance(nfield.__version__, str)
        assert nfield.__version__


class TestEveryExportResolves:
    def test_each_public_name_imports(self) -> None:
        # getattr drives the lazy loader; a moved or deleted symbol raises here.
        for name in _LAZY_NAMES:
            assert getattr(nfield, name) is not None

    def test_lazy_target_modules_are_importable(self) -> None:
        for name, module_path in nfield._dynamic_imports.items():
            module = importlib.import_module(module_path, package="nfield")
            assert hasattr(module, name)

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError):
            _ = nfield.does_not_exist
