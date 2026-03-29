"""
DockBot Tests — Binding Site Detection
========================================

Tests binding-site model creation and the manual method.
Co-crystal ligand and fpocket tests require PDB files / external tools,
so they are marked with ``pytest.mark.slow``.
"""

from __future__ import annotations

import pytest

from modules.dockbot.binding_site import from_manual, detect_binding_site
from modules.dockbot.models import BindingSite, BindingSiteMethod


class TestManualBindingSite:
    """Test manual binding-site creation."""

    def test_creates_site(self):
        site = from_manual(10.0, 20.0, 30.0)
        assert isinstance(site, BindingSite)
        assert site.center_x == 10.0
        assert site.center_y == 20.0
        assert site.center_z == 30.0
        assert site.method == BindingSiteMethod.MANUAL

    def test_custom_size(self):
        site = from_manual(0.0, 0.0, 0.0, size_x=30.0, size_y=30.0, size_z=30.0)
        assert site.size_x == 30.0
        assert site.size_y == 30.0
        assert site.size_z == 30.0

    def test_default_size(self):
        site = from_manual(0.0, 0.0, 0.0)
        assert site.size_x == 25.0  # default from from_manual


class TestDetectBindingSite:
    """Test the convenience dispatcher."""

    def test_manual_method(self):
        site = detect_binding_site(
            "dummy.pdb",
            method="manual",
            center=(5.0, 10.0, 15.0),
            size=(20.0, 20.0, 20.0),
        )
        assert site.center_x == 5.0
        assert site.method == BindingSiteMethod.MANUAL

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown binding-site method"):
            detect_binding_site("dummy.pdb", method="invalid_method")

    def test_manual_without_center_raises(self):
        with pytest.raises(ValueError, match="Manual method requires"):
            detect_binding_site("dummy.pdb", method="manual")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
