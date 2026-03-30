"""
Shared test fixtures for neorx tests.
===============================================

The HIV disease graph is used by most tests.  Building it requires
7 live API calls (~40s), so we build it ONCE per session and share
it across every test file.

Fixtures
--------
- ``hiv_graph`` — ``DiseaseGraph`` for HIV (session-scoped)
- ``hiv_networkx`` — NetworkX ``DiGraph`` conversion (session-scoped)
"""

import pytest

from modules.neorx.graph_builder import (
    build_disease_graph,
    disease_graph_to_networkx,
)
from modules.neorx.models import DiseaseGraph


@pytest.fixture(scope="session")
def hiv_graph() -> DiseaseGraph:
    """Build the HIV disease graph ONCE for the entire test session.

    Uses ``max_genes=15`` (rather than the production default of 20)
    to keep the per-gene API fan-out fast during tests — 15 genes is
    more than enough to validate correctness.
    """
    return build_disease_graph("HIV", max_genes=15, allow_mocks=True)


@pytest.fixture(scope="session")
def hiv_networkx(hiv_graph):
    """NetworkX DiGraph for HIV (session-scoped)."""
    return disease_graph_to_networkx(hiv_graph)
