"""
GenMol evaluation sub-package.

Provides metrics, distribution analysis, and visualisation
for evaluating molecular generation quality.
"""

from .metrics import (
    validity,
    uniqueness,
    novelty,
    internal_diversity,
    compute_all_metrics,
)
from .distribution import (
    compute_properties,
    compare_distributions,
    distribution_summary,
    ks_test,
)
from .visualise import (
    plot_training_curves,
    plot_property_distributions,
    plot_latent_space,
    plot_interpolation,
    plot_metrics_summary,
)

__all__ = [
    "validity",
    "uniqueness",
    "novelty",
    "internal_diversity",
    "compute_all_metrics",
    "compute_properties",
    "compare_distributions",
    "distribution_summary",
    "ks_test",
    "plot_training_curves",
    "plot_property_distributions",
    "plot_latent_space",
    "plot_interpolation",
    "plot_metrics_summary",
]
