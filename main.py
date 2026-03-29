"""
CausalTarget — Causal Drug Target Discovery
=============================================

Entry point.  For the CLI run ``causaltarget --help``.
For the API server run ``causaltarget serve``.

Programmatic usage::

    from causaltarget import (
        build_disease_graph,
        identify_causal_targets,
        run_pipeline,
    )

    result = run_pipeline("HIV", top_n_targets=5)
"""


def main() -> None:
    """Delegate to the CausalTarget CLI."""
    from modules.causal_target.__main__ import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
