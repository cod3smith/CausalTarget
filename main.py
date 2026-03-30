"""
NeoRx — Causal Drug Target Discovery
=============================================

Entry point.  For the CLI run ``neorx --help``.
For the API server run ``neorx serve``.

Programmatic usage::

    from neorx import (
        build_disease_graph,
        identify_causal_targets,
        run_pipeline,
    )

    result = run_pipeline("HIV", top_n_targets=5)
"""


def main() -> None:
    """Delegate to the NeoRx CLI."""
    from modules.neorx.__main__ import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
