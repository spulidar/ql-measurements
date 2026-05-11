"""Smoke tests for the refactored MILGRAU package imports."""

from __future__ import annotations


def test_core_package_imports() -> None:
    """The main MILGRAU subpackages should import without side effects."""
    import milgrau
    import milgrau.config
    import milgrau.io
    import milgrau.physics
    import milgrau.pipeline
    import milgrau.visualization

    assert "config" in milgrau.__all__
    assert "io" in milgrau.__all__
    assert "physics" in milgrau.__all__
    assert "pipeline" in milgrau.__all__
    assert "visualization" in milgrau.__all__


def test_pipeline_entrypoints_import() -> None:
    """Command-line entrypoint modules should import cleanly."""
    import scripts.run_libids
    import scripts.run_lipancora
    import scripts.run_liracos
    import scripts.run_lebear

    assert callable(scripts.run_libids.main)
    assert callable(scripts.run_lipancora.main)
    assert callable(scripts.run_liracos.main)
    assert callable(scripts.run_lebear.main)
