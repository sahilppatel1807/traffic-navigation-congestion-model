"""Smoke tests for project scaffolding."""


def test_src_package_imports():
    import src

    assert src is not None
