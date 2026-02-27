"""Testing imports / deployment / structure of the package."""


def test_imports():
    """Test some namespaces after importing."""
    import molzen as mz

    assert hasattr(mz, "io")
