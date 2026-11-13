from importlib.metadata import version

import wrapevofs


def test_package_version_is_consistent() -> None:
    assert wrapevofs.__version__ == "0.3.0"
    assert version("wrapevofs") == wrapevofs.__version__
