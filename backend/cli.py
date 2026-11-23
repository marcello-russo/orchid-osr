import sys
import warnings

warnings.warn("deprecated: usa 'orchid train' (cli.py shim)", DeprecationWarning, stacklevel=2)

from app.cli.main import app  # noqa: E402

if __name__ == "__main__":
    sys.exit(app())
