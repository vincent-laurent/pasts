import runpy

import matplotlib
import pytest

matplotlib.use("Agg")


@pytest.mark.slow
def test_readme_model():
    """Run the full readme example script (examples/readme/model.py)."""
    runpy.run_path("examples/readme/model.py")
