import py_compile


def test_ablations_syntax():
    """Verify src/experiments/run_ablations.py compiles without syntax errors."""
    py_compile.compile("src/experiments/run_ablations.py", doraise=True)
