import py_compile
try:
    py_compile.compile("src/experiments/run_ablations.py", doraise=True)
    print("Syntax OK")
except Exception as e:
    print(f"Syntax Error: {e}")
