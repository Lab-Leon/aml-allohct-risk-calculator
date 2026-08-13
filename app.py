"""Streamlit entrypoint.

Execute the application script for every Streamlit session. A wildcard import
only runs the module once per Python process, which leaves later sessions with
an empty page.
"""

from pathlib import Path
from runpy import run_path


run_path(
    str(Path(__file__).resolve().parent / "calculator" / "app.py"),
    run_name="__main__",
)
