"""Render the application in a fresh namespace on every Streamlit run."""

from pathlib import Path


def _render_application() -> None:
    application = Path(__file__).resolve().parent / "calculator" / "app.py"
    # Importing the UI only renders the first session. runpy.run_path renders
    # again, but temporarily changes process-wide sys.modules and sys.argv.
    # Keep UI globals local to this run without either shared-state side effect.
    namespace = {
        "__name__": "__main__",
        "__file__": str(application),
        "__package__": None,
        "__spec__": None,
        "__cached__": None,
        "__loader__": None,
    }
    source = application.read_text(encoding="utf-8")
    exec(compile(source, str(application), "exec"), namespace)


_render_application()
