import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "matplotlib",
    "seaborn",
    "yaml",
    "joblib",
    "pyarrow",
    "pandera",
    "streamlit",
    "plotly",
]

REQUIRED_FILES = [
    "params.yaml",
    "params_phase2.yaml",
    "params_phase3.yaml",
    "params_deployment.yaml",
    "src/phase1.py",
    "src/phase2.py",
    "src/phase3.py",
    "src/deployment.py",
    "src/run_project.py",
    "dashboard/app.py",
]


def main() -> None:
    result = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "modules": {},
        "files": {},
    }

    failed = False

    if sys.version_info < (3, 11):
        result["python_status"] = "unsupported"
        failed = True
    else:
        result["python_status"] = "supported"

    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(
                module_name
            )

            result["modules"][module_name] = {
                "available": True,
                "version": getattr(
                    module,
                    "__version__",
                    "unknown",
                ),
            }
        except Exception as error:
            result["modules"][module_name] = {
                "available": False,
                "error": str(error),
            }
            failed = True

    for relative_path in REQUIRED_FILES:
        path = ROOT / relative_path
        exists = path.exists()

        result["files"][relative_path] = exists

        if not exists:
            failed = True

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()