import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_script(script_path: str, *arguments: str) -> None:
    command = [
        sys.executable,
        str(ROOT / script_path),
        *arguments,
    ]

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    run_script("src/phase1.py")
    run_script("src/phase2.py")
    run_script("src/phase3.py")
    run_script("src/deployment.py", "all")


if __name__ == "__main__":
    main()