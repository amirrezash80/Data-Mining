import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_script(script: str, *arguments: str) -> None:
    command = [
        sys.executable,
        str(ROOT / script),
        *arguments,
    ]

    print("Running:", " ".join(command))

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        choices=[
            "phase1",
            "phase2",
            "phase3",
            "deployment",
        ],
        default="phase1",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    stages = [
        (
            "phase1",
            "src/phase1.py",
            (),
        ),
        (
            "phase2",
            "src/phase2.py",
            (),
        ),
        (
            "phase3",
            "src/phase3.py",
            (),
        ),
        (
            "deployment",
            "src/deployment.py",
            ("all",),
        ),
    ]

    stage_names = [
        stage[0]
        for stage in stages
    ]

    start_position = stage_names.index(
        arguments.start
    )

    for _, script, script_arguments in stages[
        start_position:
    ]:
        run_script(
            script,
            *script_arguments,
        )


if __name__ == "__main__":
    main()