"""Run project commands with the current Python interpreter."""

import argparse
import subprocess
import sys

from src.common.config import DATA_DIR, ROOT


def run_module(module: str, *args: str) -> None:
    subprocess.run([sys.executable, "-m", module, *args], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Run Ruff, tests and CSV schema validation")
    test = commands.add_parser("test", help="Run all tests or selected pytest paths")
    test.add_argument("paths", nargs="*", help="Test paths or pytest node IDs")
    tokens = commands.add_parser("tokens", help="Compute token counts and fertility")
    tokens.add_argument("--input")
    tokens.add_argument("--output")
    commands.add_parser("dashboard", help="Start the offline Streamlit dashboard")
    commands.add_parser("done", help="Check the read-only release evidence gates")
    args = parser.parse_args()

    if args.command == "check":
        run_module("ruff", "check", ".")
        run_module("pytest", "-q")  # Includes the CSV contract tests.
        if (DATA_DIR / "validate_prompts.py").is_file():
            run_module("data.validate_prompts")
    elif args.command == "test":
        run_module("pytest", "-q", *args.paths)
    elif args.command == "tokens":
        if not (ROOT / "src" / "tokens.py").stat().st_size:
            sys.exit("Token counting is pending T-001.")
        options = []
        for name in ("input", "output"):
            if value := getattr(args, name):
                options.extend((f"--{name}", value))
        run_module("src.tokens", *options)
    elif args.command == "dashboard":
        dashboard = ROOT / "app" / "dashboard.py"
        if not dashboard.stat().st_size:
            sys.exit("The offline dashboard is pending T-006.")
        run_module("streamlit", "run", str(dashboard))
    elif args.command == "done":
        run_module("src.done_check")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
