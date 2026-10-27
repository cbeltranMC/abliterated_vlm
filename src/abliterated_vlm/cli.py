"""Command-line entry point: `uv run vlm <command>`."""

import argparse
import sys

from abliterated_vlm import __version__
from abliterated_vlm.config import ENV_EXAMPLE, ENV_FILE, load_settings


def cmd_init(args: argparse.Namespace) -> int:
    if ENV_FILE.exists():
        print(f"{ENV_FILE} already exists, leaving it untouched.")
        return 0
    ENV_FILE.write_text(ENV_EXAMPLE.read_text())
    print(f"Created {ENV_FILE} from .env.example")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    from abliterated_vlm.download import download

    download(load_settings(), include_mmproj=not args.no_mmproj)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vlm", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser("init", help="create .env from .env.example")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("download", help="download the GGUF model from Hugging Face")
    p.add_argument("--no-mmproj", action="store_true", help="skip the vision projector")
    p.set_defaults(func=cmd_download)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
