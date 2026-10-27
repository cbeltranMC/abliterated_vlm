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


def cmd_build(args: argparse.Namespace) -> int:
    from abliterated_vlm.build import build

    build(load_settings(), jobs=args.jobs)
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    from abliterated_vlm import server

    return server.start(load_settings(), wait=args.wait)


def cmd_stop(args: argparse.Namespace) -> int:
    from abliterated_vlm import server

    return server.stop(load_settings())


def cmd_restart(args: argparse.Namespace) -> int:
    from abliterated_vlm import server

    settings = load_settings()
    server.stop(settings)
    return server.start(settings, wait=args.wait)


def cmd_status(args: argparse.Namespace) -> int:
    from abliterated_vlm import server

    return server.status(load_settings())


def cmd_logs(args: argparse.Namespace) -> int:
    from abliterated_vlm import server

    return server.logs(follow=args.follow, lines=args.lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vlm", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser("init", help="create .env from .env.example")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("download", help="download the GGUF model from Hugging Face")
    p.add_argument("--no-mmproj", action="store_true", help="skip the vision projector")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("build", help="clone and compile llama.cpp's llama-server with CUDA")
    p.add_argument("-j", "--jobs", type=int, help="parallel compile jobs (default: all cores)")
    p.set_defaults(func=cmd_build)

    for name, func, help_ in [
        ("start", cmd_start, "turn the API on (load the model into the GPU)"),
        ("restart", cmd_restart, "stop and start again (e.g. after editing .env)"),
    ]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--wait", type=float, default=600, help="seconds to wait for the model to load")
        p.set_defaults(func=func)

    p = sub.add_parser("stop", help="turn the API off and free the GPU")
    p.set_defaults(func=cmd_stop)

    p = sub.add_parser("status", help="show whether the server is running and GPU usage")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("logs", help="show the llama-server log")
    p.add_argument("-f", "--follow", action="store_true")
    p.add_argument("-n", "--lines", type=int, default=50)
    p.set_defaults(func=cmd_logs)


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
