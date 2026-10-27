"""Start / stop / inspect a background `llama-server` process.

The server runs detached from the shell that started it (so it survives the
SSH session closing) and is tracked through a PID file. Stopping it frees all
VRAM, which is the point: the workstation is shared with other work.
"""

import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from abliterated_vlm.build import SERVER_BIN
from abliterated_vlm.config import ROOT, Settings

RUN_DIR = ROOT / "run"
LOG_DIR = ROOT / "logs"
PID_FILE = RUN_DIR / "llama-server.pid"
LOG_FILE = LOG_DIR / "llama-server.log"


def server_args(settings: Settings) -> list[str]:
    args = [
        str(SERVER_BIN),
        "--model", str(settings.model_path),
        "--alias", settings.model_alias,
        "--host", settings.host,
        "--port", str(settings.port),
        "--ctx-size", str(settings.ctx_size),
        "--n-gpu-layers", str(settings.n_gpu_layers),
        "--flash-attn", "on",
        "--cache-type-k", settings.cache_type_k,
        "--cache-type-v", settings.cache_type_v,
        "--parallel", str(settings.parallel),
        # Chat template from the GGUF: required for tool calling (OpenCode's agent loop).
        "--jinja",
        # Qwen3 recommended sampling for thinking mode.
        "--temp", "0.6", "--top-p", "0.95", "--top-k", "20", "--min-p", "0",
    ]
    if settings.enable_vision:
        args += ["--mmproj", str(settings.mmproj_path)]
    if settings.api_key:
        args += ["--api-key", settings.api_key]
    args += shlex.split(settings.extra_args)
    return args


def read_pid() -> int | None:
    try:
        pid = int(PID_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        PID_FILE.unlink(missing_ok=True)
        return None
    except PermissionError:
        pass
    return pid


def _get(settings: Settings, path: str, timeout: float = 3) -> tuple[int, dict | None]:
    req = urllib.request.Request(settings.base_url + path)
    if settings.api_key:
        req.add_header("Authorization", f"Bearer {settings.api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, None
    except (urllib.error.URLError, OSError, ValueError):
        return 0, None


def is_healthy(settings: Settings) -> bool:
    status, _ = _get(settings, "/health")
    return status == 200


def start(settings: Settings, wait: float = 600) -> int:
    pid = read_pid()
    if pid:
        print(f"llama-server already running (pid {pid}) on {settings.base_url}")
        return 0
    if not SERVER_BIN.exists():
        print(f"{SERVER_BIN} not found: run `uv run vlm build` first.", file=sys.stderr)
        return 1
    missing = [p for p in [settings.model_path] + ([settings.mmproj_path] if settings.enable_vision else []) if not p.exists()]
    if missing:
        print(f"Missing {', '.join(map(str, missing))}: run `uv run vlm download` first.", file=sys.stderr)
        return 1

    RUN_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    args = server_args(settings)
    with LOG_FILE.open("ab") as log:
        log.write(f"\n=== {time.strftime('%F %T')} {shlex.join(args)}\n".encode())
        log.flush()
        proc = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # keep running after the SSH session ends
        )
    PID_FILE.write_text(str(proc.pid))
    print(f"Started llama-server (pid {proc.pid}), loading {settings.model_file} ...")

    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            PID_FILE.unlink(missing_ok=True)
            print(f"llama-server exited with code {proc.returncode}; see {LOG_FILE}", file=sys.stderr)
            _tail(LOG_FILE, 20)
            return 1
        if is_healthy(settings):
            print(f"Ready: {settings.base_url}/v1 (model '{settings.model_alias}')")
            return 0
        time.sleep(2)
    print(f"Still loading after {wait:.0f}s; check `uv run vlm logs`.", file=sys.stderr)
    return 1


def stop(settings: Settings, timeout: float = 30) -> int:
    pid = read_pid()
    if not pid:
        print("llama-server is not running.")
        return 0
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and read_pid():
        time.sleep(0.5)
    if read_pid():
        print("Did not exit after SIGTERM, sending SIGKILL.")
        os.killpg(pid, signal.SIGKILL)
        time.sleep(1)
    PID_FILE.unlink(missing_ok=True)
    print(f"Stopped llama-server (pid {pid}); GPU memory released.")
    return 0


def status(settings: Settings) -> int:
    pid = read_pid()
    if not pid:
        print("llama-server: stopped")
    else:
        code, _ = _get(settings, "/health")
        state = {200: "ready", 503: "loading model"}.get(code, "starting")
        print(f"llama-server: running (pid {pid}), {state}")
        print(f"  endpoint: {settings.base_url}/v1   model: {settings.model_alias}")
        _, props = _get(settings, "/props")
        if props:
            n_ctx = props.get("default_generation_settings", {}).get("n_ctx")
            print(f"  context:  {n_ctx} tokens")
    _print_gpu()
    return 0 if pid else 3


def _print_gpu() -> None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        print(f"  gpu:      {out}")
    except (OSError, subprocess.CalledProcessError):
        pass


def _tail(path: Path, n: int) -> None:
    if path.exists():
        print("".join(path.read_text(errors="replace").splitlines(keepends=True)[-n:]), file=sys.stderr)


def logs(follow: bool, lines: int) -> int:
    if not LOG_FILE.exists():
        print("No logs yet.")
        return 0
    cmd = ["tail", "-n", str(lines)] + (["-f"] if follow else []) + [str(LOG_FILE)]
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 0
