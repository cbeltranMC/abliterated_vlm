"""Clone and compile llama.cpp's `llama-server` with CUDA.

The prebuilt ggml-org CUDA images need a CUDA >= 12.8 driver; compiling
against the locally installed toolkit works with whatever driver is present.
"""

import os
import shutil
import subprocess
from pathlib import Path

from abliterated_vlm.config import ROOT, Settings

LLAMA_DIR = ROOT / "vendor" / "llama.cpp"
BUILD_DIR = LLAMA_DIR / "build"
SERVER_BIN = BUILD_DIR / "bin" / "llama-server"


def _run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _find_nvcc() -> str:
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists():
        raise SystemExit("nvcc not found: install the CUDA toolkit or put nvcc on PATH.")
    return nvcc


def build(settings: Settings, jobs: int | None = None) -> Path:
    if not LLAMA_DIR.exists():
        LLAMA_DIR.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", settings.llama_cpp_repo, str(LLAMA_DIR)])
    _run(["git", "fetch", "--tags", "origin"], cwd=LLAMA_DIR)
    _run(["git", "checkout", settings.llama_cpp_ref], cwd=LLAMA_DIR)
    if settings.llama_cpp_ref in {"master", "main"}:
        _run(["git", "pull", "--ff-only"], cwd=LLAMA_DIR)

    env = {**os.environ, "CUDACXX": _find_nvcc()}
    _run(
        [
            "cmake", "-S", str(LLAMA_DIR), "-B", str(BUILD_DIR),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_CUDA=ON",
            f"-DCMAKE_CUDA_ARCHITECTURES={settings.cuda_architectures}",
            "-DLLAMA_CURL=OFF",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
        ],
        env=env,
    )
    _run(
        ["cmake", "--build", str(BUILD_DIR), "--config", "Release",
         "--target", "llama-server", "-j", str(jobs or os.cpu_count() or 4)],
        env=env,
    )
    print(f"Built {SERVER_BIN}")
    return SERVER_BIN
