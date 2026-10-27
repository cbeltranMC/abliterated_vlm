"""Settings loaded from `.env` in the project root, with defaults for everything."""

import os
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"


@dataclass(frozen=True)
class Settings:
    model_repo: str = "huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF"
    model_file: str = "Huihui-Qwen3.8-27B-abliterated-Q4_K_L.gguf"
    mmproj_file: str = "mmproj-model-bf16.gguf"
    models_dir: str = "models"
    model_alias: str = "huihui-qwen3.8-27b"

    host: str = "127.0.0.1"
    port: int = 8080
    api_key: str = ""

    ctx_size: int = 98304
    n_gpu_layers: int = 999
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    parallel: int = 1
    enable_vision: bool = False
    extra_args: str = ""

    llama_cpp_repo: str = "https://github.com/ggml-org/llama.cpp"
    llama_cpp_ref: str = "master"
    cuda_architectures: str = "86"

    @property
    def models_path(self) -> Path:
        return _resolve(self.models_dir)

    @property
    def model_path(self) -> Path:
        return self.models_path / self.model_file

    @property
    def mmproj_path(self) -> Path:
        return self.models_path / self.mmproj_file

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def load_settings() -> Settings:
    """Defaults < `.env` < real environment variables."""
    raw = _read_env_file(ENV_FILE)
    kwargs = {}
    for f in fields(Settings):
        key = f.name.upper()
        value = os.environ.get(key, raw.get(key))
        if value is None or value == "" and f.type is not str:
            continue
        if f.type is int:
            kwargs[f.name] = int(value)
        elif f.type is bool:
            kwargs[f.name] = value.lower() in {"1", "true", "yes", "on"}
        else:
            kwargs[f.name] = value
    return Settings(**kwargs)
