"""Fetch the GGUF weights (and optional vision projector) from Hugging Face."""

from huggingface_hub import hf_hub_download

from abliterated_vlm.config import Settings


def download(settings: Settings, include_mmproj: bool = True) -> None:
    files = [settings.model_file]
    if include_mmproj:
        files.append(settings.mmproj_file)
    settings.models_path.mkdir(parents=True, exist_ok=True)
    for filename in files:
        print(f"Downloading {settings.model_repo}/{filename} -> {settings.models_path}")
        path = hf_hub_download(
            repo_id=settings.model_repo,
            filename=filename,
            local_dir=settings.models_path,
        )
        print(f"  ok: {path}")
