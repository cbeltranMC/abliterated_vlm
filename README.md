# abliterated_vlm

Run [huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF)
on this workstation with [llama.cpp](https://github.com/ggml-org/llama.cpp) and use it as a
coding-agent model in [OpenCode](https://opencode.ai) from a laptop over SSH. The API is
switched **on and off on demand**, so the GPU is free for other work the rest of the time.

```
 laptop                                         workstation (RTX 3090)
┌──────────────────────────┐   SSH tunnel   ┌──────────────────────────────────┐
│ opencode                 │                │ llama-server (OpenAI API)        │
│  └─ http://127.0.0.1:8080├───────────────►│  127.0.0.1:8080, not on the LAN  │
│ vlm-remote on | off      ├── ssh ────────►│ uv run vlm start | stop          │
└──────────────────────────┘                └──────────────────────────────────┘
```

## Model choice

| | |
|---|---|
| File | `Huihui-Qwen3.8-27B-abliterated-Q4_K_L.gguf` (19.5 GiB), the biggest 4-bit quant in the repo: Q4_K, with the abliterated tensors (`token_embd`, `output`, `ffn_down`, `ssm_out`, `attn_output`) kept at Q8_0 |
| Vision | `mmproj-model-bf16.gguf` (0.9 GiB), loaded only with `ENABLE_VISION=1` |
| Context | 96k tokens by default, with a q8_0 KV cache, everything on the GPU |
| Speed (3090) | ~1,000–1,300 tok/s prompt processing, ~31–37 tok/s generation |

Measured VRAM use:

| `CTX_SIZE` | Text only | With vision |
|---|---|---|
| 65536 | 21.6 GiB | 22.7 GiB |
| 98304 (default) | 22.8 GiB | n/a (too tight) |
| 131072 | 24.0 GiB (not recommended) | n/a |

> This is an abliterated (uncensored) model with reduced safety filtering. Review what it
> produces, especially commands it wants to run.

## Workstation: one-time setup

Requirements: NVIDIA driver, CUDA toolkit (`nvcc`), `cmake`, `git`, [uv](https://docs.astral.sh/uv/).

```bash
uv sync               # create the virtualenv
uv run vlm init       # create .env from .env.example
uv run vlm download   # ~21 GB into models/
uv run vlm build      # clone and compile llama.cpp's llama-server into vendor/ (~10 min)
```

`vlm build` compiles from source because the prebuilt ggml-org CUDA images need a
CUDA ≥ 12.8 driver, and this machine runs driver 535 (CUDA 12.2). To update llama.cpp
later, run `uv run vlm build` again (it pulls `LLAMA_CPP_REF`, `master` by default).

## Workstation: turning it on and off

```bash
uv run vlm start      # load the model into the GPU, wait until the API is ready
uv run vlm status     # running/stopped, context size, GPU memory
uv run vlm logs -f    # follow logs/llama-server.log
uv run vlm restart    # apply .env changes
uv run vlm stop       # unload and free all GPU memory
```

The server runs detached, so it keeps running after the terminal or SSH session that
started it closes. It does **not** start at boot. It only runs while you have it on.

## Laptop: one-time setup

1. **SSH access** to the workstation with a key (no password prompts), e.g. in
   `~/.ssh/config` on the laptop:

   ```
   Host workstation
       HostName 192.168.21.101      # LAN address; or 10.0.0.10 over the WireGuard VPN
       User mcso
   ```

   Check it works: `ssh workstation true`.

2. **Install the helper** (`jq` is optional, used to merge into an existing OpenCode config):

   ```bash
   scp workstation:Desktop/Projects/abliterated_vlm/scripts/laptop/vlm-remote ~/.local/bin/
   echo 'export VLM_HOST=workstation' >> ~/.bashrc   # or ~/.zshrc
   ```

   Optional variables: `VLM_DIR` (repo path on the workstation, default
   `~/Desktop/Projects/abliterated_vlm`), `VLM_PORT` (default 8080) and `VLM_LOCAL_PORT`
   (laptop-side port, if 8080 is busy there).

3. **Add the model to OpenCode**:

   ```bash
   vlm-remote config
   ```

   This writes (or merges) a `workstation` provider into `~/.config/opencode/opencode.json`
   pointing at `http://127.0.0.1:8080/v1`. It is generated from the workstation's `.env`
   (`uv run vlm opencode-config`), so the context limit, API key and vision flag stay in sync.
   Run it again after changing those settings.

## Laptop: daily workflow

```bash
vlm-remote on         # start the model on the workstation + open the SSH tunnel
opencode              # pick "Workstation llama.cpp / Huihui Qwen3.8 27B" with /models
vlm-remote off        # close the tunnel + stop the model (frees the workstation GPU)
```

Shortcuts and checks:

```bash
vlm-remote opencode                                     # 'on', then launch opencode
opencode -m workstation/huihui-qwen3.8-27b              # use the model directly
vlm-remote status                                       # server + tunnel state
vlm-remote logs                                         # follow the server log
```

If the laptop sleeps or the network drops, the tunnel dies but the model keeps running on
the workstation. Run `vlm-remote tunnel` (or `vlm-remote on`, which is idempotent) to reconnect.

### Without the helper (e.g. Windows)

```bash
ssh workstation "cd ~/Desktop/Projects/abliterated_vlm && ~/.local/bin/uv run vlm start"
ssh -N -L 8080:127.0.0.1:8080 workstation          # keep open while using OpenCode
ssh workstation "cd ~/Desktop/Projects/abliterated_vlm && ~/.local/bin/uv run vlm stop"
```

and copy the output of `uv run vlm opencode-config` into OpenCode's `opencode.json`.

## Configuration (`.env`)

Every setting has a default in [src/abliterated_vlm/config.py](src/abliterated_vlm/config.py).
Real environment variables override `.env` (e.g. `CTX_SIZE=65536 uv run vlm restart`).

| Variable | Default | Notes |
|---|---|---|
| `MODEL_FILE` | `...-Q4_K_L.gguf` | any GGUF from `MODEL_REPO`; run `vlm download` after changing it |
| `MODEL_ALIAS` | `huihui-qwen3.8-27b` | model id the API and OpenCode use |
| `HOST` / `PORT` | `127.0.0.1` / `8080` | keep localhost; access goes through SSH |
| `API_KEY` | empty | optional bearer token; `vlm-remote config` propagates it |
| `CTX_SIZE` | `98304` | see the VRAM table above |
| `CACHE_TYPE_K/V` | `q8_0` | `f16` doubles KV memory |
| `ENABLE_VISION` | `0` | `1` loads the mmproj so OpenCode can send images (use `CTX_SIZE=65536`) |
| `PARALLEL` | `1` | concurrent requests; the context is split between them |
| `EXTRA_ARGS` | empty | any other `llama-server` flags |
| `LLAMA_CPP_REF` | `master` | pin a llama.cpp tag/commit for reproducible builds |

Sampling uses Qwen3's recommended thinking-mode settings (temp 0.6, top-p 0.95, top-k 20).
The chat template is the one embedded in the GGUF (`--jinja`), which handles tool calling
and reasoning.

## Troubleshooting

- **`vlm start` exits immediately**: it prints the end of `logs/llama-server.log`. The usual
  cause is out-of-memory because something else is using the GPU (`nvidia-smi`). Lower
  `CTX_SIZE` or stop the other job.
- **`address already in use`**: another process holds `PORT`; change it in `.env` and re-run
  `vlm-remote config`.
- **Tunnel fails on the laptop**: something already listens on local port 8080; set
  `VLM_LOCAL_PORT=18080` and re-run `vlm-remote config`.
- **`uv: command not found` over SSH**: the helper adds `~/.local/bin` to `PATH`; if uv lives
  elsewhere on the workstation, adjust `remote()` in `vlm-remote`.

## Layout

```
src/abliterated_vlm/   CLI (`uv run vlm`): config, download, build, server, opencode
scripts/laptop/        vlm-remote: laptop-side on/off + tunnel + OpenCode config
models/  vendor/       weights and llama.cpp build (git-ignored)
run/  logs/            PID file and server log (git-ignored)
```
