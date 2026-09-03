# Docker Setup

This guide describes how to install, start, and configure the project in a Docker container — e.g. on a Windows PC with an NVIDIA GPU (the clinical workstation used for the Drusen experiments).

---

## 1. Prerequisites

The following must be installed once on the host:

| Software | Purpose |
|---|---|
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | Container runtime (Windows: enable the WSL2 backend) |
| NVIDIA GPU driver ≥ 525 | GPU access from inside the container |
| [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) | Enables `--gpus all` / `runtime: nvidia` in Docker |

Verify the setup:
```bash
docker run --gpus all --rm nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```
Your GPU should show up in the output.

---

## 2. Installation

### 2.1 Build the image (once, ~5–10 minutes)
```bash
docker compose build
```
The image is based on `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` (Python 3.11, CUDA 12.4) and is roughly 8 GB. Rebuilding is only necessary when `requirements.txt` or `Dockerfile` change — code changes take effect immediately because the project directory is mounted live, not baked into the image.

### 2.2 Set the data and checkpoint paths

Data and checkpoints typically live on an external drive, not inside the project folder. Both are mounted as volumes into the container — `DATA_PATH` → `/data`, `CHECKPOINT_PATH` → `/checkpoints`.

Create a `.env` file in the project root (loaded automatically by Docker Compose):
```
DATA_PATH=/path/to/the/fundus-data
CHECKPOINT_PATH=/path/to/the/external-drive/checkpoints
```

If `CHECKPOINT_PATH`/`DATA_PATH` are not set, Docker Compose falls back to `./checkpoints` / `./dataset/data` inside the project folder.

Inside the container, these are always mounted at the fixed paths `/data` and `/checkpoints` — the scripts already point at these container-internal paths; you never need to edit them yourself.

### 2.3 Set up e-mail notifications (optional)

By default the container runs through `scripts/entrypoint_with_notify.sh`: it captures the full training log to `training.log`, sends an hourly e-mail with the complete log while the run is still in progress, and sends a final summary e-mail once it finishes (success or failure), via `scripts/notification/notify_email.py`.

Example for GMX Mail - Create an app password:
1. In your GMX account: **Settings → Security → App passwords** → create a new app password (do not use your normal login password).
2. Add to the `.env` file in the project root:
```
SMTP_USER=yourname@gmx.de
SMTP_PASSWORD=the-app-password
NOTIFY_EMAIL_TO=yourname@gmx.de
```
`SMTP_HOST`/`SMTP_PORT` don't need to be set for GMX (default: `mail.gmx.net:587`).

If `SMTP_USER`/`SMTP_PASSWORD` are left unset, the notification step is skipped without failing the run — training proceeds normally, just without e-mails.

`.env` is only used for these two things — host paths (Section 2.2) and mail credentials. **What to actually run is not configured via `.env` or `-e` flags** — see Section 3.

---

## 3. Configuring what to run

There are exactly **two files** to edit, and no `-e` flags or `.env` entries are needed for any of this:

```text
scripts/run.sh              # what to run: MODEL, FUNCTION, DATA, CROSS_VALIDATION, BACKBONE/VARIANT, K_FOLDS, START_FOLD, PRETRAIN_CHECKPOINT
scripts/unet/fundus-unet.sh # unet-specific details: PRETRAINED_CHECKPOINT, EVALUATION_PER_STAGE, UNCERTAINTY_ESTIMATION or other training configurations
```

Each variable is declared where it is used, in the form `export VAR="${VAR:-default}"` (or `${VAR-default}` for `PRETRAINED_CHECKPOINT`, see below) — to change what a run does, edit the default value after the `:-` directly at that line, then start the container (Section 4). The relevant lines:

- In `scripts/run.sh`, near the top: `MODEL`, `FUNCTION`, `DATA`, `BACKBONE`, `VARIANT`.
- In `scripts/run.sh`, inside the `CROSS_VALIDATION=1` block: `K_FOLDS`, `START_FOLD`, `PRETRAIN_CHECKPOINT` (the latter only used for `FUNCTION=finetune`).
- In `scripts/unet/fundus-unet.sh`, under "Classification parameters" / "Training parameters": `EVALUATION_PER_STAGE`, `UNCERTAINTY_ESTIMATION`, `PRETRAINED_CHECKPOINT` (used for a **single**, non-CV `finetune` run — this is a different variable from `PRETRAIN_CHECKPOINT` above, see the note below).

Section 5 lists every supported combination as a concrete edit + command.

### Why `${VAR:-default}` and not a plain value?

The cross-validation orchestration scripts (`scripts/cross-validation/*.sh`) work by calling `bash scripts/run.sh` again once per fold, first exporting that fold's `MODEL`/`FUNCTION`/`FOLD`. The `${VAR:-default}` form means a value already present in the environment (i.e. set by the orchestration script for the current fold) is respected instead of being overwritten by the default — so editing the default is safe and does not break cross-validation. You will not normally need to set these via the environment yourself; editing the file directly is the intended workflow.

### One thing to know before you edit

**`PRETRAINED_CHECKPOINT` (in `fundus-unet.sh`) already has a default**, so a plain single `MODEL=unet FUNCTION=finetune CROSS_VALIDATION=0` run works out of the box — it falls back to the Mogon Phase-1 pretrain checkpoint at `/checkpoints/final-models/drusen-unet/new-run/pretrain/pretrain-mogon` (relative to your mounted `CHECKPOINT_PATH`) unless you set it to your own checkpoint path. Set `PRETRAINED_CHECKPOINT=""` instead if you want to train from scratch without any pretrained weights — though `FUNCTION=train` (Section 5.3) is the intended way to do that, since `finetune.py` refuses to start with neither `RESUME=1` nor a non-empty `PRETRAINED_CHECKPOINT`.

---

## 4. Starting the container

Once `scripts/run.sh` is configured (Section 3):

### 4.1 Detached, survives logging off

To keep training running after logging off the clinical PC, start **detached** (`-d`), not with `docker compose run` (which is bound to your terminal):

```bash
docker compose up -d
```

Follow logs live at any time (also after logging back in):
```bash
docker compose logs -f
```

Check container status:
```bash
docker compose ps
```

Stop it:
```bash
docker compose down
```

**Important:** logging off is fine as long as Docker Desktop keeps running (it runs as a Windows service / WSL2 background process, independent of your logged-in session). The PC must **not** be shut down or put to **sleep/hibernate** — that also stops the WSL2 VM and therefore the container.

### 4.2 Interactive shell (debugging, manual commands)

```bash
docker compose run --rm odc bash
```
This opens a shell *inside* the container with the same volume mounts (`/workspace`, `/data`, `/checkpoints`), without the notify wrapper — useful for debugging, or for running manual recovery commands. Inside, running `bash scripts/run.sh` executes whatever is currently configured in the file. Exit with `exit`; `--rm` cleans the container up automatically (your data stays on the mounted volumes).

---

## 5. What each configuration runs

Every row below is an edit to `scripts/run.sh`'s `USER CONFIGURATION` block, followed by `docker compose up -d` (or `docker compose run --rm odc bash scripts/run.sh` for an attached one-off run).

### 5.1 Baseline classifier (ResNet50)

| Goal | Set in `scripts/run.sh` |
|---|---|
| Train a single run | `MODEL=baseline`, `FUNCTION=train`, `CROSS_VALIDATION=0` |
| Inference on an existing checkpoint | `MODEL=baseline`, `FUNCTION=inference`, `CROSS_VALIDATION=0` |
| 10-fold cross-validation (train + evaluate every fold, then aggregate mean ± std) | `MODEL=baseline`, `CROSS_VALIDATION=1` |

### 5.2 Diffusion classifier (UNet) — pretrained finetuning

| Goal | Set in `scripts/run.sh` |
|---|---|
| Finetune a single run from the Mogon Phase-1 checkpoint | `MODEL=unet`, `FUNCTION=finetune`, `CROSS_VALIDATION=0` (uses the default `PRETRAINED_CHECKPOINT` in `fundus-unet.sh`; override it there to use your own checkpoint) |
| Inference on an existing (finetuned) checkpoint | `MODEL=unet`, `FUNCTION=inference`, `CROSS_VALIDATION=0` |
| Explanation / counterfactual visualization | `MODEL=unet`, `FUNCTION=explain`, `CROSS_VALIDATION=0` |
| 10-fold cross-validation (finetune + evaluate every fold from the Mogon checkpoint, then aggregate) | `MODEL=unet`, `FUNCTION=finetune`, `CROSS_VALIDATION=1` (optionally change `PRETRAIN_CHECKPOINT`) |

### 5.3 Diffusion classifier (UNet) — training from scratch

| Goal | Set in `scripts/run.sh` |
|---|---|
| Train a single run from scratch (no pretrained weights) | `MODEL=unet`, `FUNCTION=train`, `CROSS_VALIDATION=0` |
| 10-fold cross-validation, training from scratch on every fold, then aggregating | `MODEL=unet`, `FUNCTION=train`, `CROSS_VALIDATION=1` |

### 5.4 Resuming or reconfiguring a cross-validation run

For any of the three CV variants above, additionally set in `scripts/run.sh`:
- `START_FOLD=<i>` — resume from a specific fold (e.g. after fixing an interrupted fold 2: `START_FOLD=2`), skipping already-completed and archived folds.
- `K_FOLDS=<n>` — run fewer/more folds than the default 10.

### 5.5 Advanced UNet options (single runs)

Set the same way, at the top of `scripts/unet/fundus-unet.sh`:

| Variable | Effect |
|---|---|
| `EVALUATION_PER_STAGE` | Number of Monte Carlo majority-voting samples per classification (default `[51]`) |
| `UNCERTAINTY_ESTIMATION=true` | Also record per-sample uncertainty during `inference.py` (writes `uncertainty_predictions.json` next to the checkpoint) |

(`FOLD` and `DRUSEN_MODEL_DIR`, further down in the same file, are set automatically by the cross-validation scripts for you — you don't need to touch them for anything in this section.)

### 5.6 Other datasets/models inherited from the original framework

`MODEL=dit`, `MODEL=sd`, and `DATA=isic`/`DATA=chexpert` are inherited from the upstream Favero et al. framework and follow the same `MODEL`/`FUNCTION`/`DATA` pattern. They are not part of the Drusen experiments this thesis is built around and are not covered further here.

---

## 6. Notes

- **No data loss.** Checkpoints are written directly to the mounted external drive (`CHECKPOINT_PATH` → `/checkpoints`) and persist after the container stops. Plots/results are written into the project directory (`/workspace`), which is also mounted live.
- **Patient data stays local.** The data and checkpoint paths are only mounted as volumes — no files are copied into the image or transmitted anywhere.
- **`Input/output error` during checkpoint archiving** (seen mid-cross-validation) is a host storage fault, not a bug in the scripts — check `df -h` on the host and whether `CHECKPOINT_PATH` points at a drive/network share that stayed connected throughout the run.
- Rebuilding the image (`docker compose build`) is only needed after changing `requirements.txt` or `Dockerfile`.

---

## Appendix: Manual `docker run` reference

If Docker Compose is unavailable, the equivalent manual invocation (runs whatever is currently configured in `scripts/run.sh`, see Section 3):

**Linux / macOS:**
```bash
docker run --gpus all --rm \
  -v "$(pwd):/workspace" \
  -v "/path/to/data:/data" \
  -v "/path/to/checkpoints:/checkpoints" \
  -e PROJECT_ROOT=/workspace \
  ophthalmic-diffusion-classifier \
  bash scripts/run.sh
```

**Windows (PowerShell):**
```powershell
docker run --gpus all --rm `
  -v "${PWD}:/workspace" `
  -v "D:\fundus-data:/data" `
  -v "E:\checkpoints:/checkpoints" `
  -e PROJECT_ROOT=/workspace `
  ophthalmic-diffusion-classifier `
  bash scripts/run.sh
```
