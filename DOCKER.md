# Docker Setup

Dieser Guide beschreibt wie man das Projekt in einem Docker Container startet — z.B. auf einem Windows-PC mit NVIDIA GPU.

## Voraussetzungen

Folgendes muss einmalig auf dem Host installiert sein (ggf. IT fragen):

| Software | Zweck |
|---|---|
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | Container-Runtime (Windows: WSL2-Backend aktivieren) |
| NVIDIA GPU Treiber ≥ 525 | GPU-Zugriff aus dem Container |
| [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) | Ermöglicht `--gpus all` in Docker |

Setup prüfen:
```bash
docker run --gpus all --rm nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```
Die eigene GPU sollte in der Ausgabe erscheinen.

---

## Schnellstart mit Docker Compose

### 1. Image bauen (einmalig, ~5–10 Minuten)
```bash
docker compose build
```

### 2. Datenpfad setzen

**Linux / macOS** — in der Shell oder in einer `.env`-Datei im Projektroot:
```bash
export DATA_PATH=/pfad/zu/den/fundus-bildern
```

**Windows (PowerShell):**
```powershell
$env:DATA_PATH = "D:\fundus-data"
```

Alternativ eine `.env`-Datei im Projektroot anlegen (wird von Docker Compose automatisch geladen):
```
DATA_PATH=/pfad/zu/den/fundus-bildern
```

### 3. Training starten
```bash
# Startet scripts/run.sh
docker compose run odc
```

### 4. Interaktive Shell
```bash
docker compose run odc bash
```

---

## Hinweise

- **Kein Datenverlust.** Checkpoints und Plots werden direkt in das eingehängte Projektverzeichnis geschrieben und bleiben nach dem Containerstopp erhalten.
- **Patientendaten bleiben lokal.** Der Datenpfad wird nur als Volume eingehängt — keine Dateien werden ins Image kopiert oder übertragen.
- **Image neu bauen** ist nur nötig wenn sich `requirements.txt` oder `Dockerfile` ändern. Codeänderungen sind sofort wirksam da das Projektverzeichnis live eingehängt ist.
- Das Image ist ~8 GB groß.

---

## Referenz: Manueller `docker run`

Falls Docker Compose nicht verfügbar ist:

**Linux / macOS:**
```bash
docker run --gpus all --rm \
  -v "$(pwd):/workspace" \
  -v "/pfad/zu/daten:/data" \
  -e DATA_PATH=/data \
  -e PROJECT_ROOT=/workspace \
  ophthalmic-diffusion-classifier \
  bash scripts/run.sh train
```

**Windows (PowerShell):**
```powershell
docker run --gpus all --rm `
  -v "${PWD}:/workspace" `
  -v "D:\fundus-data:/data" `
  -e DATA_PATH=/data `
  -e PROJECT_ROOT=/workspace `
  ophthalmic-diffusion-classifier `
  bash scripts/run.sh train
```
