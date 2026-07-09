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

### 2. Daten- und Checkpoint-Pfad setzen

Daten und Checkpoints liegen typischerweise auf einer externen Festplatte, nicht im Projektordner. Beide werden per Volume in den Container eingehängt — `DATA_PATH` auf `/data`, `CHECKPOINT_PATH` auf `/checkpoints`.

**Linux / macOS** — in der Shell oder in einer `.env`-Datei im Projektroot:
```bash
export DATA_PATH=/pfad/zu/den/fundus-bildern
export CHECKPOINT_PATH=/pfad/zur/externen/platte/checkpoints
```

**Windows (PowerShell):**
```powershell
$env:DATA_PATH = "D:\fundus-data"
$env:CHECKPOINT_PATH = "E:\checkpoints"
```

Alternativ eine `.env`-Datei im Projektroot anlegen (wird von Docker Compose automatisch geladen):
```
DATA_PATH=/pfad/zu/den/fundus-bildern
CHECKPOINT_PATH=/pfad/zur/externen/platte/checkpoints
```

Wird `CHECKPOINT_PATH` nicht gesetzt, fällt Docker Compose auf `./checkpoints` im Projektordner zurück.

In `scripts/run.sh` müssen die Pfade dann auf die Mountpunkte **im Container** zeigen, nicht auf die Host-Pfade:
```bash
export PROJECT_ROOT="/workspace"
export DATA_ROOT="/data"
export INFERENCE_CHECKPOINT_FOLDER="/checkpoints/final-models"
```

### 3. E-Mail-Benachrichtigung einrichten (optional, aber empfohlen fürs Abmelden)

Der Container läuft standardmäßig über `scripts/entrypoint_with_notify.sh`: es mitschreibt den kompletten Trainingslog nach `training.log` und schickt danach — egal ob Erfolg oder Fehler — eine E-Mail mit einer Zusammenfassung (Loss-Werte, Checkpoint-Meldungen, Fehler/Traceback falls vorhanden).

Für den Versand über GMX mit App-Passwort:
1. In deinem GMX-Konto: **Einstellungen → Sicherheit → App-Passwörter** → neues App-Passwort erstellen (nicht dein normales Login-Passwort verwenden).
2. In der `.env`-Datei im Projektroot ergänzen:
```
SMTP_USER=deinname@gmx.de
SMTP_PASSWORD=das-app-passwort
NOTIFY_EMAIL_TO=deinname@gmx.de
```
`SMTP_HOST`/`SMTP_PORT` müssen für GMX nicht gesetzt werden (Standard: `mail.gmx.net:587`).

Sind `SMTP_USER`/`SMTP_PASSWORD` nicht gesetzt, wird die Benachrichtigung übersprungen — das Training läuft trotzdem normal, nur ohne Mail.

### 4. Training starten — detached, überlebt Abmelden

Damit das Training weiterläuft, nachdem du dich vom Klinik-PC abgemeldet hast, **detached** starten (`-d`), nicht mit `docker compose run` (das ist an dein Terminal gebunden):

```bash
docker compose up -d
```

Logs jederzeit live mitverfolgen (auch nach erneutem Einloggen):
```bash
docker compose logs -f
```

Container-Status prüfen:
```bash
docker compose ps
```

**Wichtig:** Abmelden ist unproblematisch, solange Docker Desktop weiterläuft (Docker Desktop läuft als Windows-Dienst/WSL2-Hintergrundprozess, unabhängig von deiner angemeldeten Sitzung). Der PC darf aber nicht **heruntergefahren** oder in den **Ruhezustand** versetzt werden — das stoppt auch die WSL2-VM und damit den Container.

### 5. Interaktive Shell
```bash
docker compose run odc bash
```
(nutzt weiterhin die normale, attached `run`-Variante ohne den Notify-Wrapper — praktisch zum Debuggen)

---

## Hinweise

- **Kein Datenverlust.** Checkpoints werden direkt auf die eingehängte externe Festplatte geschrieben (`CHECKPOINT_PATH` → `/checkpoints`) und bleiben nach dem Containerstopp erhalten. Plots landen im Projektverzeichnis (`/workspace`).
- **Patientendaten bleiben lokal.** Datenpfad und Checkpoint-Pfad werden nur als Volumes eingehängt — keine Dateien werden ins Image kopiert oder übertragen.
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
  -v "/pfad/zu/checkpoints:/checkpoints" \
  -e DATA_PATH=/data \
  -e CHECKPOINT_PATH=/checkpoints \
  -e PROJECT_ROOT=/workspace \
  ophthalmic-diffusion-classifier \
  bash scripts/run.sh train
```

**Windows (PowerShell):**
```powershell
docker run --gpus all --rm `
  -v "${PWD}:/workspace" `
  -v "D:\fundus-data:/data" `
  -v "E:\checkpoints:/checkpoints" `
  -e DATA_PATH=/data `
  -e CHECKPOINT_PATH=/checkpoints `
  -e PROJECT_ROOT=/workspace `
  ophthalmic-diffusion-classifier `
  bash scripts/run.sh train
```
