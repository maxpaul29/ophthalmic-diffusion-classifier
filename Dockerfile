FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /workspace

# Force stdout/stderr to be unbuffered. Without this, Python switches to block
# buffering whenever stdout isn't a TTY (e.g. piped through `tee` as in
# entrypoint_with_notify.sh), so print() output can sit invisible in an
# internal buffer for a very long time instead of appearing in the log/console
# in real time — easily mistaken for a hang.
ENV PYTHONUNBUFFERED=1

# OpenCV system dependencies (for CLAHE)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies.
# torch is already present in the base image at the correct version;
# pip will skip it and install only the remaining packages.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project code and data are mounted at runtime via -v / --volume.
# Nothing is baked into the image so checkpoints and dataset files
# stay on the host filesystem and survive container restarts.
