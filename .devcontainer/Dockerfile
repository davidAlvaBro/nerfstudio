FROM ghcr.io/nerfstudio-project/nerfstudio:latest
RUN apt-get update && \
     apt-get install -y --no-install-recommends git python3-pip  && \
     python3 -m pip install --no-cache-dir -U ultralytics && \
     rm -rf /var/lib/apt/lists/*