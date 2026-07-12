#!/usr/bin/env bash
# Build the V21 Qwen3 NER-only image on the 2-vCPU/4-GB judge twin.
#
# Required assets under ASSET_DIR:
#   qwen3-4b.gguf, config/models.json, wheels/*.whl
# The image is built locally first. Publishing and OCI-index wrapping happen
# only after the resource-limited replica gate passes.
set -euo pipefail

ASSET_DIR="${ASSET_DIR:-/root/v21}"
BASE_IMAGE="${BASE_IMAGE:-docker.io/kgotsomsiza/bastion:track1-v20}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-bastion:v21-local}"
V21_COMMIT="${V21_COMMIT:-53e30d9}"
V21_REVISION="${V21_REVISION:-53e30d9ef48548cb5b7a04394865feebbe5b1ce8}"
MODEL="$ASSET_DIR/qwen3-4b.gguf"

test -f "$MODEL"
test -f "$ASSET_DIR/config/models.json"
test -n "$(find "$ASSET_DIR/wheels" -maxdepth 1 -name 'llama_cpp_python*.whl' -print -quit)"

MODEL_SHA256="$(sha256sum "$MODEL")"
MODEL_SHA256="${MODEL_SHA256%% *}"
BUILT_AT="$(date -Iseconds)"

cat > "$ASSET_DIR/Dockerfile.v21" <<EOF
FROM $BASE_IMAGE

ENV LOCAL_MODEL_PATH=/app/models/model.gguf
ENV FRUGAL_WORKERS=2
ENV PYTHONPATH=/app

COPY wheels /wheels
RUN python -m pip install --no-cache-dir --no-index --find-links=/wheels llama-cpp-python \
 && rm -rf /wheels

COPY qwen3-4b.gguf /app/models/model.gguf
COPY config/models.json /app/config/models.json
EOF

cat > "$ASSET_DIR/.dockerignore" <<'EOF'
*
!Dockerfile.v21
!qwen3-4b.gguf
!wheels/
!wheels/**
!config/
!config/models.json
EOF

docker build --pull \
  --label "bastion.commit=$V21_COMMIT" \
  --label "org.opencontainers.image.revision=$V21_REVISION" \
  --label "bastion.built=$BUILT_AT" \
  --label "bastion.model.name=Qwen3-4B-Instruct-2507-Q4_K_M" \
  --label "bastion.model.sha256=$MODEL_SHA256" \
  -f "$ASSET_DIR/Dockerfile.v21" \
  -t "$OUTPUT_IMAGE" \
  "$ASSET_DIR"

docker image inspect "$OUTPUT_IMAGE" --format \
  'IMAGE={{.RepoTags}} ARCH={{.Architecture}} OS={{.Os}} SIZE={{.Size}} LABELS={{json .Config.Labels}}'
docker run --rm --entrypoint python "$OUTPUT_IMAGE" -c \
  'import llama_cpp; print("llama_cpp", llama_cpp.__version__)'
echo "MODEL_SHA256=$MODEL_SHA256"
echo "V21_LOCAL_BUILD_COMPLETE=$OUTPUT_IMAGE"
