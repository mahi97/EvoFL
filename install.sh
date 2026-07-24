#!/usr/bin/env bash
# install.sh — Modern EvoFL setup via uv + pyproject.toml
#
# Usage:
#   ./install.sh                 # CPU JAX (default)
#   ./install.sh --cuda12        # JAX CUDA 12 wheels
#   ./install.sh --cuda12-local  # local CUDA 12 toolkit
#   ./install.sh --tpu           # TPU (post-sync jax[tpu] install)
#   ./install.sh --python 3.12   # Python version for the venv
#   ./install.sh --no-verify     # skip import / smoke checks
#
# Dependencies live in pyproject.toml only (no requirements.txt).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

BACKEND="cpu"
PYTHON_VERSION="${UV_PYTHON:-3.12}"
VERIFY=1

usage() {
  sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu)           BACKEND="cpu"; shift ;;
    --cuda12)        BACKEND="cuda12"; shift ;;
    --cuda12-local)  BACKEND="cuda12-local"; shift ;;
    --tpu)           BACKEND="tpu"; shift ;;
    --python)        PYTHON_VERSION="$2"; shift 2 ;;
    --no-verify)     VERIFY=0; shift ;;
    -h|--help)       usage 0 ;;
    *)               echo "Unknown option: $1" >&2; usage 1 ;;
  esac
done

log()  { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m✓\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m✗\033[0m %s\n' "$*" >&2; exit 1; }

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    ok "uv $(uv --version | awk '{print $2}')"
    return
  fi
  log "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
  command -v uv >/dev/null 2>&1 || die "uv install failed; add ~/.local/bin to PATH"
  ok "uv installed"
}

sync_project() {
  log "Syncing project with uv (Python ${PYTHON_VERSION}, backend=${BACKEND})"
  export UV_PROJECT_ENVIRONMENT="${ROOT}/.venv"

  # Drop stale lock so extras re-resolve cleanly when backend changes
  rm -f "${ROOT}/uv.lock"

  local -a sync_args=(--python "${PYTHON_VERSION}")
  case "${BACKEND}" in
    cpu)
      uv sync "${sync_args[@]}"
      ;;
    cuda12)
      uv sync "${sync_args[@]}" --extra cuda12
      ;;
    cuda12-local)
      uv sync "${sync_args[@]}" --extra cuda12-local
      ;;
    tpu)
      # TPU wheels are not always lock-friendly; sync core then overlay jax[tpu]
      uv sync "${sync_args[@]}"
      log "Overlaying JAX TPU wheels"
      uv pip install --python "${ROOT}/.venv/bin/python" "jax[tpu]>=0.4.35,<0.5" \
        -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
      ;;
    *)
      die "Unknown backend: ${BACKEND}"
      ;;
  esac
  ok "uv sync complete"
}

verify_install() {
  [[ "${VERIFY}" -eq 1 ]] || { warn "Skipping verification"; return; }
  log "Verifying installation"
  uv run --python "${PYTHON_VERSION}" python - <<'PY'
import importlib
import sys
import traceback

core = [
    "jax", "jaxlib", "jax.numpy",
    "flax", "optax", "chex", "numpy",
    "wandb", "yaml", "dotmap", "tqdm",
    "tensorflow_datasets",
    "gymnax", "tensorflow_probability",
    "matplotlib", "brax", "torch", "torchvision",
]

failed = []
for name in core:
    try:
        importlib.import_module(name)
        print(f"  OK  {name}")
    except Exception as e:
        failed.append((name, e))
        print(f"  FAIL {name}: {e}")

import jax
print(f"\nJAX {jax.__version__}")
print(f"Devices: {jax.devices()}")
x = jax.numpy.ones((4, 4))
y = jax.jit(lambda a: a @ a.T)(x)
assert y.shape == (4, 4)
print("  OK  jax JIT matmul smoke test")

for mod in ("evosax", "utils", "utils.helpers", "utils.evo", "backprop", "backprop.sl"):
    try:
        importlib.import_module(mod)
        print(f"  OK  {mod}")
    except Exception as e:
        failed.append((mod, e))
        print(f"  FAIL {mod}: {e}")

from evosax import NetworkMapper, Strategies, ParameterReshaper, FitnessShaper
assert "CNN" in NetworkMapper
assert "OpenES" in Strategies
print("  OK  evosax NetworkMapper / Strategies")

from utils.helpers import load_config
cfg = load_config("configs/Vision-FMNIST/evofed.yaml")
assert cfg is not None
print("  OK  load_config(configs/Vision-FMNIST/evofed.yaml)")

from utils.evo import get_network_and_pholder
print("  OK  utils.evo helpers")

import args  # noqa: F401
print("  OK  args module")

net, pholder = get_network_and_pholder(
    "FMNIST", type("A", (), {"opt_name": "adam", "strategy": "OpenES"})()
)
params = net.init(jax.random.PRNGKey(0), pholder, jax.random.PRNGKey(1))
assert "params" in params
print("  OK  CNN init on FMNIST placeholder")

if failed:
    print("\nFailures:", file=sys.stderr)
    for n, e in failed:
        print(f"  - {n}: {e}", file=sys.stderr)
    sys.exit(1)

print("\n✓ All checks passed — EvoFL environment is ready.")
PY
  ok "verification passed"
}

main() {
  log "EvoFL modern install (uv + pyproject.toml)"
  ensure_uv
  sync_project
  verify_install

  cat <<MSG

────────────────────────────────────────────────────────────
EvoFL is ready.

Activate:
  source .venv/bin/activate

Or run without activating:
  uv run python evofed.py --config configs/Vision-FMNIST/evofed.yaml
  uv run python fedavg.py --config configs/Vision-FMNIST/fedavg.yaml

GPU later:
  ./install.sh --cuda12
────────────────────────────────────────────────────────────
MSG
}

main
