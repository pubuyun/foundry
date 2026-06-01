#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif [[ -x "$ROOT_DIR/../.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/../.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

"$PYTHON_BIN" - "$ROOT_DIR" <<'PY'
from __future__ import annotations

import importlib.util
import pathlib
import py_compile
import sys

root = pathlib.Path(sys.argv[1]).resolve()

OLD = '''# Global flag for cuEquivariance availability
SHOULD_USE_CUEQUIVARIANCE = False

try:
    if torch.cuda.is_available():
        if _env.bool("DISABLE_CUEQUIVARIANCE", default=False):
            logger.info("cuEquivariance usage disabled via DISABLE_CUEQUIVARIANCE")
        else:
            import cuequivariance_torch as cuet  # noqa: I001, F401

            SHOULD_USE_CUEQUIVARIANCE = True
            os.environ["CUEQ_DISABLE_AOT_TUNING"] = _env.str(
                "CUEQ_DISABLE_AOT_TUNING", default="1"
            )
            os.environ["CUEQ_DEFAULT_CONFIG"] = _env.str(
                "CUEQ_DEFAULT_CONFIG", default="1"
            )
            logger.info("cuEquivariance is available and will be used.")
'''

NEW = '''# Global flag for cuEquivariance availability
SHOULD_USE_CUEQUIVARIANCE = False


def _cuda_supports_cuequivariance_bfloat16() -> bool:
    """Return whether the active CUDA device can compile cuEquivariance BF16 kernels."""
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_bf16_supported()
    except AttributeError:
        major, _ = torch.cuda.get_device_capability()
        return major >= 8


try:
    if torch.cuda.is_available():
        if _env.bool("DISABLE_CUEQUIVARIANCE", default=False):
            logger.info("cuEquivariance usage disabled via DISABLE_CUEQUIVARIANCE")
        elif not _cuda_supports_cuequivariance_bfloat16():
            logger.info(
                "cuEquivariance disabled: CUDA device lacks native bfloat16 support"
            )
        else:
            import cuequivariance_torch as cuet  # noqa: I001, F401

            SHOULD_USE_CUEQUIVARIANCE = True
            os.environ["CUEQ_DISABLE_AOT_TUNING"] = _env.str(
                "CUEQ_DISABLE_AOT_TUNING", default="1"
            )
            os.environ["CUEQ_DEFAULT_CONFIG"] = _env.str(
                "CUEQ_DEFAULT_CONFIG", default="1"
            )
            logger.info("cuEquivariance is available and will be used.")
'''

MARKER = "def _cuda_supports_cuequivariance_bfloat16()"


def patch_file(path: pathlib.Path) -> str:
    if not path.exists():
        return "missing"

    text = path.read_text()
    if MARKER in text:
        status = "already patched"
    elif OLD in text:
        path.write_text(text.replace(OLD, NEW))
        status = "patched"
    else:
        raise RuntimeError(
            f"Could not find expected cuEquivariance block in {path}. "
            "Patch this file manually or reinstall from the patched source tree."
        )

    py_compile.compile(str(path), doraise=True)
    cache_dir = path.parent / "__pycache__"
    if cache_dir.exists():
        for pyc in cache_dir.glob("__init__*.pyc"):
            pyc.unlink()
    return status


targets: list[pathlib.Path] = []

spec = importlib.util.find_spec("foundry")
active_target: pathlib.Path | None = None
if spec and spec.origin:
    active_target = pathlib.Path(spec.origin).resolve()

for candidate in [
    root / "src" / "foundry" / "__init__.py",
    root / "foundry" / "src" / "foundry" / "__init__.py",
    root.parent / "foundry" / "src" / "foundry" / "__init__.py",
]:
    targets.append(candidate.resolve())
if active_target is not None:
    targets.append(active_target)

seen: set[pathlib.Path] = set()
patched_any = False
active_patched = active_target is None
errors: list[str] = []
for target in targets:
    if target in seen:
        continue
    seen.add(target)
    try:
        status = patch_file(target)
    except Exception as exc:
        errors.append(f"{target}: {exc}")
        continue
    if status != "missing":
        patched_any = True
        if target == active_target:
            active_patched = True
        print(f"{status}: {target}")

if not patched_any:
    raise SystemExit("No foundry/__init__.py target was found to patch.")
if errors:
    print("Patch warnings/errors:")
    for error in errors:
        print(f"  {error}")
if not active_patched:
    raise SystemExit(
        "The foundry package imported by this Python could not be patched. "
        "Re-run with write permission for that environment, or set PYTHON to the "
        "target environment's interpreter."
    )

print("RF3 Tesla T4 BF16/cuEquivariance patch installed.")
PY
