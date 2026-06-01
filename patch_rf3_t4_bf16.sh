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

FOUNDRY_OLD_BLOCK = '''# Global flag for cuEquivariance availability
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

FOUNDRY_V1_FUNC = '''def _cuda_supports_cuequivariance_bfloat16() -> bool:
    """Return whether the active CUDA device can compile cuEquivariance BF16 kernels."""
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_bf16_supported()
    except AttributeError:
        major, _ = torch.cuda.get_device_capability()
        return major >= 8
'''

FOUNDRY_NEW_BLOCK = '''# Global flag for cuEquivariance availability
SHOULD_USE_CUEQUIVARIANCE = False


def _cuda_supports_cuequivariance_bfloat16() -> bool:
    """Return whether the active CUDA device can compile cuEquivariance BF16 kernels."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        return False
    try:
        return torch.cuda.is_bf16_supported()
    except AttributeError:
        return True


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

FOUNDRY_V2_FUNC = '''def _cuda_supports_cuequivariance_bfloat16() -> bool:
    """Return whether the active CUDA device can compile cuEquivariance BF16 kernels."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        return False
    try:
        return torch.cuda.is_bf16_supported()
    except AttributeError:
        return True
'''

ATTENTION_IMPORT_OLD = '''if SHOULD_USE_CUEQUIVARIANCE:
    import cuequivariance_torch as cuet


class TriangleAttention(nn.Module):
'''

ATTENTION_IMPORT_NEW = '''if SHOULD_USE_CUEQUIVARIANCE:
    import cuequivariance_torch as cuet


def _cuda_supports_cuequivariance_bfloat16(device: torch.device) -> bool:
    """Return whether this CUDA device can compile cuEquivariance BF16 kernels."""
    if device.type != "cuda":
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 8


def _should_use_cuequivariance_for_tensor(tensor: torch.Tensor) -> bool:
    return SHOULD_USE_CUEQUIVARIANCE and _cuda_supports_cuequivariance_bfloat16(
        tensor.device
    )


class TriangleAttention(nn.Module):
'''


def clear_pycache(path: pathlib.Path) -> None:
    cache_dir = path.parent / "__pycache__"
    if cache_dir.exists():
        for pyc in cache_dir.glob(f"{path.stem}*.pyc"):
            pyc.unlink()


def patch_foundry(path: pathlib.Path) -> str:
    if not path.exists():
        return "missing"
    text = path.read_text()
    if FOUNDRY_V2_FUNC in text:
        status = "already patched"
    elif FOUNDRY_V1_FUNC in text:
        path.write_text(text.replace(FOUNDRY_V1_FUNC, FOUNDRY_V2_FUNC))
        status = "updated"
    elif FOUNDRY_OLD_BLOCK in text:
        path.write_text(text.replace(FOUNDRY_OLD_BLOCK, FOUNDRY_NEW_BLOCK))
        status = "patched"
    else:
        raise RuntimeError("expected cuEquivariance block not found")
    py_compile.compile(str(path), doraise=True)
    clear_pycache(path)
    return status


def patch_attention(path: pathlib.Path) -> str:
    if not path.exists():
        return "missing"
    text = path.read_text()
    original = text
    if "_should_use_cuequivariance_for_tensor" not in text:
        if ATTENTION_IMPORT_OLD not in text:
            raise RuntimeError("expected cuEquivariance import block not found")
        text = text.replace(ATTENTION_IMPORT_OLD, ATTENTION_IMPORT_NEW)
    text = text.replace(
        "if self.use_cuequivariance and SHOULD_USE_CUEQUIVARIANCE:",
        "if self.use_cuequivariance and _should_use_cuequivariance_for_tensor(pair):",
    )
    if text == original:
        status = "already patched"
    else:
        path.write_text(text)
        status = "patched"
    py_compile.compile(str(path), doraise=True)
    clear_pycache(path)
    return status


def spec_origin(module: str) -> pathlib.Path | None:
    spec = importlib.util.find_spec(module)
    if spec and spec.origin:
        return pathlib.Path(spec.origin).resolve()
    return None


targets: list[tuple[str, pathlib.Path, callable[[pathlib.Path], str], bool]] = []

active_foundry = spec_origin("foundry")
active_rf3 = spec_origin("rf3")

for candidate in [
    root / "src" / "foundry" / "__init__.py",
    root / "foundry" / "src" / "foundry" / "__init__.py",
    root.parent / "foundry" / "src" / "foundry" / "__init__.py",
]:
    targets.append(("foundry", candidate.resolve(), patch_foundry, False))
if active_foundry is not None:
    targets.append(("foundry", active_foundry, patch_foundry, True))

for candidate in [
    root / "models" / "rf3" / "src" / "rf3" / "model" / "layers" / "attention.py",
    root / "foundry" / "models" / "rf3" / "src" / "rf3" / "model" / "layers" / "attention.py",
    root.parent / "foundry" / "models" / "rf3" / "src" / "rf3" / "model" / "layers" / "attention.py",
]:
    targets.append(("rf3 attention", candidate.resolve(), patch_attention, False))
if active_rf3 is not None:
    targets.append(
        (
            "rf3 attention",
            active_rf3.parent / "model" / "layers" / "attention.py",
            patch_attention,
            True,
        )
    )

seen: set[pathlib.Path] = set()
patched_any = False
active_required = {path for _, path, _, active in targets if active}
active_done: set[pathlib.Path] = set()
errors: list[str] = []

for label, target, patcher, active in targets:
    if target in seen:
        if active:
            active_done.add(target)
        continue
    seen.add(target)
    try:
        status = patcher(target)
    except Exception as exc:
        errors.append(f"{label}: {target}: {exc}")
        continue
    if status != "missing":
        patched_any = True
        if active:
            active_done.add(target)
        print(f"{status}: {label}: {target}")

if not patched_any:
    raise SystemExit("No patch targets were found.")
if errors:
    print("Patch warnings/errors:")
    for error in errors:
        print(f"  {error}")
missing_active = active_required - active_done
if missing_active:
    missing = ", ".join(str(path) for path in sorted(missing_active))
    raise SystemExit(
        "The package imported by this Python could not be fully patched: "
        f"{missing}. Re-run with write permission for that environment, or set "
        "PYTHON to the target environment's interpreter."
    )

print("RF3 Tesla T4 BF16/cuEquivariance patch installed.")
PY
