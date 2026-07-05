"""Runtime helpers for ORSDet evaluation."""

from __future__ import annotations

import os
from pathlib import Path
import site
import sys
import types


EVAL_DIR = Path(__file__).resolve().parents[2]
ORSDET_DIR = EVAL_DIR.parent
ROOT_DIR = ORSDET_DIR.parent
CIANNA_DIR = ROOT_DIR / "src"

ORSDET_PROFILE = "flux_head_shared_angle_target_source_obb_phys"
DEFAULT_ORSDET_RUN_DIR = ORSDET_DIR / "flux_head" / "outputs" / "orsdet"
DEFAULT_ORSDET_OUT_DIR = EVAL_DIR / "outputs" / "orsdet_eval"
DEFAULT_CIANNA_RUN_DIR = ORSDET_DIR / "run_hbb"
DEFAULT_CIANNA_OUT_DIR = EVAL_DIR / "outputs" / "hbb_eval"

OBB_PROFILES = (ORSDET_PROFILE,)
ALL_PROFILES = OBB_PROFILES + ("cianna_hbb",)

DEFAULT_RUN_DIR = DEFAULT_ORSDET_RUN_DIR
DEFAULT_OUT_DIR = DEFAULT_ORSDET_OUT_DIR


def default_run_dir(profile: str) -> Path:
    if profile == ORSDET_PROFILE:
        return DEFAULT_ORSDET_RUN_DIR
    if profile == "cianna_hbb":
        return DEFAULT_CIANNA_RUN_DIR
    raise ValueError("Unsupported profile: %s" % profile)


def default_out_dir(profile: str) -> Path:
    if profile == ORSDET_PROFILE:
        return DEFAULT_ORSDET_OUT_DIR
    if profile == "cianna_hbb":
        return DEFAULT_CIANNA_OUT_DIR
    raise ValueError("Unsupported profile: %s" % profile)


def drop_user_site() -> None:
    os.environ["PYTHONNOUSERSITE"] = "1"
    user_paths = []
    try:
        paths = site.getusersitepackages()
    except Exception:
        paths = []
    if isinstance(paths, str):
        paths = [paths]
    user_paths.extend(paths)
    if os.environ.get("PYTHONUSERBASE"):
        user_paths.append(os.environ["PYTHONUSERBASE"])

    resolved = []
    for path in user_paths:
        if not path:
            continue
        try:
            resolved.append(Path(path).expanduser().resolve())
        except OSError:
            continue
    if not resolved:
        return

    filtered = []
    for entry in sys.path:
        if not entry:
            filtered.append(entry)
            continue
        try:
            entry_path = Path(entry).expanduser().resolve()
        except OSError:
            filtered.append(entry)
            continue
        if any(entry_path == user_path or user_path in entry_path.parents for user_path in resolved):
            continue
        filtered.append(entry)
    sys.path[:] = filtered


def set_cuda_device(device: int | None) -> None:
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)


def _prepend_path(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def _cianna_backend_dirs() -> list[Path]:
    candidates = []
    preferred = CIANNA_DIR / "build" / "lib.cianna4090-cuda" / "CIANNA.so"
    if preferred.is_file():
        candidates.append(preferred.parent)

    build_libs = sorted(
        CIANNA_DIR.glob("build/lib.*/CIANNA*.so"),
        key=lambda path: path.stat().st_mtime,
    )
    candidates.extend(path.parent for path in build_libs)

    source_libs = sorted(
        CIANNA_DIR.glob("CIANNA*.so"),
        key=lambda path: path.stat().st_mtime,
    )
    candidates.extend(path.parent for path in source_libs)

    unique = []
    for path in candidates:
        if path not in unique:
            unique.append(path)
    return unique


def configure_paths() -> None:
    for path in (
        EVAL_DIR / "src",
        ORSDET_DIR,
        ORSDET_DIR / "geometry" / "src",
        ORSDET_DIR / "target_source" / "src",
        ORSDET_DIR / "angle" / "src",
        ORSDET_DIR / "candidates" / "src",
        ORSDET_DIR / "nms" / "src",
        ORSDET_DIR / "detector" / "src",
    ):
        _prepend_path(path)

    for path in _cianna_backend_dirs():
        _prepend_path(path)


def install_numba_fallback_if_needed() -> None:
    try:
        import numba  # noqa: F401
        return
    except ImportError:
        pass

    def jit(*jit_args, **jit_kwargs):
        if jit_args and callable(jit_args[0]) and len(jit_args) == 1 and not jit_kwargs:
            return jit_args[0]

        def decorator(func):
            return func

        return decorator

    stub = types.ModuleType("numba")
    stub.jit = jit
    sys.modules["numba"] = stub


def ensure_dirs(run_dir: Path, out_dir: Path | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "net_save").mkdir(exist_ok=True)
    (run_dir / "fwd_res").mkdir(exist_ok=True)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)


def parse_epoch_from_name(path: Path, prefix: str) -> int | None:
    try:
        return int(path.stem.split(prefix)[-1])
    except (IndexError, ValueError):
        return None


def available_epochs(
    run_dir: Path,
    subdir: str,
    prefix: str,
    epoch_start: int | None = None,
    epoch_end: int | None = None,
    epoch_interv: int = 1,
) -> list[int]:
    epochs = []
    for path in (run_dir / subdir).glob("%s*.dat" % prefix):
        epoch = parse_epoch_from_name(path, prefix)
        if epoch is None:
            continue
        if epoch_start is not None and epoch < epoch_start:
            continue
        if epoch_end is not None and epoch > epoch_end:
            continue
        if epoch_start is not None and epoch_interv > 1 and (epoch - epoch_start) % epoch_interv != 0:
            continue
        epochs.append(epoch)
    return sorted(set(epochs))


def detector_mode_for_profile(profile: str) -> str | None:
    return "shared_angle" if profile == ORSDET_PROFILE else None


def is_flux_head_profile(profile: str) -> bool:
    return profile == ORSDET_PROFILE


def expected_fwd_bytes(aux_module, profile: str = "cianna_hbb") -> int:
    if profile not in ALL_PROFILES:
        raise ValueError("Unsupported profile for expected_fwd_bytes: %s" % profile)
    nb_aux = aux_module.nb_param
    if profile == ORSDET_PROFILE:
        try:
            from orsdet_detector import detector_layout

            nb_aux = detector_layout("shared_angle").total_aux
        except Exception:
            nb_aux = 5
    n_float = (
        aux_module.nb_area_h
        * aux_module.nb_area_w
        * aux_module.nb_box
        * (8 + nb_aux)
        * aux_module.yolo_nb_reg
        * aux_module.yolo_nb_reg
    )
    return int(n_float * 4)


def fwd_file_complete(path: Path, expected_bytes: int) -> bool:
    return path.is_file() and path.stat().st_size == expected_bytes
