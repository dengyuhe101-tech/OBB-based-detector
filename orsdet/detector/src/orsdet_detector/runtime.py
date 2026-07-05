"""Runtime helpers for detector slim OBB + physical shape training."""

from __future__ import annotations

import sys
import types
import os
from pathlib import Path


DETECTOR_DIR = Path(__file__).resolve().parents[2]
SKAO_DIR = DETECTOR_DIR.parent
ROOT_DIR = SKAO_DIR.parent
CIANNA_DIR = ROOT_DIR / "src"
CANDIDATE_DIR = SKAO_DIR / "candidates"
ANGLE_DIR = SKAO_DIR / "angle"
GEOMETRY_DIR = SKAO_DIR / "geometry"
TARGET_SOURCE_DIR = SKAO_DIR / "target_source"
DEFAULT_TARGET_SOURCE = os.environ.get("CIANNA_DETECTOR_TARGET_SOURCE", "target_source").strip().lower() or "target_source"
DEFAULT_GEOMETRY_RUN_DIR = DETECTOR_DIR / "outputs" / "train_geometry"
DEFAULT_TARGET_SOURCE_RUN_DIR = DETECTOR_DIR / "outputs" / "train_target_source"


def normalize_target_source(source: str | None) -> str:
    value = (source or DEFAULT_TARGET_SOURCE).strip().lower()
    if value not in ("geometry", "target_source"):
        raise ValueError("target source must be 'geometry' or 'target_source'.")
    return value


def target_table_path(source: str | None = None) -> Path:
    source = normalize_target_source(source)
    if source == "target_source":
        return TARGET_SOURCE_DIR / "rotated_training_source_table.csv"
    return GEOMETRY_DIR / "rotated_training_source_table.csv"


def default_run_dir_for_target_source(source: str | None = None, slim_mode: str | None = None) -> Path:
    from .decode import normalize_slim_mode

    source = normalize_target_source(source)
    mode = normalize_slim_mode(slim_mode)
    if source == "target_source":
        return DETECTOR_DIR / "outputs" / ("train_target_source_%s" % mode)
    return DETECTOR_DIR / "outputs" / ("train_geometry_%s" % mode)


DEFAULT_RUN_DIR = default_run_dir_for_target_source(DEFAULT_TARGET_SOURCE)


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
        DETECTOR_DIR / "src",
        CANDIDATE_DIR / "src",
        ANGLE_DIR / "src",
        TARGET_SOURCE_DIR / "src",
        GEOMETRY_DIR / "src",
        SKAO_DIR,
    ):
        _prepend_path(path)

    for path in _cianna_backend_dirs():
        _prepend_path(path)


def set_yolo_params_checked(cnn, **kwargs):
    try:
        return cnn.set_yolo_params(**kwargs)
    except (SystemError, TypeError) as exc:
        messages = [str(exc)]
        for attr in ("__cause__", "__context__"):
            nested = getattr(exc, attr, None)
            if nested is not None:
                messages.append(str(nested))
        if "nb_angle" not in "\n".join(messages):
            raise

        cianna_path = getattr(cnn, "__file__", "<unknown>")
        raise RuntimeError(
            "Loaded CIANNA backend does not support ORSDet angle outputs "
            "(`nb_angle`).\n"
            "CIANNA backend: %s\n"
            "Build/install the bundled backend in ORSDet/src, for example:\n"
            "  cd %s\n"
            "  python -m pip install -e src --no-build-isolation\n"
            "Then rerun `python test.py --gpu 0`."
            % (cianna_path, ROOT_DIR)
        ) from exc


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
