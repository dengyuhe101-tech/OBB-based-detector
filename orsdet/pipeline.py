#!/usr/bin/env python3
"""Public train/test orchestration for the ORSDet release."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ORSDET = ROOT / "orsdet"
CIANNA = ROOT / "src"
RESOURCES = ORSDET / "resources"
TRAIN_INTERNAL = ORSDET / "flux_head" / "scripts" / "train_pipeline_internal.py"
EVAL_INTERNAL = ORSDET / "eval" / "scripts" / "evaluate.py"
PROFILE = "flux_head_shared_angle_target_source_obb_phys"
DEFAULT_CHECKPOINT = ROOT / "weights" / "net0_s2700.dat"
RAW_FILES = (
    "sdc1_560MHz_1000h.fits",
    "PrimaryBeam_560MHz.fits",
    "TrainingSet_560MHz.txt",
    "True_560MHz.txt",
)


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.check_call([str(part) for part in cmd], cwd=ROOT, env=env)


def _path_contains(path: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(path)
        return True
    except ValueError:
        return False


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _prune_directory(root: Path, keep_paths: set[Path]) -> None:
    if not root.exists() or not root.is_dir():
        return
    root = root.resolve()
    for child in list(root.iterdir()):
        child_resolved = child.resolve()
        if child_resolved in keep_paths:
            continue
        if any(_path_contains(child_resolved, keep) for keep in keep_paths):
            _prune_directory(child, keep_paths)
            try:
                if child.is_dir() and not any(child.iterdir()) and child.resolve() not in keep_paths:
                    child.rmdir()
            except OSError:
                pass
        else:
            _remove_path(child)


def prune_test_outputs(run_dir: Path, out_dir: Path, epoch: int) -> None:
    outputs_root = (ROOT / "outputs").resolve()
    roots = []
    for root in (run_dir.resolve(), out_dir.resolve()):
        if root == outputs_root or outputs_root not in root.parents:
            raise RuntimeError(
                "Refusing to prune test outputs outside %s: %s\n"
                "Use output directories under ORSDet/outputs or pass --keep-intermediates."
                % (outputs_root, root)
            )
        if root not in roots:
            roots.append(root)

    final_outputs = {
        (out_dir / "catalogs" / ("catalog_sdc1_%04d.txt" % epoch)).resolve(),
        (out_dir / "score_summary.txt").resolve(),
        (out_dir / "scores" / ("score_epoch_%04d.txt" % epoch)).resolve(),
    }
    missing = [path for path in final_outputs if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Refusing to prune test outputs because final files are missing:\n  %s"
            % "\n  ".join(str(path) for path in missing)
        )

    for root in sorted(roots, key=lambda path: len(path.parts), reverse=True):
        _prune_directory(root, final_outputs)
        try:
            if root.exists() and root.is_dir() and not any(root.iterdir()) and not any(
                _path_contains(root, keep) for keep in final_outputs
            ):
                root.rmdir()
        except OSError:
            pass


def raw_data_dir(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    env_value = os.environ.get("SDC1_RAW_DATA_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (ROOT / "external_data" / "560Mhz-1kh").resolve()


def require_raw_data(path: Path) -> None:
    missing = [name for name in RAW_FILES if not (path / name).is_file()]
    if missing:
        raise SystemExit(
            "Missing official SDC1 raw files in %s:\n  %s\n"
            "Set SDC1_RAW_DATA_DIR or pass --raw-data-dir."
            % (path, "\n  ".join(missing))
        )


def runtime_env(raw_dir: Path | None = None, gpu: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    if raw_dir is not None:
        env["SDC1_RAW_DATA_DIR"] = str(raw_dir)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def materialize_checkpoint(checkpoint: Path, run_dir: Path, epoch: int) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    target = run_dir / "net_save" / ("net0_s%04d.dat" % int(epoch))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(checkpoint)
    except OSError:
        shutil.copy2(checkpoint, target)
    return target


def materialize_release_metadata(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = (
        "TrainingSet_perscut.txt",
        "train_cat_norm_lims.txt",
    )
    for name in metadata:
        src = RESOURCES / name
        dst = run_dir / name
        if src.is_file() and not dst.is_file():
            shutil.copy2(src, dst)

    train_norm = run_dir / "train_norm.txt"
    if not train_norm.is_file():
        src = RESOURCES / "train_norm.txt"
        if not src.is_file():
            src = RESOURCES / "train_cat_norm_lims.txt"
        if src.is_file():
            shutil.copy2(src, train_norm)

    run_info = run_dir / "run_info.txt"
    if not run_info.is_file():
        run_info.write_text("slim_mode=shared_angle\nprofile=%s\n" % PROFILE, encoding="utf-8")


def score_catalog(catalog: Path, truth: Path, *, train: bool = False) -> None:
    from ska_sdc import Sdc1Scorer

    catalog = catalog.expanduser().resolve()
    truth = truth.expanduser().resolve()
    if not catalog.is_file():
        raise SystemExit(f"Catalog not found: {catalog}")
    if not truth.is_file():
        raise SystemExit(f"Truth catalog not found: {truth}")

    scorer = Sdc1Scorer.from_txt(str(catalog), str(truth), freq=560, sub_skiprows=0, truth_skiprows=0)
    scorer.run(mode=0, train=train, detail=True)
    score = scorer.score
    purity = float(score.n_match / score.n_det) if int(score.n_det) else float("nan")
    print("score: %.10f" % float(score.value))
    print("n_det: %d" % int(score.n_det))
    print("n_match: %d" % int(score.n_match))
    print("n_bad: %d" % int(score.n_bad))
    print("n_false: %d" % int(score.n_false))
    print("acc: %.10f" % float(score.acc_pc))
    print("purity: %.10f" % purity)


def train_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train ORSDet and package flux head checkpoints.")
    parser.add_argument("--raw-data-dir", default=None)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs" / "train")
    parser.add_argument("--epochs", type=int, default=2700)
    parser.add_argument("--epoch-start", type=int, default=100)
    parser.add_argument("--epoch-interv", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--control-interv", type=int, default=10)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--new-run", action="store_true")
    args = parser.parse_args(argv)

    raw_dir = raw_data_dir(args.raw_data_dir)
    require_raw_data(raw_dir)
    run_dir = args.run_dir.expanduser().resolve()
    cmd = [
        sys.executable,
        TRAIN_INTERNAL,
        "--run-dir",
        run_dir,
        "--epochs",
        args.epochs,
        "--epoch-start",
        args.epoch_start,
        "--epoch-end",
        args.epochs,
        "--save-every",
        args.save_every,
        "--control-interv",
        args.control_interv,
    ]
    if args.epoch_interv is not None:
        cmd.extend(["--epoch-interv", args.epoch_interv])
    if args.gpu is not None:
        cmd.extend(["--gpu", args.gpu])
    if args.new_run:
        cmd.append("--new-run")
    run(cmd, env=runtime_env(raw_dir, args.gpu))


def test_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run ORSDet inference/evaluation or score an existing catalog.")
    parser.add_argument("--raw-data-dir", default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--truth", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs" / "test_run")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "test_eval")
    parser.add_argument("--epoch", type=int, default=2700)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--no-run-pred", action="store_true")
    parser.add_argument("--train-score", action="store_true")
    parser.add_argument("--keep-intermediates", action="store_true")
    args = parser.parse_args(argv)

    raw_dir = raw_data_dir(args.raw_data_dir)
    if args.catalog is not None:
        truth = args.truth or (raw_dir / "True_560MHz.txt")
        score_catalog(args.catalog, truth, train=args.train_score)
        return

    require_raw_data(raw_dir)
    run_dir = args.run_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    materialize_checkpoint(args.checkpoint, run_dir, args.epoch)
    materialize_release_metadata(run_dir)

    cmd = [
        sys.executable,
        EVAL_INTERNAL,
        args.epoch,
        "--profile",
        PROFILE,
        "--run-dir",
        run_dir,
        "--out-dir",
        out_dir,
        "--batch-size",
        args.batch_size,
        "--flux-head-force",
    ]
    if not args.no_run_pred:
        cmd.append("--run-pred")
    if args.gpu is not None:
        cmd.extend(["--device", args.gpu])
    if args.train_score:
        cmd.append("--train-score")
    run(cmd, env=runtime_env(raw_dir, args.gpu))
    if not args.keep_intermediates:
        prune_test_outputs(run_dir, out_dir, args.epoch)
