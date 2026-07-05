#!/usr/bin/env python3
"""Full-image prediction for detector."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from env_guard import drop_user_site

drop_user_site()

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DETECTOR_DIR = SCRIPT_DIR.parent
SKAO_DIR = DETECTOR_DIR.parent
DETECTOR_NB_LAYERS = 18
sys.path.insert(0, str(DETECTOR_DIR / "src"))
sys.path.insert(0, str(SKAO_DIR / "candidates" / "src"))
sys.path.insert(0, str(SKAO_DIR / "angle" / "src"))
sys.path.insert(0, str(SKAO_DIR / "target_source" / "src"))
sys.path.insert(0, str(SKAO_DIR / "geometry" / "src"))
sys.path.insert(0, str(SKAO_DIR))


def i_ar(values):
    return np.asarray(values, dtype="int")


def f_ar(values):
    return np.asarray(values, dtype="float32")


def ensure_postprocess_metadata(run_dir: Path) -> None:
    train_norm = run_dir / "train_norm.txt"
    if train_norm.is_file():
        return
    train_cat_norm = run_dir / "train_cat_norm_lims.txt"
    if train_cat_norm.is_file():
        train_norm.write_text(train_cat_norm.read_text(encoding="utf-8"), encoding="utf-8")


def checkpoint_epochs(run_dir: Path, epoch_start: int | None = None, epoch_end: int | None = None, epoch_interv: int = 1):
    epochs = []
    for path in (run_dir / "net_save").glob("net0_s*.dat"):
        try:
            epoch = int(path.stem.split("_s")[-1])
        except ValueError:
            continue
        if epoch_start is not None and epoch < epoch_start:
            continue
        if epoch_end is not None and epoch > epoch_end:
            continue
        if epoch_start is not None and epoch_interv > 1 and (epoch - epoch_start) % epoch_interv != 0:
            continue
        epochs.append(epoch)
    if not epochs:
        raise FileNotFoundError("No checkpoint found in %s" % (run_dir / "net_save"))
    return sorted(epochs)


def latest_checkpoint_epoch(run_dir: Path) -> int:
    return checkpoint_epochs(run_dir)[-1]


def read_run_slim_mode(run_dir: Path) -> str | None:
    info = run_dir / "run_info.txt"
    if not info.is_file():
        return None
    for line in info.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("slim_mode="):
            return line.split("=", 1)[1].split()[0]
    return None


def training_tile_mask(dg, *, halo_tiles: int = 2) -> np.ndarray:
    """Tiles needed by the training-region flux head post-NMS pass.

    The mask intentionally matches post_detector.py --training-only. Prediction keeps
    the on-disk fwd_res shape full-size, but only these tiles are forwarded.
    """
    mask = np.zeros((dg.nb_area_h, dg.nb_area_w), dtype=bool)
    train_xmin = 16383.0
    train_xmax = 19853.0
    train_ymin = 16730.0
    train_ymax = 19921.0
    margin = float(max(0, halo_tiles) * dg.patch_shift + dg.fwd_image_size)
    for ph in range(dg.nb_area_h):
        tile_ymin = ph * dg.patch_shift - dg.orig_offset
        tile_ymax = tile_ymin + dg.fwd_image_size
        for pw in range(dg.nb_area_w):
            tile_xmin = pw * dg.patch_shift - dg.orig_offset
            tile_xmax = tile_xmin + dg.fwd_image_size
            if (
                tile_xmax >= train_xmin - margin
                and tile_xmin <= train_xmax + margin
                and tile_ymax >= train_ymin - margin
                and tile_ymin <= train_ymax + margin
            ):
                mask[ph, pw] = True
    return mask


def create_tile_pred(dg, tile_indices: np.ndarray) -> np.ndarray:
    """Create normalized prediction patches for a selected flat tile list."""
    pred = np.zeros((int(tile_indices.size), dg.fwd_image_size * dg.fwd_image_size), dtype="float32")
    patch = np.zeros((dg.fwd_image_size, dg.fwd_image_size), dtype="float32")
    full_data_norm = np.clip(dg.full_img, dg.min_pix, dg.max_pix)
    full_data_norm = (full_data_norm - dg.min_pix) / (dg.max_pix - dg.min_pix)
    full_data_norm = np.tanh(3.0 * full_data_norm)

    for out_i, flat_i in enumerate(tile_indices.astype(np.int64, copy=False)):
        p_y = int(flat_i // dg.nb_area_w)
        p_x = int(flat_i % dg.nb_area_w)

        xmin = p_x * dg.patch_shift - dg.orig_offset
        xmax = p_x * dg.patch_shift + dg.fwd_image_size - dg.orig_offset
        ymin = p_y * dg.patch_shift - dg.orig_offset
        ymax = p_y * dg.patch_shift + dg.fwd_image_size - dg.orig_offset

        px_min = 0
        px_max = dg.fwd_image_size
        py_min = 0
        py_max = dg.fwd_image_size

        patch[:, :] = 0.0
        if xmin < 0:
            px_min = -xmin
            xmin = 0
        if ymin < 0:
            py_min = -ymin
            ymin = 0
        if xmax > dg.map_pixel_size:
            px_max = dg.fwd_image_size - (xmax - dg.map_pixel_size)
            xmax = dg.map_pixel_size
        if ymax > dg.map_pixel_size:
            py_max = dg.fwd_image_size - (ymax - dg.map_pixel_size)
            ymax = dg.map_pixel_size

        patch[py_min:py_max, px_min:px_max] = full_data_norm[ymin:ymax, xmin:xmax]
        pred[out_i, :] = patch.flatten("C")
    return pred


def write_roi_fwd_marker(path: Path, *, tile_indices: np.ndarray, dg, layout, halo_tiles: int) -> None:
    """Record that fwd_res contains compact training-region ROI tiles only."""
    channels = dg.nb_box * layout.output_stride
    compact_expected = int(tile_indices.size * channels * dg.yolo_nb_reg * dg.yolo_nb_reg)
    actual_bytes = path.stat().st_size if path.is_file() else -1
    if actual_bytes != compact_expected * 4:
        raise ValueError(
            "%s has %d bytes, expected %d for compact ROI fwd_res."
            % (path, actual_bytes, compact_expected * 4)
        )

    marker_path = path.with_name(path.stem + ".training_roi.npz")
    np.savez(
        marker_path,
        tile_indices=tile_indices.astype(np.int32, copy=False),
        nb_area_h=np.array([dg.nb_area_h], dtype=np.int32),
        nb_area_w=np.array([dg.nb_area_w], dtype=np.int32),
        channels=np.array([channels], dtype=np.int32),
        yolo_nb_reg=np.array([dg.yolo_nb_reg], dtype=np.int32),
        halo_tiles=np.array([halo_tiles], dtype=np.int32),
    )
    summary_path = path.with_name(path.stem + ".training_roi.txt")
    summary_path.write_text(
        "training_roi_only=1\n"
        "tiles=%d\n"
        "total_tiles=%d\n"
        "channels=%d\n"
        "yolo_nb_reg=%d\n"
        "halo_tiles=%d\n"
        "format=compact\n"
        % (
            int(tile_indices.size),
            int(dg.nb_area_h * dg.nb_area_w),
            int(channels),
            int(dg.yolo_nb_reg),
            int(halo_tiles),
        ),
        encoding="utf-8",
    )
    print("Wrote compact ROI fwd_res marker:", marker_path)


def main():
    from orsdet_detector import DEFAULT_RUN_DIR, configure_paths, install_numba_fallback_if_needed
    from orsdet_detector import normalize_slim_mode, detector_layout, detector_target_dim
    from orsdet_detector import set_yolo_params_checked

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("epoch", nargs="?", type=int)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--slim-mode",
        default=None,
        choices=(
            "size",
            "shared_angle",
            "size-angle",
            "size_angle",
            "flux_refine",
            "shared_angle_flux_refine",
            "native_flux_head",
            "flux_calib_gate",
            "shared_angle_flux_calib_gate",
        ),
        help="detector branch. Defaults to run_info.txt slim_mode, then size.",
    )
    parser.add_argument("--latest", action="store_true", help="Only predict the latest checkpoint.")
    parser.add_argument("--epoch-start", type=int)
    parser.add_argument("--epoch-end", type=int)
    parser.add_argument("--epoch-interv", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--comp-meth", default="C_CUDA")
    parser.add_argument("--mixed-precision", default="FP32C_FP32A")
    parser.add_argument("--angle-scale", type=float, default=2.0)
    parser.add_argument("--angle-unit-norm-scale", type=float, default=0.0)
    parser.add_argument("--min-angle-iou", type=float, default=-0.1)
    parser.add_argument(
        "--training-roi-only",
        action="store_true",
        help="Forward only tiles needed by training-region flux head; pad fwd_res back to the standard full-image shape.",
    )
    parser.add_argument("--roi-halo-tiles", type=int, default=2)
    args = parser.parse_args()

    configure_paths()
    install_numba_fallback_if_needed()

    import CIANNA as cnn
    import data_gen as dg

    run_dir = args.run_dir.resolve()
    slim_mode = normalize_slim_mode(args.slim_mode or read_run_slim_mode(run_dir) or "size")
    layout = detector_layout(slim_mode)
    if args.epoch is not None:
        epochs = [args.epoch]
    elif args.latest:
        epochs = [latest_checkpoint_epoch(run_dir)]
    else:
        epochs = checkpoint_epochs(run_dir, args.epoch_start, args.epoch_end, args.epoch_interv)
    if args.epoch is None and len(epochs) > 1:
        for epoch in epochs:
            subprocess.check_call(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "pred_detector.py"),
                    str(epoch),
                    "--run-dir",
                    str(run_dir),
                    "--batch-size",
                    str(args.batch_size),
                    "--comp-meth",
                    args.comp_meth,
                    "--mixed-precision",
                    args.mixed_precision,
                    "--slim-mode",
                    layout.mode,
                    "--angle-scale",
                    str(args.angle_scale),
                    "--angle-unit-norm-scale",
                    str(args.angle_unit_norm_scale),
                    "--min-angle-iou",
                    str(args.min_angle_iou),
                ]
                + (["--training-roi-only", "--roi-halo-tiles", str(args.roi_halo_tiles)] if args.training_roi_only else [])
            )
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "fwd_res").mkdir(exist_ok=True)
    os.chdir(run_dir)

    dg.dataset_perscut(dg.TRAINING_CATALOG_PATH, "TrainingSet_perscut.txt", 18)
    cnn.init(
        in_dim=i_ar([dg.fwd_image_size, dg.fwd_image_size]),
        in_nb_ch=1,
        out_dim=detector_target_dim(dg.max_nb_obj_per_image, layout.mode),
        bias=0.1,
        b_size=args.batch_size,
        comp_meth=args.comp_meth,
        dynamic_load=1,
        mixed_precision=args.mixed_precision,
        adv_size=30,
        inference_only=1,
    )

    dg.init_data_gen()
    ensure_postprocess_metadata(run_dir)
    roi_tile_indices = None
    if args.training_roi_only:
        roi_mask = training_tile_mask(dg, halo_tiles=args.roi_halo_tiles)
        roi_tile_indices = np.flatnonzero(roi_mask.reshape(-1))
        print(
            "Training-region ROI forward: %d / %d tiles"
            % (int(roi_tile_indices.size), int(dg.nb_area_h * dg.nb_area_w))
        )
        input_data = create_tile_pred(dg, roi_tile_indices)
    else:
        input_data = dg.create_full_pred()
    targets = np.zeros((input_data.shape[0], detector_target_dim(dg.max_nb_obj_per_image, layout.mode)), dtype="float32")
    cnn.create_dataset("TEST", input_data.shape[0], input_data[:, :], targets[:, :])

    prior_w = f_ar([6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 12.0, 9.0, 24.0])
    prior_h = f_ar([6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 9.0, 12.0, 24.0])
    prior_size = np.vstack((prior_w, prior_h))
    prior_noobj_prob = f_ar([0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.01, 0.01, 0.01])
    error_scales = cnn.set_error_scales(position=36.0, size=0.2, probability=0.5, objectness=2.0, parameters=5.0)
    param_ind_values = [2.0, 2.0, 2.0] + [1.0] * max(0, layout.nb_param - 3)
    param_ind_scales = f_ar(param_ind_values)
    iou_limits = cnn.set_IoU_limits(
        good_IoU_lim=0.5,
        low_IoU_best_box_assoc=-0.1,
        min_prob_IoU_lim=-0.3,
        min_obj_IoU_lim=-0.3,
        min_param_IoU_lim=-0.1,
    )
    fit_parts = cnn.set_fit_parts(position=1, size=1, probability=1, objectness=1, parameters=1)
    slopes_and_maxes = cnn.set_slopes_and_maxes(
        position=cnn.set_sm_single(slope=0.5, fmax=6.0, fmin=-6.0),
        size=cnn.set_sm_single(slope=0.5, fmax=1.2, fmin=-1.2),
        probability=cnn.set_sm_single(slope=0.2, fmax=6.0, fmin=-6.0),
        objectness=cnn.set_sm_single(slope=0.5, fmax=6.0, fmin=-6.0),
        parameters=cnn.set_sm_single(slope=0.5, fmax=1.5, fmin=-0.2),
    )

    set_yolo_params_checked(
        cnn,
        nb_box=dg.nb_box,
        nb_class=dg.nb_class,
        nb_param=layout.nb_param,
        max_nb_obj_per_image=dg.max_nb_obj_per_image,
        prior_size=prior_size,
        prior_noobj_prob=prior_noobj_prob,
        IoU_type="RotIoU",
        prior_dist_type="OFFSET",
        error_scales=error_scales,
        param_ind_scales=param_ind_scales,
        slopes_and_maxes=slopes_and_maxes,
        IoU_limits=iou_limits,
        fit_parts=fit_parts,
        error_type="natural",
        no_override=1,
        raw_output=0,
        nb_angle=layout.nb_angle,
        angle_scale=args.angle_scale,
        angle_unit_norm_scale=args.angle_unit_norm_scale,
        angle_slope=1.0,
        angle_fmax=1.2,
        angle_fmin=-1.2,
        min_angle_IoU_lim=args.min_angle_iou,
        fit_angle=1,
    )

    for epoch in epochs:
        model_path = run_dir / "net_save" / ("net0_s%04d.dat" % epoch)
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        cnn.load(str(model_path), epoch, nb_layers=DETECTOR_NB_LAYERS, bin=1)
        cnn.forward(no_error=1, saving=2, repeat=1, drop_mode="AVG_MODEL")
        fwd_path = run_dir / "fwd_res" / ("net0_%04d.dat" % epoch)
        if roi_tile_indices is not None:
            write_roi_fwd_marker(
                fwd_path,
                tile_indices=roi_tile_indices,
                dg=dg,
                layout=layout,
                halo_tiles=args.roi_halo_tiles,
            )
        print(fwd_path)


if __name__ == "__main__":
    main()
