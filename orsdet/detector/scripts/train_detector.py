#!/usr/bin/env python3
"""Train detector with one native OBB size branch and physical shape heads."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from threading import Thread

from env_guard import drop_user_site

drop_user_site()

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DETECTOR_DIR = SCRIPT_DIR.parent
SKAO_DIR = DETECTOR_DIR.parent
sys.path.insert(0, str(DETECTOR_DIR / "src"))
sys.path.insert(0, str(SKAO_DIR / "candidates" / "src"))
sys.path.insert(0, str(SKAO_DIR / "angle" / "src"))
sys.path.insert(0, str(SKAO_DIR / "target_source" / "src"))
sys.path.insert(0, str(SKAO_DIR / "geometry" / "src"))
sys.path.insert(0, str(SKAO_DIR))


def next_run_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    used = []
    for path in base_dir.glob("run*"):
        if path.is_dir() and path.name[3:].isdigit():
            used.append(int(path.name[3:]))
    return base_dir / ("run%d" % ((max(used) + 1) if used else 1))


def i_ar(values):
    return np.asarray(values, dtype="int")


def f_ar(values):
    return np.asarray(values, dtype="float32")


def write_error_summary(run_dir: Path):
    path = run_dir / "error.txt"
    if not path.is_file():
        return
    data = np.genfromtxt(path, names=True, dtype=np.float64)
    if data.shape == ():
        data = data.reshape(1)
    angle = np.asarray(data["angle_loss"], dtype=np.float64)
    total = np.asarray(data["total_loss"], dtype=np.float64)
    lines = [
        "rows: %d" % angle.size,
        "total_loss_initial: %.8g" % total[0],
        "total_loss_final: %.8g" % total[-1],
        "angle_loss_initial: %.8g" % angle[0],
        "angle_loss_final: %.8g" % angle[-1],
        "angle_loss_delta: %.8g" % (angle[-1] - angle[0]),
        "angle_loss_min: %.8g" % np.min(angle),
    ]
    (run_dir / "error_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    from orsdet_detector import (
        DEFAULT_TARGET_SOURCE,
        HardGroupSamplerConfig,
        DetectorDataBuilder,
        normalize_slim_mode,
        detector_layout,
    )
    from orsdet_detector import (
        configure_paths,
        default_run_dir_for_target_source,
        install_numba_fallback_if_needed,
        normalize_target_source,
        set_yolo_params_checked,
        target_table_path,
        detector_target_dim,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--new-run", action="store_true")
    parser.add_argument("--runs-root", type=Path, default=DETECTOR_DIR / "outputs" / "runs")
    parser.add_argument(
        "--slim-mode",
        default="size",
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
        help="detector branch: size removes duplicate OBB W/H; shared_angle also shares one angle pair.",
    )
    parser.add_argument(
        "--target-source",
        choices=("target_source", "geometry"),
        default=DEFAULT_TARGET_SOURCE,
        help="OBB target table source. Default is target_source; use geometry for direct geometry-derived targets.",
    )
    parser.add_argument(
        "--target-table",
        type=Path,
        default=None,
        help="Explicit rotated_training_source_table.csv path. Overrides --target-source.",
    )
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("CIANNA_DETECTOR_EPOCHS", "10")))
    parser.add_argument("--control-interv", type=int, default=int(os.environ.get("CIANNA_CONTROL_INTERV", "10")))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--comp-meth", default="C_CUDA")
    parser.add_argument("--mixed-precision", default="FP32C_FP32A")
    parser.add_argument("--learning-rate", type=float, default=0.00015)
    parser.add_argument("--end-lr-prop", type=float, default=0.02)
    parser.add_argument("--angle-scale", type=float, default=2.0)
    parser.add_argument("--angle-unit-norm-scale", type=float, default=0.0)
    parser.add_argument("--min-angle-iou", type=float, default=-0.01)
    parser.add_argument("--fit-position", type=int, default=1)
    parser.add_argument("--fit-size", type=int, default=1)
    parser.add_argument("--fit-probability", type=int, default=1)
    parser.add_argument("--fit-objectness", type=int, default=1)
    parser.add_argument("--fit-class", type=int, default=1)
    parser.add_argument("--fit-angle", type=int, default=1)
    parser.add_argument(
        "--small-elongated-boost-factor",
        type=float,
        default=1.4,
        help="Default is 1.4. Set to 1.0, or pass --no-small-elongated-boost, to disable this target-weight boost.",
    )
    parser.add_argument("--small-elongated-sqrt-area", type=float, default=8.0)
    parser.add_argument("--small-elongated-aspect", type=float, default=1.5)
    parser.add_argument(
        "--no-small-elongated-boost",
        action="store_true",
        help="Disable the default small-elongated angle-weight boost and use factor=1.0.",
    )
    parser.add_argument(
        "--hard-group-sampling",
        action="store_true",
        help="Enable Libra-R-CNN-style balanced source-centered sampling for faint/bright/large/crowded groups.",
    )
    parser.add_argument(
        "--hard-group-fraction",
        type=float,
        default=0.35,
        help="Fraction of non-noise training patches drawn from hard flux/size/crowding groups.",
    )
    parser.add_argument(
        "--hard-group-jitter",
        type=int,
        default=48,
        help="Pixel jitter around the selected hard source center.",
    )
    parser.add_argument(
        "--hard-group-seed",
        type=int,
        default=None,
        help="Optional seed for the hard-group sampler.",
    )
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--fit-parameters",
        type=int,
        default=1,
        help="CIANNA fit_parts.parameters value. Use -1 to disable base physical parameter gradients for staged warm starts.",
    )
    parser.add_argument(
        "--param-loss-scale",
        type=float,
        default=5.0,
        help="CIANNA error_scales.parameters value for flux/bmaj/bmin regression.",
    )
    parser.add_argument("--flux-refine-mode", type=int, default=0)
    parser.add_argument("--flux-refine-loss-scale", type=float, default=0.0)
    parser.add_argument("--flux-refine-gate-loss-scale", type=float, default=0.0)
    parser.add_argument("--flux-refine-final-loss-scale", type=float, default=0.0)
    parser.add_argument("--flux-refine-delta-norm-scale", type=float, default=0.25)
    parser.add_argument("--flux-refine-delta-param-index", type=int, default=3)
    parser.add_argument("--flux-refine-gate-param-index", type=int, default=4)
    parser.add_argument("--flux-refine-detach-base", type=int, default=1)
    parser.add_argument("--flux-refine-gate-margin-norm", type=float, default=0.01)
    parser.add_argument(
        "--flux-refine-target-table",
        type=Path,
        default=None,
        help="Optional flux-refine CSV target table with source_id and delta_log_adjust columns.",
    )
    parser.add_argument("--flux-refine-target-source-col", default="source_id")
    parser.add_argument("--flux-refine-target-delta-adjust-col", default="delta_log_adjust")
    parser.add_argument(
        "--flux-refine-target-only",
        action="store_true",
        help="When a flux-refine table is provided, train only sources present in that table.",
    )
    parser.add_argument("--skip-valid-angle-report", action="store_true")
    parser.add_argument("--valid-angle-report-every", type=int, default=0)
    parser.add_argument("--keep-valid-forward", action="store_true")
    parser.add_argument(
        "--freeze-all-but-last",
        action="store_true",
        help="Freeze all layers except the final YOLO conv. Used by ORSDet flux head native delta-head training.",
    )
    parser.add_argument(
        "--freeze-layer-count",
        type=int,
        default=18,
        help="Number of detector layers to pass to CIANNA.set_frozen_layers when --freeze-all-but-last is set.",
    )
    parser.add_argument("load_epoch", nargs="?", type=int, default=0)
    args = parser.parse_args()
    args.target_source = normalize_target_source(args.target_source)
    args.slim_mode = normalize_slim_mode(args.slim_mode)
    layout = detector_layout(args.slim_mode)
    if args.no_small_elongated_boost:
        args.small_elongated_boost_factor = 1.0

    configure_paths()
    install_numba_fallback_if_needed()

    import CIANNA as cnn
    import data_gen as dg
    from orsdet_angle import (
        append_angle_history_row,
        metrics_from_valid_targets_and_forward,
        parse_yolo_forward,
        write_valid_angle_report,
    )
    from orsdet_angle.angle_weight_variants import SmallElongatedBoostConfig

    target_table = (args.target_table or target_table_path(args.target_source)).resolve()
    run_dir_default = default_run_dir_for_target_source(args.target_source, args.slim_mode)
    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
    elif args.new_run:
        run_dir = next_run_dir(args.runs_root.resolve())
    else:
        run_dir = run_dir_default.resolve()
    flux_refine_target_table = args.flux_refine_target_table.resolve() if args.flux_refine_target_table is not None else None
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "net_save").mkdir(exist_ok=True)
    (run_dir / "fwd_res").mkdir(exist_ok=True)
    os.chdir(run_dir)
    with (run_dir / "run_info.txt").open("a", encoding="utf-8") as f:
        f.write("mode=train_detector argv=%s\n" % " ".join(sys.argv))
        f.write("slim_mode=%s nb_param=%d nb_angle=%d target_stride=%d total_aux=%d\n" % (
            layout.mode,
            layout.nb_param,
            layout.nb_angle,
            layout.target_stride,
            layout.total_aux,
        ))
        f.write("target_source=%s\n" % args.target_source)
        f.write("target_table=%s\n" % target_table)
        f.write("run_dir_default_for_target=%s\n" % run_dir_default)
        f.write("python_executable=%s\n" % sys.executable)
        f.write("numpy_version=%s numpy_file=%s\n" % (np.__version__, np.__file__))
        f.write(
            "small_elongated_boost_factor=%.6g small_elongated_sqrt_area=%.6g "
            "small_elongated_aspect=%.6g\n"
            % (
                args.small_elongated_boost_factor,
                args.small_elongated_sqrt_area,
                args.small_elongated_aspect,
            )
        )
        f.write(
            "hard_group_sampling=%s hard_group_fraction=%.6g hard_group_jitter=%d hard_group_seed=%s\n"
            % (
                args.hard_group_sampling,
                args.hard_group_fraction,
                args.hard_group_jitter,
                args.hard_group_seed,
            )
        )
        f.write(
            "flux_refine_mode=%d delta_loss_scale=%.6g gate_loss_scale=%.6g final_loss_scale=%.6g "
            "delta_norm_scale=%.6g delta_param_index=%d gate_param_index=%d detach_base=%d gate_margin_norm=%.6g\n"
            % (
                args.flux_refine_mode,
                args.flux_refine_loss_scale,
                args.flux_refine_gate_loss_scale,
                args.flux_refine_final_loss_scale,
                args.flux_refine_delta_norm_scale,
                args.flux_refine_delta_param_index,
                args.flux_refine_gate_param_index,
                args.flux_refine_detach_base,
                args.flux_refine_gate_margin_norm,
            )
        )
        f.write(
            "flux_refine_target_table=%s source_col=%s delta_adjust_col=%s target_only=%d\n"
            % (
                flux_refine_target_table,
                args.flux_refine_target_source_col,
                args.flux_refine_target_delta_adjust_col,
                1 if args.flux_refine_target_only else 0,
            )
        )
        f.write(
            "stage_controls fit_parameters=%d param_loss_scale=%.6g\n"
            % (args.fit_parameters, args.param_loss_scale)
        )
        f.write(
            "fit_parts position=%d size=%d probability=%d objectness=%d class=%d parameters=%d angle=%d\n"
            % (
                args.fit_position,
                args.fit_size,
                args.fit_probability,
                args.fit_objectness,
                args.fit_class,
                args.fit_parameters,
                args.fit_angle,
            )
        )
        f.write(
            "freeze_all_but_last=%d freeze_layer_count=%d\n"
            % (1 if args.freeze_all_but_last else 0, args.freeze_layer_count)
        )

    dg.dataset_perscut(dg.TRAINING_CATALOG_PATH, "TrainingSet_perscut.txt", 18)
    cnn.init(
        in_dim=i_ar([dg.image_size, dg.image_size]),
        in_nb_ch=1,
        out_dim=detector_target_dim(dg.max_nb_obj_per_image, layout.mode),
        bias=0.1,
        b_size=args.batch_size,
        comp_meth=args.comp_meth,
        dynamic_load=1,
        mixed_precision=args.mixed_precision,
        adv_size=30,
    )

    dg.init_data_gen()
    boost_config = None
    if args.small_elongated_boost_factor > 1.0:
        boost_config = SmallElongatedBoostConfig(
            small_sqrt_area_cutoff=args.small_elongated_sqrt_area,
            elongated_aspect_cutoff=args.small_elongated_aspect,
            boost_factor=args.small_elongated_boost_factor,
            max_weight=2.0,
        )
    hard_group_config = HardGroupSamplerConfig(
        enabled=bool(args.hard_group_sampling),
        fraction=float(args.hard_group_fraction),
        jitter=int(args.hard_group_jitter),
        seed=args.hard_group_seed,
    )
    builder = DetectorDataBuilder(
        dg,
        source_table_path=target_table,
        slim_mode=layout.mode,
        small_elongated_boost=boost_config,
        hard_group_sampler=hard_group_config,
    )
    if flux_refine_target_table is not None:
        import pandas as pd

        target_csv = flux_refine_target_table
        if not target_csv.is_file():
            raise FileNotFoundError("flux-refine flux refine target table not found: %s" % target_csv)
        target_frame = pd.read_csv(target_csv)
        target_summary = builder.apply_flux_refine_target_table(
            target_frame,
            source_id_col=args.flux_refine_target_source_col,
            delta_adjust_col=args.flux_refine_target_delta_adjust_col,
            target_only=bool(args.flux_refine_target_only),
        )
        builder.reset_hard_group_sampler(hard_group_config)
        with (run_dir / "run_info.txt").open("a", encoding="utf-8") as f:
            f.write("flux_refine_target_summary=%s\n" % target_summary)
        print("flux-refine flux target summary:", target_summary)
    builder.save_norm_lims(run_dir / "train_norm.txt")
    with (run_dir / "run_info.txt").open("a", encoding="utf-8") as f:
        f.write("hard_group_sampler: %s\n" % builder.hard_group_description())

    def create_train_batch():
        return builder.create_train_batch(return_meta=False)

    def create_valid_batch():
        return builder.create_valid_batch(return_meta=False)

    def data_augm():
        input_data, targets = create_train_batch()
        cnn.delete_dataset("TRAIN_buf", silent=1)
        cnn.create_dataset("TRAIN_buf", dg.nb_images_iter, input_data[:, :], targets[:, :], silent=1)

    input_data, targets = create_train_batch()
    input_valid, targets_valid = create_valid_batch()
    np.save(run_dir / "valid_targets.npy", targets_valid)
    cnn.create_dataset("TRAIN", dg.nb_images_iter, input_data[:, :], targets[:, :])
    cnn.create_dataset("VALID", dg.nb_valid, input_valid[:, :], targets_valid[:, :])

    prior_w = f_ar([6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 12.0, 9.0, 24.0])
    prior_h = f_ar([6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 9.0, 12.0, 24.0])
    prior_size = np.vstack((prior_w, prior_h))
    prior_noobj_prob = f_ar([0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.01, 0.01, 0.01])
    error_scales = cnn.set_error_scales(
        position=36.0,
        size=0.2,
        probability=0.5,
        objectness=2.0,
        parameters=args.param_loss_scale,
    )
    param_ind_values = [2.0, 2.0, 2.0] + [1.0] * max(0, layout.nb_param - 3)
    param_ind_scales = f_ar(param_ind_values)
    iou_limits = cnn.set_IoU_limits(
        good_IoU_lim=0.30,
        low_IoU_best_box_assoc=-0.01,
        min_prob_IoU_lim=-0.01,
        min_obj_IoU_lim=-0.01,
        min_param_IoU_lim=-0.01,
    )
    fit_parts = cnn.set_fit_parts(
        position=args.fit_position,
        size=args.fit_size,
        probability=args.fit_probability,
        objectness=args.fit_objectness,
        classes=args.fit_class,
        parameters=args.fit_parameters,
    )
    slopes_and_maxes = cnn.set_slopes_and_maxes(
        position=cnn.set_sm_single(slope=0.5, fmax=6.0, fmin=-6.0),
        size=cnn.set_sm_single(slope=0.5, fmax=1.2, fmin=-1.2),
        probability=cnn.set_sm_single(slope=0.2, fmax=6.0, fmin=-6.0),
        objectness=cnn.set_sm_single(slope=0.5, fmax=6.0, fmin=-6.0),
        parameters=cnn.set_sm_single(slope=0.5, fmax=1.5, fmin=-0.2),
    )

    nb_yolo_filters = set_yolo_params_checked(
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
        strict_box_size=0,
        min_prior_forced_scaling=0.0,
        rand_startup=dg.nb_images_iter * 10,
        rand_prob_best_box_assoc=0.90,
        rand_prob=0.02,
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
        fit_angle=args.fit_angle,
        scorer_aux_flux_log_max=float(builder.norm_lims[0, 0]),
        scorer_aux_flux_log_min=float(builder.norm_lims[0, 1]),
        flux_refine_mode=args.flux_refine_mode,
        flux_refine_delta_param_index=args.flux_refine_delta_param_index,
        flux_refine_gate_param_index=args.flux_refine_gate_param_index,
        flux_refine_detach_base=args.flux_refine_detach_base,
        flux_refine_loss_scale=args.flux_refine_loss_scale,
        flux_refine_gate_loss_scale=args.flux_refine_gate_loss_scale,
        flux_refine_final_loss_scale=args.flux_refine_final_loss_scale,
        flux_refine_delta_norm_scale=args.flux_refine_delta_norm_scale,
        flux_refine_gate_margin_norm=args.flux_refine_gate_margin_norm,
    )

    if args.load_epoch > 0:
        cnn.load("net_save/net0_s%04d.dat" % args.load_epoch, args.load_epoch, bin=1)
    else:
        cnn.conv(f_size=i_ar([5, 5]), nb_filters=32, stride=i_ar([1, 1]), padding=i_ar([2, 2]), activation="RELU")
        cnn.conv(f_size=i_ar([2, 2]), nb_filters=16, stride=i_ar([2, 2]), padding=i_ar([0, 0]), activation="RELU")
        cnn.conv(f_size=i_ar([3, 3]), nb_filters=24, stride=i_ar([1, 1]), padding=i_ar([1, 1]), activation="RELU")
        cnn.conv(f_size=i_ar([3, 3]), nb_filters=32, stride=i_ar([1, 1]), padding=i_ar([1, 1]), activation="RELU")
        cnn.conv(f_size=i_ar([2, 2]), nb_filters=64, stride=i_ar([2, 2]), padding=i_ar([0, 0]), activation="RELU")
        cnn.conv(f_size=i_ar([1, 1]), nb_filters=128, stride=i_ar([1, 1]), padding=i_ar([0, 0]), activation="RELU")
        cnn.conv(f_size=i_ar([3, 3]), nb_filters=192, stride=i_ar([1, 1]), padding=i_ar([1, 1]), activation="RELU")
        cnn.conv(f_size=i_ar([2, 2]), nb_filters=128, stride=i_ar([2, 2]), padding=i_ar([0, 0]), activation="RELU")
        cnn.conv(f_size=i_ar([1, 1]), nb_filters=192, stride=i_ar([1, 1]), padding=i_ar([0, 0]), activation="RELU")
        cnn.conv(f_size=i_ar([3, 3]), nb_filters=384, stride=i_ar([1, 1]), padding=i_ar([1, 1]), activation="RELU")
        cnn.conv(f_size=i_ar([1, 1]), nb_filters=256, stride=i_ar([1, 1]), padding=i_ar([0, 0]), activation="RELU")
        cnn.conv(f_size=i_ar([3, 3]), nb_filters=384, stride=i_ar([1, 1]), padding=i_ar([1, 1]), activation="RELU")
        cnn.conv(f_size=i_ar([2, 2]), nb_filters=512, stride=i_ar([2, 2]), padding=i_ar([0, 0]), activation="RELU")
        cnn.conv(f_size=i_ar([1, 1]), nb_filters=768, stride=i_ar([1, 1]), padding=i_ar([0, 0]), activation="RELU")
        cnn.conv(f_size=i_ar([3, 3]), nb_filters=1024, stride=i_ar([1, 1]), padding=i_ar([1, 1]), activation="RELU")
        cnn.norm(group_size=4)
        cnn.conv(f_size=i_ar([1, 1]), nb_filters=2048, stride=i_ar([1, 1]), padding=i_ar([0, 0]), activation="RELU", drop_rate=0.25)
        cnn.conv(f_size=i_ar([1, 1]), nb_filters=nb_yolo_filters, stride=i_ar([1, 1]), padding=i_ar([0, 0]), activation="YOLO")

    if args.freeze_all_but_last:
        if args.freeze_layer_count < 2:
            raise ValueError("--freeze-layer-count must be >= 2 when --freeze-all-but-last is used.")
        frozen = np.ones(int(args.freeze_layer_count), dtype=np.int32)
        frozen[-1] = 0
        cnn.set_frozen_layers(frozen)
        print("ORSDet flux head native mode: frozen all layers except final YOLO conv (%d entries)." % args.freeze_layer_count)

    if shutil.which("pdflatex") is not None:
        cnn.print_arch_tex("./arch/", "arch", activation=1, dropout=1)

    grid_size = dg.image_size // dg.c_size
    valid_test_ready = False

    def detector_obb_angle_metrics(targets, forward):
        compact_nb_angle = 2
        metric_nb_param = min(layout.nb_param, 3)
        compact_target_stride = 7 + metric_nb_param + compact_nb_angle + 1
        compact_output_stride = 8 + metric_nb_param + compact_nb_angle
        output_stride = 8 + layout.nb_param + layout.nb_angle
        compact_targets = np.zeros(
            (targets.shape[0], 1 + dg.max_nb_obj_per_image * compact_target_stride),
            dtype=targets.dtype,
        )
        compact_targets[:, 0] = targets[:, 0]
        for sample_i in range(targets.shape[0]):
            n_obj = int(min(max(targets[sample_i, 0], 0), dg.max_nb_obj_per_image))
            for obj_i in range(n_obj):
                src = 1 + obj_i * layout.target_stride
                dst = 1 + obj_i * compact_target_stride
                block = targets[sample_i, src : src + layout.target_stride]
                compact_targets[sample_i, dst : dst + 7 + metric_nb_param] = block[: 7 + metric_nb_param]
                compact_targets[sample_i, dst + 7 + metric_nb_param : dst + 7 + metric_nb_param + compact_nb_angle] = block[
                    layout.row_angle_start : layout.row_angle_start + compact_nb_angle
                ]
                compact_targets[sample_i, dst + compact_target_stride - 1] = block[layout.target_stride - 1]

        compact_forward = np.zeros(
            (forward.shape[0], dg.nb_box * compact_output_stride, forward.shape[2]),
            dtype=forward.dtype,
        )
        for box_i in range(dg.nb_box):
            src = box_i * output_stride
            dst = box_i * compact_output_stride
            compact_forward[:, dst : dst + 8 + metric_nb_param, :] = forward[:, src : src + 8 + metric_nb_param, :]
            angle_src = src + 8 + layout.nb_param
            angle_dst = dst + 8 + metric_nb_param
            compact_forward[:, angle_dst : angle_dst + compact_nb_angle, :] = forward[
                :, angle_src : angle_src + compact_nb_angle, :
            ]

        return metrics_from_valid_targets_and_forward(
            compact_targets,
            compact_forward,
            image_size=dg.image_size,
            grid_size=grid_size,
            nb_box=dg.nb_box,
            lims=builder.norm_lims,
            nb_param=metric_nb_param,
            nb_angle=compact_nb_angle,
            unit_norm_weight=args.angle_unit_norm_scale,
        )

    def run_valid_angle_check(epoch: int, subdir: str, append_history: bool = True):
        nonlocal valid_test_ready
        report_dir = run_dir / subdir
        report_dir.mkdir(parents=True, exist_ok=True)
        fwd_path = run_dir / "fwd_res" / ("net0_%04d.dat" % epoch)
        backup_path = fwd_path.with_suffix(fwd_path.suffix + ".before_valid")
        if fwd_path.is_file():
            if backup_path.exists():
                backup_path.unlink()
            fwd_path.replace(backup_path)
        if not valid_test_ready:
            cnn.delete_dataset("TEST", silent=1)
            cnn.create_dataset("TEST", dg.nb_valid, input_valid[:, :], targets_valid[:, :], silent=1)
            valid_test_ready = True
        cnn.forward(no_error=1, saving=1, repeat=1, drop_mode="AVG_MODEL", silent=1)
        forward = parse_yolo_forward(fwd_path, dg.nb_valid, grid_size, nb_yolo_filters)
        metrics = detector_obb_angle_metrics(targets_valid, forward)
        if append_history:
            append_angle_history_row(run_dir / "angle_history.csv", epoch, metrics)
        if args.keep_valid_forward:
            fwd_path.replace(report_dir / "valid_fwd.dat")
        else:
            fwd_path.unlink(missing_ok=True)
        if backup_path.is_file():
            backup_path.replace(fwd_path)
        return metrics, write_valid_angle_report(metrics, report_dir)

    warmup_delay = 40
    momentum = 0.8
    lr_decay = 0.0005
    TRAIN_SILENT = 1
    VALIDATION_ONLY_PROGRESS = 2
    for block in range(args.load_epoch, args.load_epoch + args.epochs):
        t = Thread(target=data_augm)
        t.start()
        if (block + 1) <= warmup_delay:
            loc_lr = 0.98 * args.learning_rate * ((block + 1) / warmup_delay) + 0.02 * args.learning_rate
        else:
            loc_lr = args.learning_rate
        silent_mode = (
            VALIDATION_ONLY_PROGRESS if args.control_interv > 0 and (block + 1) % args.control_interv == 0 else TRAIN_SILENT
        )
        cnn.train(
            nb_iter=1,
            learning_rate=loc_lr,
            end_learning_rate=loc_lr * args.end_lr_prop,
            shuffle_every=0,
            momentum=momentum,
            lr_decay=lr_decay,
            weight_decay=0.0,
            control_interv=args.control_interv,
            save_every=args.save_every,
            silent=silent_mode,
            save_bin=1,
        )
        t.join()
        cnn.swap_data_buffers("TRAIN")
        current_epoch = block + 1
        if args.valid_angle_report_every > 0 and current_epoch % args.valid_angle_report_every == 0:
            run_valid_angle_check(current_epoch, "angle_report_epoch_%04d" % current_epoch)

    if not args.skip_valid_angle_report:
        final_epoch = args.load_epoch + args.epochs
        append_final = not (args.valid_angle_report_every > 0 and final_epoch % args.valid_angle_report_every == 0)
        metrics, report = run_valid_angle_check(final_epoch, "angle_report", append_history=append_final)
        with (run_dir / "angle_report.txt").open("w", encoding="utf-8") as f:
            f.write(report)
        print(report)

    write_error_summary(run_dir)
    print("detector run directory:", run_dir)


if __name__ == "__main__":
    main()
