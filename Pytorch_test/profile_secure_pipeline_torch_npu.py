import argparse
import csv
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch_npu

from secure_npu_resnet_pipeline import PipelineConfig, secure_infer


def build_cfg(image_path: str, image_size: int, topk: int) -> PipelineConfig:
    return PipelineConfig(
        image_path=image_path,
        image_size=image_size,
        key_hex="2b7e151628aed2a6abf7158809cf4f3c",
        iv_hex="f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff",
        use_pretrained=True,
        topk=topk,
        strict_float_view=False,
    )


def pick_activities() -> List[torch_npu.profiler.ProfilerActivity]:
    supported = set(torch_npu.profiler.supported_activities())
    acts = []
    if torch_npu.profiler.ProfilerActivity.CPU in supported:
        acts.append(torch_npu.profiler.ProfilerActivity.CPU)
    if torch_npu.profiler.ProfilerActivity.NPU in supported:
        acts.append(torch_npu.profiler.ProfilerActivity.NPU)
    if not acts:
        acts = [torch_npu.profiler.ProfilerActivity.NPU]
    return acts


def make_experimental_config():
    # Keep config minimal and widely compatible.
    try:
        return torch_npu.profiler._ExperimentalConfig(
            export_type=[torch_npu.profiler.ExportType.Text],
            profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            msprof_tx=False,
            l2_cache=False,
            op_attr=False,
            data_simplification=False,
            record_op_args=False,
            gc_detect_threshold=None,
        )
    except Exception:
        return None


def _safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return -1.0


def find_latest_profile_dir(output_root: str) -> Optional[Path]:
    root = Path(output_root)
    if not root.exists():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0]


def print_top_ops_from_csv(profile_dir: Path, topn: int = 20) -> None:
    op_csv = profile_dir / "ASCEND_PROFILER_OUTPUT" / "operator_details.csv"
    if not op_csv.exists():
        print("\nProfiler output missing operator_details.csv, skip top-op summary.")
        return

    with op_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("\noperator_details.csv is empty.")
        return

    # Prefer common duration fields across versions.
    duration_keys = [
        "Device Self Duration With AICore(us)",
        "Device Total Duration With AICore(us)",
        "Device Self Duration(us)",
        "Device Total Duration(us)",
        "Host Self Duration(us)",
        "Host Total Duration(us)",
        "Task Duration(us)",
        "Task Duration", 
        "Duration(us)",
        "Duration",
    ]
    op_name_keys = ["Op Name", "OP Type", "Kernel Name", "Name"]

    present_keys = [k for k in duration_keys if k in rows[0]]
    dur_key = None
    best_sum = -1.0
    for key in present_keys:
        s = sum(_safe_float(r.get(key, "")) for r in rows)
        if s > best_sum:
            best_sum = s
            dur_key = key
    name_key = next((k for k in op_name_keys if k in rows[0]), None)
    if dur_key is None:
        print("\nCannot find duration column in operator_details.csv.")
        return

    ranked = sorted(rows, key=lambda r: _safe_float(r.get(dur_key, "")), reverse=True)

    print("\n=== Top Operators (from operator_details.csv) ===")
    print(f"Sort column: {dur_key}")
    for i, r in enumerate(ranked[:topn], start=1):
        name = r.get(name_key, "UNKNOWN") if name_key else "UNKNOWN"
        dur = r.get(dur_key, "N/A")
        print(f"{i:2d}. {name} | duration={dur}")


def print_top_kernels_from_csv(profile_dir: Path, topn: int = 20) -> None:
    kernel_csv = profile_dir / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv"
    if not kernel_csv.exists():
        return

    with kernel_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return

    duration_keys = ["Duration(us)", "Task Duration(us)", "Duration"]
    name_keys = ["Name", "Kernel Name", "Op Name"]
    present_d = [k for k in duration_keys if k in rows[0]]
    dur_key = next(iter(present_d), None)
    name_key = next((k for k in name_keys if k in rows[0]), None)
    if dur_key is None:
        return

    ranked = sorted(rows, key=lambda r: _safe_float(r.get(dur_key, "")), reverse=True)
    print("\n=== Top Kernels (from kernel_details.csv) ===")
    print(f"Sort column: {dur_key}")
    for i, r in enumerate(ranked[:topn], start=1):
        name = r.get(name_key, "UNKNOWN") if name_key else "UNKNOWN"
        dur = r.get(dur_key, "N/A")
        print(f"{i:2d}. {name} | duration={dur}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile secure NPU pipeline with torch_npu.profiler")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1, help="Profiler warmup steps")
    parser.add_argument("--active", type=int, default=3, help="Profiler active steps")
    parser.add_argument("--skip-first", type=int, default=0)
    parser.add_argument("--output", type=str, default="./torch_npu_profiler_output")
    parser.add_argument("--row-limit", type=int, default=30)
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    Path(args.output).mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args.image, args.image_size, args.topk)

    # One warmup execution before profiling, avoids first-run compile effects in capture window.
    _ = secure_infer(cfg)
    torch.npu.synchronize()

    total_steps = args.skip_first + args.warmup + args.active
    activities = pick_activities()
    exp_cfg = make_experimental_config()

    schedule = torch_npu.profiler.schedule(
        wait=0,
        warmup=args.warmup,
        active=args.active,
        repeat=1,
        skip_first=args.skip_first,
    )

    profile_kwargs = dict(
        activities=activities,
        schedule=schedule,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(args.output),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        with_flops=False,
    )

    if exp_cfg is not None:
        profile_kwargs["experimental_config"] = exp_cfg

    with torch_npu.profiler.profile(**profile_kwargs) as prof:
        for _ in range(total_steps):
            _ = secure_infer(cfg)
            torch.npu.synchronize()
            prof.step()

    latest = find_latest_profile_dir(args.output)
    if latest is None:
        print("\nNo profiler directory generated.")
        return 1

    # Optional post-analysis pass provided by torch_npu profiler.
    try:
        torch_npu.profiler.profiler.analyse(str(latest))
    except Exception:
        pass

    print_top_ops_from_csv(latest, topn=args.row_limit)
    print_top_kernels_from_csv(latest, topn=args.row_limit)
    print(f"\nTrace directory: {latest}")
    print("Use tensorboard to view trace if needed:")
    print(f"tensorboard --logdir {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
