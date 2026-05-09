import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch_npu

from secure_npu_resnet_pipeline import (
    PipelineConfig,
    aes_ctr_encrypt_cpu_u8,
    build_resnet50,
    decrypted_u8_to_model_input,
    get_imagenet_labels,
    hex_to_u8_tensor,
    load_image_to_chw_u8,
    npu_aes_ctr_decrypt,
    topk_to_string,
)


@dataclass
class StageResult:
    name: str
    ms: float


def _npu_elapsed_ms(start_evt: torch.npu.Event, end_evt: torch.npu.Event) -> float:
    torch.npu.synchronize()
    return float(start_evt.elapsed_time(end_evt))


def run_once_staged(cfg: PipelineConfig, model: torch.nn.Module) -> Tuple[torch.Tensor, List[StageResult]]:
    stages: List[StageResult] = []

    # Stage 1: CPU prepare and encrypt.
    t0 = time.perf_counter()
    plain_chw_u8_cpu = load_image_to_chw_u8(cfg.image_path, cfg.image_size)
    cipher_u8_cpu = aes_ctr_encrypt_cpu_u8(plain_chw_u8_cpu, cfg.key_hex, cfg.iv_hex)
    stages.append(StageResult("cpu_prepare_encrypt", (time.perf_counter() - t0) * 1000.0))

    device = "npu"

    # Stage 2: H2D copy.
    s_h2d = torch.npu.Event(enable_timing=True)
    e_h2d = torch.npu.Event(enable_timing=True)
    s_h2d.record()
    cipher_u8_npu = cipher_u8_cpu.to(device, non_blocking=False)
    key_u8_npu = hex_to_u8_tensor(cfg.key_hex, expected_len=len(bytes.fromhex(cfg.key_hex)), device=device)
    iv_u8_npu = hex_to_u8_tensor(cfg.iv_hex, expected_len=16, device=device)
    e_h2d.record()
    stages.append(StageResult("h2d_cipher_key_iv", _npu_elapsed_ms(s_h2d, e_h2d)))

    # Stage 3: NPU decrypt.
    s_dec = torch.npu.Event(enable_timing=True)
    e_dec = torch.npu.Event(enable_timing=True)
    s_dec.record()
    dec_u8_npu = npu_aes_ctr_decrypt(cipher_u8_npu, key_u8_npu, iv_u8_npu)
    e_dec.record()
    stages.append(StageResult("npu_decrypt", _npu_elapsed_ms(s_dec, e_dec)))

    # Stage 4: type conversion and normalization.
    s_cast = torch.npu.Event(enable_timing=True)
    e_cast = torch.npu.Event(enable_timing=True)
    s_cast.record()
    model_input = decrypted_u8_to_model_input(
        dec_u8_npu,
        plain_chw_u8_cpu.shape,
        strict_float_view=cfg.strict_float_view,
    )
    e_cast.record()
    stages.append(StageResult("type_convert_normalize", _npu_elapsed_ms(s_cast, e_cast)))

    # Stage 5: model inference.
    s_inf = torch.npu.Event(enable_timing=True)
    e_inf = torch.npu.Event(enable_timing=True)
    with torch.no_grad():
        s_inf.record()
        logits = model(model_input)
        probs = torch.softmax(logits, dim=1)
        e_inf.record()
    stages.append(StageResult("model_infer", _npu_elapsed_ms(s_inf, e_inf)))

    # Stage 6: D2H probabilities only.
    s_d2h = torch.npu.Event(enable_timing=True)
    e_d2h = torch.npu.Event(enable_timing=True)
    s_d2h.record()
    probs_cpu = probs.cpu()
    e_d2h.record()
    stages.append(StageResult("d2h_result_only", _npu_elapsed_ms(s_d2h, e_d2h)))

    # Wipe plaintext from NPU memory.
    dec_u8_npu.zero_()
    del dec_u8_npu, model_input, cipher_u8_npu, key_u8_npu, iv_u8_npu, plain_chw_u8_cpu, cipher_u8_cpu
    torch.npu.synchronize()
    torch.npu.empty_cache()

    return probs_cpu, stages


def aggregate(results: List[List[StageResult]]) -> List[StageResult]:
    if not results:
        return []

    names = [s.name for s in results[0]]
    sums: Dict[str, float] = {n: 0.0 for n in names}
    for run in results:
        for s in run:
            sums[s.name] += s.ms

    count = float(len(results))
    return [StageResult(n, sums[n] / count) for n in names]


def print_stage_report(avg_stages: List[StageResult]) -> None:
    total = sum(s.ms for s in avg_stages)
    print("\n=== Stage Timing Report (avg ms) ===")
    if total <= 0:
        for s in avg_stages:
            print(f"{s.name:24s}: {s.ms:10.3f} ms")
        return

    for s in avg_stages:
        ratio = (s.ms / total) * 100.0
        print(f"{s.name:24s}: {s.ms:10.3f} ms   ({ratio:6.2f}%)")
    print(f"{'total':24s}: {total:10.3f} ms")


def export_fallback_csvs(
    avg_stages: List[StageResult],
    out_dir: str,
    image_size: int,
    tile_length: Optional[int],
    core_num: Optional[int],
) -> None:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)

    stage_map = {s.name: s.ms * 1000.0 for s in avg_stages}  # ms -> us
    copy_in = stage_map.get("h2d_cipher_key_iv", 0.0)
    compute = (
        stage_map.get("npu_decrypt", 0.0)
        + stage_map.get("type_convert_normalize", 0.0)
        + stage_map.get("model_infer", 0.0)
    )
    copy_out = stage_map.get("d2h_result_only", 0.0)

    stage_csv = p / "stage_distribution_fallback.csv"
    with stage_csv.open("w", encoding="utf-8") as f:
        f.write("Stage,Duration_us\n")
        f.write(f"CopyIn,{copy_in:.3f}\n")
        f.write(f"Compute,{compute:.3f}\n")
        f.write(f"CopyOut,{copy_out:.3f}\n")

    input_total_len = 3 * int(image_size) * int(image_size)
    tiling_csv = p / "data_scale_tiling_info_fallback.csv"
    with tiling_csv.open("w", encoding="utf-8") as f:
        f.write(
            "InputTotalLength_estimated_elements,TileLength,CoreNum_median_BlockDim,CoreNum_max_BlockDim,Source,Note\n"
        )
        f.write(
            f"{input_total_len},{'' if tile_length is None else tile_length},{'' if core_num is None else core_num},{'' if core_num is None else core_num},"
            "staged_profiler_fallback,"
            "Generated from staged timing script when raw profiler CSV lacks fields.\n"
        )


def export_multi_scale_fallback_csv(rows: List[Dict[str, float]], out_dir: str) -> None:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    out_csv = p / "multi_scale_performance_fallback.csv"
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("ImageSize,InputTotalLength_elements,TotalDuration_us,Throughput_MBps_estimated\n")
        for r in rows:
            f.write(
                f"{int(r['ImageSize'])},{int(r['InputTotalLength_elements'])},{r['TotalDuration_us']:.3f},{r['Throughput_MBps_estimated']:.6f}\n"
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage timing profiler for secure NPU inference pipeline")
    p.add_argument("--image", type=str, required=True, help="Input image path")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--iters", type=int, default=5, help="Measured iterations")
    p.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--out", type=str, default="./csv_perf_report", help="Output directory for fallback CSV files")
    p.add_argument("--tile-length", type=int, default=None, help="Optional tiling parameter tileLength")
    p.add_argument("--core-num", type=int, default=None, help="Optional tiling parameter core num")
    p.add_argument(
        "--image-sizes",
        type=str,
        default="",
        help="Optional comma-separated image sizes for multi-scale measurement, e.g. 128,224,320",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not torch.npu.is_available():
        raise RuntimeError("NPU is not available.")

    cfg = PipelineConfig(
        image_path=args.image,
        image_size=args.image_size,
        key_hex="2b7e151628aed2a6abf7158809cf4f3c",
        iv_hex="f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff",
        use_pretrained=True,
        topk=args.topk,
        strict_float_view=False,
    )

    model = build_resnet50(device="npu", use_pretrained=cfg.use_pretrained)

    for _ in range(max(0, args.warmup)):
        _probs, _ = run_once_staged(cfg, model)

    all_runs: List[List[StageResult]] = []
    last_probs: Optional[torch.Tensor] = None
    for _ in range(max(1, args.iters)):
        last_probs, stages = run_once_staged(cfg, model)
        all_runs.append(stages)

    avg_stages = aggregate(all_runs)
    print_stage_report(avg_stages)
    export_fallback_csvs(
        avg_stages=avg_stages,
        out_dir=args.out,
        image_size=args.image_size,
        tile_length=args.tile_length,
        core_num=args.core_num,
    )

    labels = get_imagenet_labels(cfg.use_pretrained)
    if last_probs is not None:
        print("\n=== Inference Result ===")
        print(topk_to_string(last_probs, cfg.topk, labels=labels))

    # Optional multi-scale measurements.
    multi_sizes = []
    if args.image_sizes.strip():
        multi_sizes = [int(x.strip()) for x in args.image_sizes.split(",") if x.strip()]
    multi_rows: List[Dict[str, float]] = []
    for size in multi_sizes:
        cfg_scale = PipelineConfig(
            image_path=args.image,
            image_size=int(size),
            key_hex=cfg.key_hex,
            iv_hex=cfg.iv_hex,
            use_pretrained=cfg.use_pretrained,
            topk=cfg.topk,
            strict_float_view=cfg.strict_float_view,
        )
        for _ in range(max(0, args.warmup)):
            _p, _ = run_once_staged(cfg_scale, model)
        runs_scale: List[List[StageResult]] = []
        for _ in range(max(1, args.iters)):
            _p, st = run_once_staged(cfg_scale, model)
            runs_scale.append(st)
        avg_scale = aggregate(runs_scale)
        total_ms = sum(s.ms for s in avg_scale)
        total_us = total_ms * 1000.0
        input_total_len = 3 * int(size) * int(size)
        throughput = 0.0
        if total_us > 0:
            throughput = float(input_total_len / (total_us / 1e6) / 1e6)
        multi_rows.append(
            {
                "ImageSize": float(size),
                "InputTotalLength_elements": float(input_total_len),
                "TotalDuration_us": float(total_us),
                "Throughput_MBps_estimated": float(throughput),
            }
        )
    if multi_rows:
        export_multi_scale_fallback_csv(multi_rows, args.out)
        print("\n=== Multi-scale Fallback ===")
        for r in multi_rows:
            print(
                f"size={int(r['ImageSize'])}, input={int(r['InputTotalLength_elements'])}, "
                f"total_us={r['TotalDuration_us']:.3f}, throughput={r['Throughput_MBps_estimated']:.6f} MB/s"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
