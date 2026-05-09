#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import html
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch_npu

from profile_secure_pipeline_staged import aggregate, run_once_staged
from secure_npu_resnet_pipeline import PipelineConfig, build_resnet50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export total performance summary CSV + HTML")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--image-sizes", type=str, default="128,224,320")
    p.add_argument("--micro-sizes", type=str, default="4096,16384,65536,262144,1048576,4194304")
    p.add_argument("--micro-warmup", type=int, default=20)
    p.add_argument("--micro-iters", type=int, default=100)
    p.add_argument("--out-dir", type=str, default="./perf_summary_report")
    p.add_argument("--profiler-output", type=str, default="./torch_npu_profiler_output")
    p.add_argument("--run-profiler", action="store_true")
    p.add_argument("--profiler-warmup", type=int, default=1)
    p.add_argument("--profiler-active", type=int, default=2)
    return p.parse_args()


def _to_int_list(raw: str) -> List[int]:
    out: List[int] = []
    for s in raw.split(","):
        s = s.strip()
        if not s:
            continue
        out.append(int(s))
    return out


def _safe_ratio(part: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return part / total * 100.0


def run_stage_benchmark(args: argparse.Namespace) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
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

    runs = []
    for _ in range(max(1, args.iters)):
        _probs, stages = run_once_staged(cfg, model)
        runs.append(stages)

    avg = aggregate(runs)
    total_ms = sum(s.ms for s in avg)

    stage_rows: List[Dict[str, str]] = []
    for s in avg:
        stage_rows.append(
            {
                "section": "end_to_end_stage",
                "metric": s.name,
                "value": f"{s.ms:.6f}",
                "unit": "ms",
                "ratio_pct": f"{_safe_ratio(s.ms, total_ms):.2f}",
                "note": f"image_size={args.image_size}, warmup={args.warmup}, iters={args.iters}",
            }
        )

    stage_rows.append(
        {
            "section": "end_to_end_stage",
            "metric": "total",
            "value": f"{total_ms:.6f}",
            "unit": "ms",
            "ratio_pct": "100.00",
            "note": f"image_size={args.image_size}, warmup={args.warmup}, iters={args.iters}",
        }
    )

    multi_rows: List[Dict[str, str]] = []
    for size in _to_int_list(args.image_sizes):
        cfg_scale = PipelineConfig(
            image_path=args.image,
            image_size=size,
            key_hex=cfg.key_hex,
            iv_hex=cfg.iv_hex,
            use_pretrained=cfg.use_pretrained,
            topk=cfg.topk,
            strict_float_view=cfg.strict_float_view,
        )
        for _ in range(max(0, args.warmup)):
            _probs, _ = run_once_staged(cfg_scale, model)

        scale_runs = []
        for _ in range(max(1, args.iters)):
            _probs, st = run_once_staged(cfg_scale, model)
            scale_runs.append(st)

        scale_avg = aggregate(scale_runs)
        total_us = sum(s.ms for s in scale_avg) * 1000.0
        input_total_len = 3 * size * size
        throughput = 0.0
        if total_us > 0:
            throughput = float(input_total_len / (total_us / 1e6) / 1e6)

        multi_rows.append(
            {
                "section": "multi_scale_throughput",
                "metric": f"image_size_{size}",
                "value": f"{throughput:.6f}",
                "unit": "MB/s",
                "ratio_pct": "",
                "note": f"input_elements={input_total_len}, total_us={total_us:.3f}",
            }
        )

    return stage_rows, multi_rows


def run_microbenchmark(args: argparse.Namespace) -> List[Dict[str, str]]:
    sizes = _to_int_list(args.micro_sizes)
    warmup = max(0, args.micro_warmup)
    iters = max(1, args.micro_iters)

    key = torch.randint(0, 256, (16,), dtype=torch.uint8, device="npu")
    iv = torch.randint(0, 256, (16,), dtype=torch.uint8, device="npu")

    rows: List[Dict[str, str]] = []
    for n in sizes:
        x = torch.randint(0, 256, (n,), dtype=torch.uint8, device="npu")
        for _ in range(warmup):
            _ = torch_npu.aes_ctr_encrypt(x, key, iv)
        torch.npu.synchronize()

        s_evt = torch.npu.Event(enable_timing=True)
        e_evt = torch.npu.Event(enable_timing=True)
        s_evt.record()
        for _ in range(iters):
            _ = torch_npu.aes_ctr_encrypt(x, key, iv)
        e_evt.record()
        torch.npu.synchronize()

        total_ms = float(s_evt.elapsed_time(e_evt))
        avg_ms = total_ms / iters
        throughput = (n / 1e6) / (avg_ms / 1e3) if avg_ms > 0 else 0.0
        rows.append(
            {
                "section": "aes_ctr_microbenchmark",
                "metric": f"size_{n}_bytes",
                "value": f"{throughput:.6f}",
                "unit": "MB/s",
                "ratio_pct": "",
                "note": f"avg_ms={avg_ms:.6f}, warmup={warmup}, iters={iters}",
            }
        )
        rows.append(
            {
                "section": "aes_ctr_microbenchmark",
                "metric": f"size_{n}_bytes_latency",
                "value": f"{avg_ms:.6f}",
                "unit": "ms",
                "ratio_pct": "",
                "note": f"warmup={warmup}, iters={iters}",
            }
        )

    return rows


def run_profiler_capture(args: argparse.Namespace) -> None:
    cmd = [
        "/usr/bin/python3",
        "profile_secure_pipeline_torch_npu.py",
        "--image",
        args.image,
        "--image-size",
        str(args.image_size),
        "--topk",
        str(args.topk),
        "--warmup",
        str(args.profiler_warmup),
        "--active",
        str(args.profiler_active),
        "--skip-first",
        "0",
        "--output",
        args.profiler_output,
        "--row-limit",
        "20",
    ]
    subprocess.run(cmd, check=True)


def latest_profile_dir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0]


def parse_top_kernel_rows(profile_dir: Optional[Path]) -> List[Dict[str, str]]:
    if profile_dir is None:
        return []
    kernel_csv = profile_dir / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv"
    if not kernel_csv.exists():
        return []

    parsed: List[Tuple[str, float]] = []
    with kernel_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            name = r.get("Name") or r.get("Kernel Name") or r.get("Op Name") or "UNKNOWN"
            dur_raw = r.get("Duration(us)") or r.get("Task Duration(us)") or r.get("Duration") or ""
            try:
                dur = float(dur_raw)
            except Exception:
                continue
            parsed.append((name, dur))

    parsed.sort(key=lambda x: x[1], reverse=True)
    rows: List[Dict[str, str]] = []
    for i, (name, dur) in enumerate(parsed[:20], start=1):
        rows.append(
            {
                "section": "torch_npu_profiler_top_kernels",
                "metric": f"top_{i}_{name}",
                "value": f"{dur:.6f}",
                "unit": "us",
                "ratio_pct": "",
                "note": f"profile_dir={profile_dir.name}",
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["section", "metric", "value", "unit", "ratio_pct", "note"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_html(path: Path, rows: List[Dict[str, str]], meta: Dict[str, str]) -> None:
    sections: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        sections.setdefault(r["section"], []).append(r)

    blocks: List[str] = []
    for sec, items in sections.items():
        lines = [
            f"<h2>{html.escape(sec)}</h2>",
            "<table>",
            "<thead><tr><th>metric</th><th>value</th><th>unit</th><th>ratio_pct</th><th>note</th></tr></thead>",
            "<tbody>",
        ]
        for it in items:
            lines.append(
                "<tr>"
                f"<td>{html.escape(it['metric'])}</td>"
                f"<td>{html.escape(it['value'])}</td>"
                f"<td>{html.escape(it['unit'])}</td>"
                f"<td>{html.escape(it['ratio_pct'])}</td>"
                f"<td>{html.escape(it['note'])}</td>"
                "</tr>"
            )
        lines.extend(["</tbody>", "</table>"])
        blocks.append("\n".join(lines))

    meta_lines = "".join(
        f"<li><b>{html.escape(k)}:</b> {html.escape(v)}</li>" for k, v in meta.items()
    )

    doc = f"""<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>AES CTR Performance Total Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 24px; color: #1f2328; }}
    h1 {{ margin-bottom: 8px; }}
    h2 {{ margin-top: 28px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 8px; font-size: 14px; }}
    th {{ background: #f6f8fa; text-align: left; }}
    tr:nth-child(even) {{ background: #fafbfc; }}
    .meta {{ background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px; padding: 12px 16px; }}
  </style>
</head>
<body>
  <h1>AES CTR Performance Total Report</h1>
  <div class=\"meta\">
    <ul>{meta_lines}</ul>
  </div>
  {''.join(blocks)}
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"image not found: {args.image}")
    if not torch.npu.is_available():
        raise RuntimeError("NPU is not available")

    if args.run_profiler:
        run_profiler_capture(args)

    stage_rows, multi_rows = run_stage_benchmark(args)
    micro_rows = run_microbenchmark(args)

    pdir = latest_profile_dir(Path(args.profiler_output))
    profiler_rows = parse_top_kernel_rows(pdir)

    all_rows = stage_rows + multi_rows + micro_rows + profiler_rows

    out_dir = Path(args.out_dir)
    total_csv = out_dir / "performance_total_summary.csv"
    total_html = out_dir / "performance_total_summary.html"
    write_csv(total_csv, all_rows)

    meta = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image": args.image,
        "image_size": str(args.image_size),
        "warmup": str(args.warmup),
        "iters": str(args.iters),
        "micro_warmup": str(args.micro_warmup),
        "micro_iters": str(args.micro_iters),
        "profiler_output": args.profiler_output,
        "latest_profiler_dir": pdir.name if pdir else "N/A",
    }
    write_html(total_html, all_rows, meta)

    print(f"total csv: {total_csv}")
    print(f"total html: {total_html}")
    print(f"rows: {len(all_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
