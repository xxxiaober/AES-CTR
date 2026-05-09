import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch_npu

try:
    from Crypto.Cipher import AES
except Exception:
    AES = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from torchvision.models import ResNet50_Weights, resnet50
except Exception:
    ResNet50_Weights = None
    resnet50 = None


@dataclass
class PipelineConfig:
    image_path: Optional[str]
    image_size: int
    key_hex: str
    iv_hex: str
    use_pretrained: bool
    topk: int
    strict_float_view: bool


def get_imagenet_labels(use_pretrained: bool) -> Optional[List[str]]:
    if not use_pretrained or ResNet50_Weights is None:
        return None

    labels = ResNet50_Weights.DEFAULT.meta.get("categories")
    if isinstance(labels, list) and len(labels) > 0:
        return labels
    return None


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def hex_to_u8_tensor(hex_str: str, expected_len: int, device: str) -> torch.Tensor:
    raw = bytes.fromhex(hex_str)
    _require(
        len(raw) == expected_len,
        f"Expected {expected_len} bytes from hex, got {len(raw)} bytes.",
    )
    return torch.tensor(list(raw), dtype=torch.uint8, device=device)


def load_image_to_chw_u8(image_path: Optional[str], image_size: int) -> torch.Tensor:
    if image_path is None:
        # Synthetic image for smoke testing.
        return torch.randint(0, 256, (3, image_size, image_size), dtype=torch.uint8)

    _require(Image is not None, "Pillow is required for image loading. Please install pillow.")
    _require(os.path.exists(image_path), f"Image file not found: {image_path}")

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((image_size, image_size), resample=Image.BILINEAR)
        arr = torch.tensor(list(img.tobytes()), dtype=torch.uint8)

    # PIL returns HWC bytes. Rearrange into CHW for model input.
    hwc = arr.view(image_size, image_size, 3)
    chw = hwc.permute(2, 0, 1).contiguous()
    return chw


def aes_ctr_encrypt_cpu_u8(plain_u8: torch.Tensor, key_hex: str, iv_hex: str) -> torch.Tensor:
    _require(plain_u8.device.type == "cpu", "CPU encryption expects CPU tensor.")
    _require(plain_u8.dtype == torch.uint8, "CPU encryption expects uint8 tensor.")
    _require(AES is not None, "pycryptodome is required. Please install pycryptodome.")

    plain_bytes = bytes(plain_u8.contiguous().view(-1).tolist())
    key = bytes.fromhex(key_hex)
    iv = bytes.fromhex(iv_hex)
    _require(len(key) in (16, 24, 32), "AES key length must be 16/24/32 bytes.")
    _require(len(iv) == 16, "AES CTR IV length must be 16 bytes.")

    cipher = AES.new(key, AES.MODE_CTR, nonce=b"", initial_value=int.from_bytes(iv, "big"))
    ct = cipher.encrypt(plain_bytes)
    return torch.tensor(list(ct), dtype=torch.uint8)


def npu_aes_ctr_decrypt(cipher_u8_npu: torch.Tensor, key_u8_npu: torch.Tensor, iv_u8_npu: torch.Tensor) -> torch.Tensor:
    # CTR mode decryption equals encryption: P = C XOR KS == encrypt(C, key, iv)
    return torch_npu.aes_ctr_encrypt(cipher_u8_npu, key_u8_npu, iv_u8_npu)


def decrypted_u8_to_model_input(
    dec_u8_npu: torch.Tensor,
    chw_shape: Tuple[int, int, int],
    strict_float_view: bool,
) -> torch.Tensor:
    c, h, w = chw_shape
    expected = c * h * w
    dec_flat = dec_u8_npu.view(-1)
    _require(dec_flat.numel() == expected, f"Decrypted bytes mismatch: expected {expected}, got {dec_flat.numel()}")

    if strict_float_view:
        _require(
            dec_flat.numel() % 4 == 0,
            "float32 view requires byte length divisible by 4.",
        )
        # This is only correct when source bytes were originally float32 bit patterns.
        x = dec_flat.view(torch.float32)
        _require(x.numel() == expected // 4, "Unexpected float view shape.")
        # Caller is responsible for using matching input encoding.
        return x

    # Recommended image path: interpret decrypted bytes as uint8 pixels then cast.
    x = dec_flat.view(c, h, w).to(torch.float32) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)
    x = (x - mean) / std
    return x.unsqueeze(0)


def build_resnet50(device: str, use_pretrained: bool) -> torch.nn.Module:
    _require(resnet50 is not None, "torchvision is required for ResNet50. Please install torchvision.")

    weights = None
    if use_pretrained:
        _require(
            ResNet50_Weights is not None,
            "Your torchvision version does not expose ResNet50_Weights.",
        )
        weights = ResNet50_Weights.DEFAULT

    model = resnet50(weights=weights)
    model.eval()
    model.to(device)
    return model


def secure_infer(cfg: PipelineConfig) -> torch.Tensor:
    _require(torch.npu.is_available(), "NPU is not available.")

    # 1) CPU prepare: image -> uint8 bytes -> AES-CTR ciphertext.
    plain_chw_u8_cpu = load_image_to_chw_u8(cfg.image_path, cfg.image_size)
    cipher_u8_cpu = aes_ctr_encrypt_cpu_u8(plain_chw_u8_cpu, cfg.key_hex, cfg.iv_hex)

    # 2) H2D transfer: ciphertext + key + iv.
    device = "npu"
    cipher_u8_npu = cipher_u8_cpu.to(device, non_blocking=True)
    key_u8_npu = hex_to_u8_tensor(cfg.key_hex, expected_len=len(bytes.fromhex(cfg.key_hex)), device=device)
    iv_u8_npu = hex_to_u8_tensor(cfg.iv_hex, expected_len=16, device=device)

    # 3) NPU decrypt.
    dec_u8_npu = npu_aes_ctr_decrypt(cipher_u8_npu, key_u8_npu, iv_u8_npu)

    # 4) Type conversion.
    model_input = decrypted_u8_to_model_input(
        dec_u8_npu,
        plain_chw_u8_cpu.shape,
        strict_float_view=cfg.strict_float_view,
    )

    # 5) Model inference.
    model = build_resnet50(device=device, use_pretrained=cfg.use_pretrained)
    with torch.no_grad():
        logits = model(model_input)
        probs = torch.softmax(logits, dim=1)

    # 6) Return only probabilities to CPU; wipe and release plaintext on NPU.
    result_cpu = probs.cpu()
    dec_u8_npu.zero_()
    del dec_u8_npu, model_input, cipher_u8_npu, key_u8_npu, iv_u8_npu, plain_chw_u8_cpu, cipher_u8_cpu
    torch.npu.synchronize()
    torch.npu.empty_cache()
    return result_cpu


def topk_to_string(probs: torch.Tensor, k: int, labels: Optional[List[str]] = None) -> str:
    values, indices = torch.topk(probs[0], k=min(k, probs.shape[1]))
    lines = ["Top-k probabilities:"]
    for rank, (idx, val) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
        if labels is not None and 0 <= idx < len(labels):
            lines.append(f"  {rank}. class_id={idx}, label={labels[idx]}, prob={val:.6f}")
        else:
            lines.append(f"  {rank}. class_id={idx}, prob={val:.6f}")
    return "\n".join(lines)


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Secure NPU inference pipeline with AES-CTR")
    parser.add_argument("--image", type=str, default=None, help="Path to RGB image. If omitted, use synthetic input.")
    parser.add_argument("--image-size", type=int, default=224, help="Image resize resolution (square).")
    parser.add_argument(
        "--key-hex",
        type=str,
        default="2b7e151628aed2a6abf7158809cf4f3c",
        help="AES key in hex. 16/24/32 bytes.",
    )
    parser.add_argument(
        "--iv-hex",
        type=str,
        default="f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff",
        help="CTR initial counter (16 bytes) in hex.",
    )
    parser.add_argument("--pretrained", action="store_true", help="Use torchvision pretrained ResNet50 weights.")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable torchvision pretrained weights.")
    parser.add_argument("--topk", type=int, default=5, help="Top-k classes to print.")
    parser.add_argument(
        "--strict-float-view",
        action="store_true",
        help="Use byte-level float32 view path. Only valid if plaintext was float32 bytes.",
    )
    args = parser.parse_args()

    use_pretrained = True
    if args.no_pretrained:
        use_pretrained = False
    elif args.pretrained:
        use_pretrained = True

    return PipelineConfig(
        image_path=args.image,
        image_size=args.image_size,
        key_hex=args.key_hex,
        iv_hex=args.iv_hex,
        use_pretrained=use_pretrained,
        topk=args.topk,
        strict_float_view=args.strict_float_view,
    )


def main() -> int:
    cfg = parse_args()
    probs = secure_infer(cfg)
    labels = get_imagenet_labels(cfg.use_pretrained)
    print(topk_to_string(probs, cfg.topk, labels=labels))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
