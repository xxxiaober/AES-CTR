import sys
import time

import numpy as np
import torch
import torch_npu
import torchvision.models as models
import torchvision.transforms as transforms
from Crypto.Cipher import AES
from Crypto.Util import Counter
from PIL import Image

import npu_custom_ops


def resolve_custom_op():
    if hasattr(npu_custom_ops, "aes_ctr_encrypt"):
        return npu_custom_ops.aes_ctr_encrypt, "npu_custom_ops.aes_ctr_encrypt"

    if hasattr(torch.ops, "npu_custom_ops") and hasattr(torch.ops.npu_custom_ops, "aes_ctr_encrypt"):
        return torch.ops.npu_custom_ops.aes_ctr_encrypt, "torch.ops.npu_custom_ops.aes_ctr_encrypt"

    if hasattr(torch_npu, "aes_ctr_encrypt"):
        return torch_npu.aes_ctr_encrypt, "torch_npu.aes_ctr_encrypt"

    if hasattr(torch.ops, "npu") and hasattr(torch.ops.npu, "aes_ctr_encrypt"):
        return torch.ops.npu.aes_ctr_encrypt, "torch.ops.npu.aes_ctr_encrypt"

    raise RuntimeError(
        "Cannot find custom op 'aes_ctr_encrypt'. "
        "Please ensure custom op is built and loaded correctly."
    )


print("=====================================================")
print("Zero-Trust Secure Inference Pipeline (PyTorch Custom Op)")
print("=====================================================")

if len(sys.argv) < 2:
    print("Usage: python3 final_test_pytorch.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
aes_ctr_op, op_name = resolve_custom_op()
print(f"[0] Using custom op entry: {op_name}")
print(f"[0] Using image: {image_path}")

# [1] Prepare input and do AES-CTR encryption on CPU
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
image = Image.open(image_path).convert("RGB")
plain_float_tensor = transform(image).unsqueeze(0).to(torch.float32).contiguous()
plain_bytes = plain_float_tensor.view(torch.uint8).numpy().tobytes()

key_cpu = torch.randint(0, 256, (16,), dtype=torch.uint8)
iv_cpu = torch.randint(0, 256, (16,), dtype=torch.uint8)

print(f"[1] CPU Encrypting: {image_path}")
ctr = Counter.new(128, initial_value=int.from_bytes(bytes(iv_cpu.tolist()), byteorder="big"))
cipher = AES.new(bytes(key_cpu.tolist()), AES.MODE_CTR, counter=ctr)
cipher_bytes = cipher.encrypt(plain_bytes)

cipher_tensor_npu = torch.from_numpy(np.frombuffer(cipher_bytes, dtype=np.uint8).copy()).npu()
key_npu = key_cpu.npu()
iv_npu = iv_cpu.npu()

if key_npu.numel() != 16 or iv_npu.numel() != 16:
    raise RuntimeError("AesCtrEncrypt requires key/iv length to be exactly 16 bytes.")

if cipher_tensor_npu.numel() % 32 != 0:
    raise RuntimeError(
        f"AesCtrEncrypt currently requires input length aligned to 32 bytes, got {cipher_tensor_npu.numel()}."
    )

# [2] Warm up custom op and model
print("[2] Warming up NPU model and custom op...")
weights = models.ResNet50_Weights.IMAGENET1K_V1
model = models.resnet50(weights=weights).eval().npu()

for _ in range(3):
    _ = aes_ctr_op(cipher_tensor_npu, key_npu, iv_npu)

dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32).npu()
with torch.no_grad():
    _ = model(dummy_input)

torch.npu.synchronize()
print("    -> Warm-up complete")

# [3] Steady-state benchmark
print("\n[3] NPU execution: custom op -> float cast -> ResNet50")
torch.npu.synchronize()
start_npu = time.time()

plain_uint8_npu = aes_ctr_op(cipher_tensor_npu, key_npu, iv_npu)
plain_float_npu = plain_uint8_npu.view(torch.float32).reshape(1, 3, 224, 224).contiguous()

with torch.no_grad():
    logits_npu = model(plain_float_npu)

torch.npu.synchronize()
print(f"    -> NPU pipeline took: {(time.time() - start_npu) * 1000:.2f} ms")

# [4] Verify result
prediction_idx = logits_npu.argmax(dim=1).cpu().item()
categories = weights.meta["categories"]
print(f"\nSecure inference result: [Class {prediction_idx}] {categories[prediction_idx]}")
