import os
import sys

import torch
import torch_npu 
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time
import npu_custom_ops
from Crypto.Cipher import AES
from Crypto.Util import Counter

print("=====================================================")
print("🛡️  Zero-Trust Secure Inference Pipeline (Final Profiling) 🛡️")
print("=====================================================")

IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "test_image.png"

# ==========================================
# [0] 准备数据与 CPU 加密
# ==========================================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image = Image.open(IMAGE_PATH).convert('RGB')
plain_float_tensor = transform(image).unsqueeze(0).to(torch.float32).contiguous()
plain_bytes = plain_float_tensor.view(torch.uint8).numpy().tobytes()

key_cpu = torch.randint(0, 256, (16,), dtype=torch.uint8)
iv_cpu  = torch.randint(0, 256, (16,), dtype=torch.uint8)

print(f"[1] CPU Encrypting: {IMAGE_PATH}")
ctr = Counter.new(128, initial_value=int.from_bytes(bytes(iv_cpu.tolist()), byteorder='big'))
cipher = AES.new(bytes(key_cpu.tolist()), AES.MODE_CTR, counter=ctr)
cipher_bytes = cipher.encrypt(plain_bytes)

# 将数据下发到 NPU
cipher_tensor_npu = torch.from_numpy(np.frombuffer(cipher_bytes, dtype=np.uint8).copy()).npu()
key_npu = key_cpu.npu()
iv_npu = iv_cpu.npu()

# ==========================================
# [1] 终极预热阶段 (Warm-up) - 剥离所有冷启动税！
# ==========================================
print("[2] Warming up NPU Model AND Custom Operator...")
weights = models.ResNet50_Weights.IMAGENET1K_V1
model = models.resnet50(weights=weights).eval().npu()

# 1. 预热自定义算子 (消耗掉算子加载的 4 秒)
for _ in range(3):
    _ = npu_custom_ops.aes_ctr_encrypt(cipher_tensor_npu, key_npu, iv_npu)

# 2. 预热 ResNet50 模型 (消耗掉图编译的 13 秒)
dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32).npu()
with torch.no_grad():
    _ = model(dummy_input)

torch.npu.synchronize()
print("    -> All NPU Warm-ups Complete! Engine Ready.")

# ==========================================
# [2] 真实稳态测速 (Benchmark)
# ==========================================
print("\n[3] NPU Execution: Custom Decrypt(Encrypt) -> Float32 Cast -> ResNet50")
torch.npu.synchronize()
start_npu = time.time()

# A. 高速硬件解密 (~3 ms)
plain_uint8_npu = npu_custom_ops.aes_ctr_encrypt(cipher_tensor_npu, key_npu, iv_npu)

# B. 零拷贝映射，并强制内存连续 (防止 ResNet 触发动态步长重编译)
plain_float_npu = plain_uint8_npu.view(torch.float32).reshape(1, 3, 224, 224).contiguous()

# C. 稳态模型推理 (~5-10 ms)
with torch.no_grad():
    logits_npu = model(plain_float_npu)

torch.npu.synchronize()
# 奇迹时刻！
print(f"    -> NPU Pipeline Took: {(time.time() - start_npu)*1000:.2f} ms")

# ==========================================
# [3] 结果验证
# ==========================================
prediction_idx = logits_npu.argmax(dim=1).cpu().item()
categories = weights.meta["categories"]
print(f"\n🌟 Secure Inference Result: [Class {prediction_idx}] {categories[prediction_idx]}")