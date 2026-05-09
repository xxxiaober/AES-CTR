import sys
from typing import List

import torch
import torch_npu

try:
    from Crypto.Cipher import AES
except Exception:
    AES = None


def to_hex(data: bytes) -> str:
    return data.hex()


def first_diff_indices(a: bytes, b: bytes, limit: int = 16) -> List[int]:
    diffs = []
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            diffs.append(i)
            if len(diffs) >= limit:
                break
    if len(a) != len(b) and len(diffs) < limit:
        diffs.append(min(len(a), len(b)))
    return diffs


def aes_ctr_cpu_reference(key: bytes, iv: bytes, plaintext: bytes) -> bytes:
    if AES is None:
        raise RuntimeError(
            "pycryptodome is not installed. Please run: pip install pycryptodome"
        )
    cipher = AES.new(key, AES.MODE_CTR, nonce=b"", initial_value=int.from_bytes(iv, "big"))
    return cipher.encrypt(plaintext)


def aes_ctr_npu(key: bytes, iv: bytes, plaintext: bytes) -> bytes:
    x = torch.tensor(list(plaintext), dtype=torch.uint8, device="npu")
    k = torch.tensor(list(key), dtype=torch.uint8, device="npu")
    v = torch.tensor(list(iv), dtype=torch.uint8, device="npu")
    y = torch_npu.aes_ctr_encrypt(x, k, v)
    return bytes(y.cpu().tolist())


def main() -> int:
    print("torch:", torch.__version__)
    print("torch_npu:", torch_npu.__version__)
    print("npu device count:", torch.npu.device_count())
    print("has torch_npu.aes_ctr_encrypt:", hasattr(torch_npu, "aes_ctr_encrypt"))

    key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
    iv = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff")
    plaintext = bytes(range(64))

    print("key:", to_hex(key))
    print("iv :", to_hex(iv))
    print("pt :", to_hex(plaintext))

    npu_ct = aes_ctr_npu(key, iv, plaintext)
    print("npu ct:", to_hex(npu_ct))

    if AES is None:
        print("WARN: pycryptodome not installed, skip CPU reference compare.")
        return 0

    cpu_ct = aes_ctr_cpu_reference(key, iv, plaintext)
    print("cpu ct:", to_hex(cpu_ct))

    if npu_ct != cpu_ct:
        diffs = first_diff_indices(npu_ct, cpu_ct)
        print("FAIL: NPU result does not match CPU AES-CTR reference")
        print("first diff indices:", diffs)
        return 1

    # CTR mode encryption is symmetric; encrypting ciphertext with same key/iv restores plaintext.
    npu_rt = aes_ctr_npu(key, iv, npu_ct)
    if npu_rt != plaintext:
        diffs = first_diff_indices(npu_rt, plaintext)
        print("FAIL: round-trip check failed")
        print("first diff indices:", diffs)
        return 2

    print("PASS: NPU output matches CPU reference and round-trip check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
