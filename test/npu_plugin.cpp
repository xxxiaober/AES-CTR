#include <torch/extension.h>
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/framework/OpCommand.h"

at::Tensor aes_ctr_encrypt_npu(const at::Tensor& x, const at::Tensor& key, const at::Tensor& iv) {
    // 🚀 终极修复点：适配 2.8.0 的全新 c10_npu 命名空间！
    c10_npu::NPUGuard guard(x.device());

    auto x_contig = x.contiguous();
    auto key_contig = key.contiguous();
    auto iv_contig = iv.contiguous();

    // 申请输出张量
    at::Tensor y = at::empty_like(x_contig);

    // 调用底层算子
    at_npu::native::OpCommand cmd;
    cmd.Name("AesCtrEncrypt")
       .Input(x_contig)
       .Input(key_contig)
       .Input(iv_contig)
       .Output(y)
       .Run();
    
    return y;
}

// 绑定给 Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("aes_ctr_encrypt", &aes_ctr_encrypt_npu, "NPU AES CTR Encrypt for PyTorch 2.8.0");
}