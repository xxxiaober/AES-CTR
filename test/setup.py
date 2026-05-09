from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch_npu.utils.cpp_extension import NpuExtension
import os
import torch_npu

# 获取 CANN 环境变量路径
CANN_HOME = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
TORCH_NPU_INCLUDE = os.path.join(os.path.dirname(torch_npu.__file__), 'include')

setup(
    name='npu_custom_ops',
    version='2.8.0', # 跟随框架版本
    ext_modules=[
        NpuExtension(
            name='npu_custom_ops',
            sources=['npu_plugin.cpp'],
            include_dirs=[
                os.path.join(CANN_HOME, "include"),
                os.path.join(CANN_HOME, "opp/vendors/customize/op_api/include"),
                TORCH_NPU_INCLUDE
            ],
            library_dirs=[
                os.path.join(CANN_HOME, "lib64"),
                os.path.join(CANN_HOME, "opp/vendors/customize/op_api/lib")
            ],
            libraries=['cust_opapi', 'ascendcl', 'nnopbase'], 
            # 🚀 适配 2.8.0：要求更高的 C++ 标准以兼容现代 ATen API
            extra_compile_args=['-std=c++17', '-O3'] 
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)