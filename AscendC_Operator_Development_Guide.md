# Ascend C 自定义算子开发实战指南

> 本文档结合 `AesCtrEncryptProject` 实例项目与官方 Ascend C 算子开发教程，系统梳理从算子开发到 PyTorch 集成运行的完整流程。

---

## 一、Ascend C 算子开发概述

### 1.1 什么是 Ascend C

Ascend C 是 CANN（Compute Architecture for Neural Networks）推出的面向算子开发的编程语言，原生支持 C/C++ 标准规范。基于 Ascend C 编写的算子程序，通过编译器编译和运行时调度，运行在昇腾 AI 处理器（NPU）上。

Ascend C 提供多层级的 API 体系：

| API 层级 | 说明 |
|---------|------|
| **语言扩展层 C API** | 开放芯片完备编程能力，基于指针编程 |
| **基础 API** | 基于 Tensor 的 C++ 类库 API，单指令级抽象 |
| **高阶 API** | 封装单核公共算法（卷积、矩阵运算等） |
| **算子模板库** | 提供算子完整实现参考，简化 Tiling 开发 |
| **Python 前端 (PyAsc)** | 基于 Python 接口开发高性能算子 |

### 1.2 算子开发核心概念

- **Host 侧**：运行在 CPU 上的逻辑，包括算子注册、Tiling 策略、Shape/数据类型推导
- **Device 侧**：运行在 NPU 上的核函数，负责实际的并行计算
- **Tiling**：将大数据切分为适合单核处理的切片（Tile），是多核并行和流水线的核心
- **GM (Global Memory)**：NPU 全局内存，用于 Host 与 Device 的数据交互
- **LB/UB (Local Buffer / Unified Buffer)**：NPU 片上高速缓存，算子计算在 UB 中进行

---

## 二、项目目录结构

一个标准的 Ascend C 自定义算子项目结构如下（以 `AesCtrEncryptProject` 为例）：

```
AesCtrEncryptProject/
├── CMakeLists.txt              # 根 CMake 配置
├── CMakePresets.json           # CMake 预设参数
├── build.sh                    # 一键构建脚本
├── cmake/                      # 构建工具链与函数库
│   ├── config.cmake
│   ├── func.cmake
│   ├── intf.cmake
│   └── util/                   # 辅助脚本（版本生成、打包等）
├── framework/                  # 框架适配插件（可选）
│   └── tf_plugin/              # TensorFlow 解析插件
│       ├── CMakeLists.txt
│       └── tensorflow_aes_ctr_encrypt_plugin.cc
├── op_host/                    # Host 侧代码
│   ├── CMakeLists.txt
│   ├── aes_ctr_encrypt.cpp     # 算子注册、Tiling、Shape 推导
│   └── aes_ctr_encrypt_tiling.h # TilingData 结构定义
├── op_kernel/                  # Device 侧代码
│   ├── CMakeLists.txt
│   └── aes_ctr_encrypt.cpp     # 核函数实现
├── scripts/                    # 安装/卸载脚本
│   ├── install.sh
│   ├── upgrade.sh
│   └── help.info
└── build_out/                  # 构建输出目录
```

---

## 三、算子开发详细步骤

### 步骤 1：定义 TilingData 结构（Host 侧）

TilingData 是 Host 侧与 Device 侧通信的桥梁，用于传递切分参数。

**文件**：`op_host/aes_ctr_encrypt_tiling.h`

```cpp
#ifndef AES_CTR_ENCRYPT_TILING_H
#define AES_CTR_ENCRYPT_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AesCtrEncryptTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);  // 总数据长度
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);   // 每个 Tile 的长度
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AesCtrEncrypt, AesCtrEncryptTilingData)
}
#endif
```

> **要点**：`TILING_DATA_FIELD_DEF` 定义的数据字段会按顺序打包成二进制，通过 GM 传递给 Device 侧的 `Init` 函数。

---

### 步骤 2：编写 Host 侧算子注册与 Tiling 策略

**文件**：`op_host/aes_ctr_encrypt.cpp`

#### 2.1 Tiling 函数

Tiling 函数决定如何将数据分配到多个 AI Core 上并行处理。

```cpp
#include "aes_ctr_encrypt_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    AesCtrEncryptTilingData tiling;
    auto xShape = context->GetInputShape(0);
    uint32_t totalLength = xShape->GetStorageShape().GetShapeSize();

    // 获取平台信息（最大核数）
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t maxCoreNum = ascendcPlatform.GetCoreNum();

    // --- 核心逻辑：计算多核切分 ---
    const uint32_t BLOCK_SIZE = 16;  // AES 块大小
    const uint32_t ALIGN_SIZE = 32;  // NPU 对齐要求

    uint32_t minDataPerCore = ALIGN_SIZE;
    uint32_t usedCoreNum = maxCoreNum;

    if (totalLength <= minDataPerCore) {
        usedCoreNum = 1;
    } else {
        uint32_t avgData = totalLength / maxCoreNum;
        uint32_t alignedAvgData = (avgData / ALIGN_SIZE) * ALIGN_SIZE;
        if (alignedAvgData == 0) {
            usedCoreNum = totalLength / ALIGN_SIZE;
        } else {
            usedCoreNum = maxCoreNum;
        }
    }

    uint32_t dataPerCore = (totalLength / usedCoreNum / ALIGN_SIZE) * ALIGN_SIZE;
    uint32_t tileLength = 2048;
    if (dataPerCore < tileLength) {
        tileLength = (dataPerCore / 256) * 256;
        if (tileLength == 0) tileLength = 256;
    }

    // 填充 Tiling 数据
    tiling.set_totalLength(totalLength);
    tiling.set_tileLength(tileLength);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), 
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    
    // 设置实际启动的核数
    context->SetBlockDim(usedCoreNum);

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
```

#### 2.2 Shape 与数据类型推导

```cpp
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    if (x_shape == nullptr || y_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *y_shape = *x_shape;  // 输出形状与输入相同
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
```

#### 2.3 算子注册

```cpp
namespace ops {
class AesCtrEncrypt : public OpDef {
public:
    explicit AesCtrEncrypt(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND});
    
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND});

        this->Input("iv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND});

        this->SetInferShape(optiling::InferShape);
        this->SetInferDataType(optiling::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910");   // 支持的芯片型号
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(AesCtrEncrypt);
} // namespace ops
```

> **关键字段说明**：
> - `Input/Output`：定义算子的输入输出，需指定名称、是否必填、数据类型、数据格式
> - `SetInferShape/SetInferDataType`：注册形状和数据类型推导函数
> - `SetTiling`：注册 Tiling 策略函数
> - `AddConfig`：声明算子支持的 AI 处理器型号

---

### 步骤 3：编写 Device 侧核函数

**文件**：`op_kernel/aes_ctr_encrypt.cpp`

Device 侧是算子的实际执行体，核心遵循 **CopyIn -> Compute -> CopyOut** 的流水线范式。

#### 3.1 核函数类基本结构

```cpp
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;  // 双缓冲，实现流水线并行

class KernelAesCtr {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR key, GM_ADDR iv, GM_ADDR y, 
                                 uint32_t totalLen, uint32_t tileLen) {
        this->totalLength = totalLen;
        this->tileLength = tileLen;

        // 多核分工：根据 blockIdx 计算每个核处理的数据偏移和长度
        uint32_t blockIdx = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();
        uint32_t dataPerCore = (totalLength / blockNum / 32) * 32;
        uint32_t offset = blockIdx * dataPerCore;

        this->currentCoreDataLength = dataPerCore;
        if (blockIdx == blockNum - 1) {
            this->currentCoreDataLength = totalLength - offset;  // 最后一个核处理余量
        }
        
        this->loopCount = this->currentCoreDataLength / tileLength;
        if (this->currentCoreDataLength % tileLength != 0) this->loopCount++;

        // 设置 Global Memory 访问基地址
        xGm.SetGlobalBuffer((__gm__ uint8_t*)x + offset, this->currentCoreDataLength);
        yGm.SetGlobalBuffer((__gm__ uint8_t*)y + offset, this->currentCoreDataLength);

        // 初始化 Queue 和 Buffer（用于流水线）
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength);
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength);
        pipe.InitBuffer(keyStreamBuf, tileLength);
        pipe.InitBuffer(tempBuf, tileLength);

        // 读取 key/iv 到本地并做密钥扩展
        uint8_t localKey[16];
        for (int i = 0; i < 16; i++) localKey[i] = ((__gm__ uint8_t*)key)[i];
        KeyExpansion(localKey, expandedKey, sbox);
        for (int i = 0; i < 16; i++) baseIv[i] = ((__gm__ uint8_t*)iv)[i];
        BigEndAdd(baseIv, offset / 16);
    }

    __aicore__ inline void Process() {
        for (int32_t i = 0; i < this->loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    // ... CopyIn, Compute, CopyOut, 算法实现 ...
    
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<TPosition::VECCALC> keyStreamBuf;
    TBuf<TPosition::VECCALC> tempBuf;
    
    GlobalTensor<uint8_t> xGm, yGm;
    uint8_t sbox[256], expandedKey[176], baseIv[16];
    uint32_t totalLength, tileLength, currentCoreDataLength, loopCount;
};
```

#### 3.2 全局入口函数

```cpp
extern "C" __global__ __aicore__ void aes_ctr_encrypt(
    GM_ADDR x, GM_ADDR key, GM_ADDR iv, GM_ADDR y, 
    GM_ADDR workspace, GM_ADDR tiling) {
    
    GET_TILING_DATA(tilingData, tiling);  // 解析 Host 传递的 Tiling 数据
    KernelAesCtr op;
    op.Init(x, key, iv, y, tilingData.totalLength, tilingData.tileLength);
    op.Process();
}
```

> **Device 侧编程要点**：
> 1. `__aicore__` 修饰符表示函数运行在 NPU AI Core 上
> 2. `__gm__` 指针指向 Global Memory
> 3. `TPipe`/`TQue`/`TBuf` 是 Ascend C 的内存管理和流水线原语
> 4. `DataCopy` 用于 GM 与 UB/LB 之间的数据传输
> 5. 计算 API（如 `AscendC::Or`, `AscendC::And`, `AscendC::Sub`）直接在 Tensor 上执行向量化操作

---

### 步骤 4：配置 CMake 构建系统

#### 4.1 根 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.16.0)
project(opp)

include(cmake/config.cmake)
include(cmake/func.cmake)
include(cmake/intf.cmake)

# 根据目录自动添加子模块
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/framework)
    add_subdirectory(framework)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op_host)
    add_subdirectory(op_host)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)
    add_subdirectory(op_kernel)
endif()
```

#### 4.2 CMakePresets.json（关键配置）

```json
{
    "version": 1,
    "configurePresets": [
        {
            "name": "default",
            "binaryDir": "${sourceDir}/build_out",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Release",
                "ENABLE_SOURCE_PACKAGE": "True",
                "ENABLE_BINARY_PACKAGE": "True",
                "ASCEND_COMPUTE_UNIT": "ascend910",
                "vendor_name": "customize",
                "ASCEND_CANN_PACKAGE_PATH": "/usr/local/Ascend/ascend-toolkit/latest",
                "ENABLE_CROSS_COMPILE": "False",
                "ASCEND_PACK_SHARED_LIBRARY": "False"
            }
        }
    ]
}
```

> **关键参数**：
> - `vendor_name`：自定义算子厂商名称，安装后会存放在 `opp/vendors/{vendor_name}/` 下
> - `ASCEND_COMPUTE_UNIT`：目标芯片架构（ascend910 / ascend910b 等）
> - `ENABLE_BINARY_PACKAGE`：是否生成 `.run` 安装包
> - `ASCEND_PACK_SHARED_LIBRARY`：是否打包为单一动态库（二进制发布场景）

---

### 步骤 5：构建算子包

```bash
# 进入项目目录
cd AesCtrEncryptProject

# 执行构建脚本
bash build.sh
```

构建成功后，在 `build_out/` 目录下会生成：

```
build_out/
├── op_host/
│   ├── libcust_opapi.so          # ACLNN API 动态库
│   ├── libcust_opmaster_rt2.0.so # Tiling 库
│   └── libcust_opsproto_rt2.0.so # 算子原型库
├── op_kernel/
│   └── AesCtrEncrypt_ascend910.o # 核函数二进制
└── custom_opp_openEuler_aarch64.run   # 一键安装包
```

---

## 四、安装算子到 CANN 环境

### 方式一：使用 .run 安装包（推荐）

```bash
# 进入构建输出目录
cd AesCtrEncryptProject/build_out

# 执行安装脚本
bash custom_opp_openEuler_aarch64.run --install-path=/usr/local/Ascend/opp

# 或安装到默认路径（需要 ASCEND_OPP_PATH 环境变量）
bash custom_opp_openEuler_aarch64.run
```

安装完成后，算子文件会被部署到：

```
/usr/local/Ascend/opp/vendors/customize/
├── op_api/
│   ├── include/
│   │   ├── aclnn_aes_ctr_encrypt.h    # ACLNN 接口头文件
│   │   └── op_proto.h
│   └── lib/
│       └── libcust_opapi.so           # ACLNN 动态库
├── op_proto/
│   └── lib/
│       └── libcust_opsproto_rt2.0.so  # 算子原型库
├── op_impl/
│   └── ai_core/
│       └── tbe/
│           └── op_tiling/
│               └── libcust_opmaster_rt2.0.so  # Tiling 库
└── framework/
    └── tensorflow/
        └── libcust_tf_parsers.so      # 框架解析插件
```

### 方式二：手动拷贝 so 文件（开发调试）

```bash
# 设置环境变量
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/opp/vendors/customize
export LD_LIBRARY_PATH=${ASCEND_CUSTOM_OPP_PATH}/op_api/lib:$LD_LIBRARY_PATH

# 手动复制编译产物（如需要）
cp build_out/op_host/libcust_opapi.so ${ASCEND_CUSTOM_OPP_PATH}/op_api/lib/
cp build_out/op_host/libcust_opmaster_rt2.0.so ${ASCEND_CUSTOM_OPP_PATH}/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/
```

---

## 五、PyTorch 集成与运行

### 5.1 编写 PyTorch C++ 扩展插件

为了将自定义算子接入 `torch_npu`，需要编写一个 Pybind11 扩展，调用 ACLNN 接口。

**文件**：`test/npu_plugin.cpp`

```cpp
#include <torch/extension.h>
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/framework/OpCommand.h"

at::Tensor aes_ctr_encrypt_npu(const at::Tensor& x, const at::Tensor& key, const at::Tensor& iv) {
    // 设备守卫：确保在当前 NPU 设备上执行
    c10_npu::NPUGuard guard(x.device());

    auto x_contig = x.contiguous();
    auto key_contig = key.contiguous();
    auto iv_contig = iv.contiguous();

    // 申请输出张量
    at::Tensor y = at::empty_like(x_contig);

    // 使用 OpCommand 调用底层算子
    at_npu::native::OpCommand cmd;
    cmd.Name("AesCtrEncrypt")
       .Input(x_contig)
       .Input(key_contig)
       .Input(iv_contig)
       .Output(y)
       .Run();
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("aes_ctr_encrypt", &aes_ctr_encrypt_npu, "NPU AES CTR Encrypt");
}
```

### 5.2 编写 setup.py 构建 Python 包

**文件**：`test/setup.py`

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch_npu.utils.cpp_extension import NpuExtension
import os
import torch_npu

CANN_HOME = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
TORCH_NPU_INCLUDE = os.path.join(os.path.dirname(torch_npu.__file__), 'include')

setup(
    name='npu_custom_ops',
    version='2.8.0',
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
            extra_compile_args=['-std=c++17', '-O3']
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

### 5.3 编译安装 Python 扩展

```bash
cd test

# 编译并安装
python3 setup.py build_ext --inplace

# 或安装到 site-packages
python3 setup.py install
```

编译成功后，会生成 `npu_custom_ops.cpython-311-aarch64-linux-gnu.so`。

### 5.4 在 PyTorch 中调用自定义算子

```python
import torch
import torch_npu
import npu_custom_ops

# 准备数据（需为 uint8 类型，长度对齐到 32 字节）
cipher_tensor_npu = torch.from_numpy(cipher_bytes).npu()
key_npu = key_cpu.npu()
iv_npu = iv_cpu.npu()

# 调用自定义算子
plain_uint8_npu = npu_custom_ops.aes_ctr_encrypt(cipher_tensor_npu, key_npu, iv_npu)

# 转换为 float 继续后续模型推理
plain_float_npu = plain_uint8_npu.view(torch.float32).reshape(1, 3, 224, 224).contiguous()
```

### 5.5 完整推理流程示例

```python
import torch
import torch_npu
import torchvision.models as models
import npu_custom_ops

# 1. 加载模型到 NPU
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).eval().npu()

# 2. 准备加密数据（假设已从 CPU 侧加密）
cipher_tensor = ...  # shape: [N], dtype: uint8, device: npu
key = ...            # shape: [16], dtype: uint8, device: npu
iv = ...             # shape: [16], dtype: uint8, device: npu

# 3. NPU 侧硬件解密
plain_uint8 = npu_custom_ops.aes_ctr_encrypt(cipher_tensor, key, iv)

# 4. 类型转换后输入模型
input_tensor = plain_uint8.view(torch.float32).reshape(1, 3, 224, 224).contiguous()

# 5. 模型推理
with torch.no_grad():
    logits = model(input_tensor)

print(f"Prediction: {logits.argmax(dim=1).item()}")
```

---

## 六、纯 C++ ACLNN 接口调用（可选）

除了 PyTorch 封装，也可以直接在 C++ 应用中使用 ACLNN 接口调用自定义算子。

### 6.1 两段式调用接口

Ascend C 算子编译后会自动生成 ACLNN 接口，命名规范为 `aclnn{OpName}`：

```cpp
#include "aclnn_aes_ctr_encrypt.h"

// 步骤 A: 获取 Workspace 大小
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
aclnnAesCtrEncryptGetWorkspaceSize(tensorX, tensorKey, tensorIv, tensorY, 
                                   &workspaceSize, &executor);

// 步骤 B: 申请 Workspace 并执行
void* workspaceAddr = nullptr;
if (workspaceSize > 0) {
    aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}
aclnnAesCtrEncrypt(workspaceAddr, workspaceSize, executor, stream);
aclrtSynchronizeStream(stream);
```

### 6.2 CMake 编译配置

```cmake
cmake_minimum_required(VERSION 3.14)
project(AclnnAesTest)
set(CMAKE_CXX_STANDARD 17)

set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
include_directories(
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/include
)
link_directories(
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib
)

add_executable(aclnn_aes_test main.cpp)
target_link_libraries(aclnn_aes_test ascendcl nnopbase cust_opapi)
```

---

## 七、环境检查清单

在开发和运行前，请确保以下环境变量已正确配置：

```bash
# 1. CANN 基础环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 自定义算子路径（安装 .run 后会自动生成 set_env.bash）
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/opp/vendors/customize
export LD_LIBRARY_PATH=${ASCEND_CUSTOM_OPP_PATH}/op_api/lib:$LD_LIBRARY_PATH

# 3. PyTorch / torch_npu 环境
export PYTHONPATH=/path/to/torch_npu:$PYTHONPATH

# 4. 验证算子是否安装成功
ls $ASCEND_CUSTOM_OPP_PATH/op_api/lib/libcust_opapi.so
ls $ASCEND_CUSTOM_OPP_PATH/op_api/include/aclnn_aes_ctr_encrypt.h
```

---

## 八、常见问题与调试技巧

| 问题现象 | 可能原因 | 解决方案 |
|---------|---------|---------|
| `Op AesCtrEncrypt is not found` | 算子未安装或 LD_LIBRARY_PATH 未设置 | 执行 `.run` 安装包，并导出 `LD_LIBRARY_PATH` |
| `Tiling failed` / 核启动异常 | Tiling 参数计算错误（如未对齐） | 检查 ALIGN_SIZE 和 SetBlockDim 逻辑 |
| `Segmentation fault` in Device | UB 越界访问或内存未初始化 | 检查 `InitBuffer` 大小和 `DataCopy` 长度 |
| 编译报错 `c10_npu not found` | torch_npu 版本不匹配 | 确保头文件路径包含 `torch_npu/include` |
| Python 导入 `.so` 失败 | C++ ABI 不兼容或 CANN 库未链接 | 使用 `NpuExtension` 编译，检查 `libraries` 参数 |
| 精度不一致 | 数据类型转换或字节序问题 | 检查 `uint8`/`float32` 转换逻辑，确认大端序处理 |

---

## 九、参考文档与资源

1. **官方 Ascend C 算子开发文档**
   - https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_map_10_0002.html

2. **CANN 社区版下载**
   - https://www.hiascend.com/software/cann/community

3. **torch_npu GitHub**
   - https://github.com/Ascend/pytorch

4. **本实例项目**
   - `AesCtrEncryptProject/`：完整算子实现
   - `test/`：PyTorch 集成测试与性能分析
   - `aclnn_test/`：纯 C++ ACLNN 调用示例

---

## 十、总结

Ascend C 自定义算子开发的完整流程可归纳为：

```
┌─────────────────────────────────────────────────────────────┐
│  1. 定义 TilingData（Host）                                  │
│  2. 编写 Tiling / InferShape / 算子注册（Host）              │
│  3. 编写核函数 Init / Process / CopyIn / Compute / CopyOut   │
│  4. 配置 CMake 并执行 build.sh 编译                          │
│  5. 运行 .run 安装包部署到 CANN 环境                         │
│  6. 编写 PyTorch C++ 扩展（npu_plugin.cpp + setup.py）      │
│  7. python3 setup.py build_ext --inplace 编译 Python 模块   │
│  8. import npu_custom_ops 并在 PyTorch 中调用                │
└─────────────────────────────────────────────────────────────┘
```

通过遵循以上步骤，你可以将任意自定义算法高效地部署到昇腾 NPU 上，并与 PyTorch 训练/推理流水线无缝集成。
