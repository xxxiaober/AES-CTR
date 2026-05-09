# AES-CTR 算子性能优化方案 · 第一版

> 目标芯片：华为昇腾 910ProB  
> 开发工具：Ascend CANN Toolkit 8.1.RC1  
> 分析基准：`AesCtrEncryptProject/op_kernel/aes_ctr_encrypt.cpp`  
> 文档日期：2026-04-19

---

## 一、现状分析与性能瓶颈

通过阅读当前内核代码，识别出以下主要瓶颈：

| 编号 | 问题描述 | 影响等级 |
|------|----------|----------|
| P1 | XOR 通过 `Or/And/Sub` 三条指令模拟，浪费向量槽位 | 高 |
| P2 | 密钥流生成完全标量化，未利用向量单元 | 高 |
| P3 | `SetValue` 逐字节写入密钥流缓冲区，访存效率极低 | 高 |
| P4 | S-Box 在每次构造函数中动态拷贝 256 字节 | 中 |
| P5 | Tile 大小固定 2048 字节，远未充分利用 L1 缓冲区 | 中 |
| P6 | 每个核独立执行密钥扩展（KeyExpansion），重复计算 | 中 |
| P7 | AES 轮函数循环未展开，存在循环控制开销 | 中 |
| P8 | `BUFFER_NUM=2` 流水线深度不足，CopyIn/Compute/CopyOut 重叠度低 | 低 |
| P9 | Tiling 字段过少，核内重复计算 offset 和 loopCount | 低 |

---

## 二、优化方案详述

### 优化项 1：使用 `AscendC::Xor` 替换三指令模拟 XOR

**问题定位：** `op_kernel/aes_ctr_encrypt.cpp` 第 116-124 行

当前实现利用 `(a | b) - (a & b) = a ^ b` 的数学等价关系，用 `Or` + `And` + `Sub` 三条向量指令模拟 XOR，消耗 3 个向量流水周期。

CANN 8.1.RC1 的 AscendC API 已在 910ProB 上支持 `AscendC::Xor` 内置向量指令，可直接替换。

**修改位置：** `Compute()` 函数

```cpp
// 当前代码（3条指令）
AscendC::Or(y_16, x_16, ks_16, elements16);
AscendC::And(temp_16, x_16, ks_16, elements16);
uint32_t elements32 = (currentTileLen + 3) / 4;
LocalTensor<int32_t> y_32  = yLocal.ReinterpretCast<int32_t>();
LocalTensor<int32_t> temp_32 = tempLocal.ReinterpretCast<int32_t>();
AscendC::Sub(y_32, y_32, temp_32, elements32);

// 优化后（1条指令，同时可释放 tempBuf）
uint32_t elements8 = currentTileLen;
AscendC::Xor(yLocal, xLocal, ksLocal, elements8);
```

同时可删除 `TBuf<TPosition::VECCALC> tempBuf` 的声明与初始化，节省约 2048 字节的片上缓冲区，可将节省的空间用于扩大 Tile。

**预期收益：** Compute 阶段向量指令数减少 ~67%，片上内存节省 2KB。

---

### 优化项 2：S-Box 改为编译期常量

**问题定位：** `op_kernel/aes_ctr_encrypt.cpp` 第 15-33 行

当前 S-Box 在每次 `KernelAesCtr` 构造时执行 256 次赋值循环，每个核每次调用都会重复执行。

**修改方式：** 将 S-Box 提升为文件级 `constexpr` 静态数组，编译器会将其放入只读常量区，所有核共享，无运行时初始化开销。

```cpp
// 在类定义之前，文件顶部添加
static constexpr uint8_t AES_SBOX[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
    // ... 完整 256 字节 ...
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68,
    0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

// 类成员中删除 uint8_t sbox[256]
// 所有 sbox 引用改为 AES_SBOX
```

**预期收益：** 每次算子调用节省 256 次赋值，每个核节省约 256 字节栈空间。

---

### 优化项 3：密钥流写入改用 DataCopy 批量搬运

**问题定位：** `op_kernel/aes_ctr_encrypt.cpp` 第 156-159 行

```cpp
for (int j = 0; j < 16; j++) {
    if (b * 16 + j < currentTileLen) {
        ksLocal.SetValue(b * 16 + j, keyStream[j]);  // 逐字节写入，极低效
    }
}
```

`SetValue` 是标量写操作，每次写 1 字节，16 字节需要 16 次调用。应改为先将 `keyStream[16]` 写入一个临时的 GM 对齐缓冲区，再用 `DataCopy` 批量搬运，或直接使用指针赋值配合内存对齐。

**修改方式：** 利用 AscendC 的 `SetAtomicNone` + 直接内存写入：

```cpp
// 优化后：使用 uint8_t* 直接写入本地 tensor 底层内存
uint8_t* ksPtr = reinterpret_cast<uint8_t*>(
    GetLocalPointer(ksLocal) + b * AES_BLOCK_BYTES);
uint32_t copyLen = (b * 16 + 16 <= currentTileLen) ? 16
                 : (currentTileLen - b * 16);
for (int j = 0; j < (int)copyLen; j++) ksPtr[j] = keyStream[j];
```

或者，将 `keyStream` 声明为对齐到 32 字节的局部数组，并在完整块时直接用 `DataCopy` 从标量寄存器搬运到 Local Tensor。

**预期收益：** 密钥流填充效率提升，减少标量-向量切换开销。

---

### 优化项 4：AES 轮函数循环展开

**问题定位：** `op_kernel/aes_ctr_encrypt.cpp` 第 222-225 行

```cpp
for (int r = 1; r < 10; r++) {
    SubShiftMix(state, sbox);
    AddRoundKey(state, rKeys + 16 * r);
}
```

AES-128 固定 10 轮，循环次数编译期已知，添加 `#pragma unroll` 可消除循环控制指令，并为编译器提供更多内联和指令调度空间。

**修改方式：**

```cpp
#pragma unroll
for (int r = 1; r < 10; r++) {
    SubShiftMix(state, AES_SBOX);
    AddRoundKey(state, rKeys + 16 * r);
}
```

同样对 `SubShiftMix` 和 `SubShift` 内部的 `for (int i = 0; i < 16; i++)` 循环添加 `#pragma unroll 16`。

**预期收益：** 标量 AES 计算吞吐提升约 10-20%，减少分支预测开销。

---

### 优化项 5：扩大 Tile 大小，充分利用片上缓冲区

**问题定位：** `op_host/aes_ctr_encrypt.cpp` 第 44 行

当前 `tileLength = 2048`，910ProB 每个 AI Core 的 L1 缓冲区为 256KB。

当前片上内存占用估算（BUFFER_NUM=2）：
- `inQueueX`：2 × 2048 = 4096 字节
- `outQueueY`：2 × 2048 = 4096 字节
- `keyStreamBuf`：2048 字节
- `tempBuf`：2048 字节（优化项 1 后可删除）

合计约 12KB，仅占 L1 的 4.7%，严重浪费。

**修改方式：** 在 Tiling 函数中动态查询可用缓冲区大小，计算最优 Tile：

```cpp
// op_host/aes_ctr_encrypt.cpp TilingFunc 中
uint64_t ubSize = 0;
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

// 优化项1删除tempBuf后，片上占用 = BUFFER_NUM*2*tileLen + tileLen
// 保留 50% UB 余量，防止编译器临时变量溢出
uint32_t maxTile = (uint32_t)(ubSize * 0.5) / (BUFFER_NUM * 2 + 1);
// 对齐到 256 字节（AES 块的 16 倍，且满足向量对齐）
maxTile = (maxTile / 256) * 256;
// 上限 64KB，防止单 tile 过大导致调度延迟
uint32_t tileLength = std::min(maxTile, (uint32_t)65536);
if (dataPerCore < tileLength) tileLength = (dataPerCore / 256) * 256;
if (tileLength == 0) tileLength = 256;
```

**预期收益：** Tile 从 2048 扩大到约 32768 字节，CopyIn/CopyOut 次数减少 16 倍，流水线启动开销大幅降低。

---

### 优化项 6：密钥扩展移至 Host 侧，通过 Workspace 传递

**问题定位：** `op_kernel/aes_ctr_encrypt.cpp` 第 65-66 行

每个 AI Core 在 `Init()` 中独立执行 `KeyExpansion`，生成相同的 176 字节扩展密钥。对于 N 核并行，这是 N 倍的重复计算。

**修改方式：**

1. 在 `op_host/aes_ctr_encrypt.cpp` 的 `TilingFunc` 中申请 Workspace：

```cpp
// 申请 176 字节 workspace 存放扩展密钥
context->SetTilingKey(1);
uint64_t workspaceSize = AES_EXPANDED_KEY_BYTES; // 176
context->SetWorkspaceSize(0, workspaceSize);
```

2. 在 Tiling 阶段（或通过 aclnn 的 GetWorkspaceSize 阶段）在 CPU 上完成密钥扩展，将结果写入 workspace。

3. 内核侧直接从 `workspace` GM 地址读取扩展密钥，跳过 `KeyExpansion` 调用。

> **注意：** 此优化需要在 aclnn 接口层（`aclnn_aes_ctr_encrypt.cpp`）配合修改，将密钥扩展逻辑移至 `GetWorkspaceSize` 阶段执行。实施前需确认 CANN 8.1.RC1 的 workspace 写入时机是否支持此模式。

**预期收益：** 消除每核 KeyExpansion 的重复计算，对核数多（如 24 核）场景收益显著。

---

### 优化项 7：扩充 Tiling 字段，消除核内重复计算

**问题定位：** `op_host/aes_ctr_encrypt_tiling.h` 与 `op_kernel/aes_ctr_encrypt.cpp` 第 40-54 行

当前 Tiling 只传递 `totalLength` 和 `tileLength`，每个核在 `Init()` 中重新计算 `dataPerCore`、`offset`、`loopCount`，且计算逻辑与 Host 侧 Tiling 函数存在潜在不一致风险。

**修改方式：** 扩充 Tiling 结构体：

```cpp
// aes_ctr_encrypt_tiling.h
BEGIN_TILING_DATA_DEF(AesCtrEncryptTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, dataPerCore);    // 新增：每核数据量（除最后一核）
  TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);    // 新增：实际使用核数
END_TILING_DATA_DEF;
```

内核侧 `Init()` 直接使用 `tilingData.dataPerCore` 和 `tilingData.usedCoreNum`，无需重新计算。

**预期收益：** 消除核内冗余计算，降低 Init 阶段延迟，同时提高 Host/Device 逻辑一致性。

---

## 三、优化优先级与实施路线图

| 优先级 | 优化项 | 改动范围 | 预期收益 | 风险 |
|--------|--------|----------|----------|------|
| P0 | 优化项 1：Xor 指令替换 | kernel 仅改 Compute() | 向量指令 -67% | 低，需确认 CANN 8.1 Xor API |
| P0 | 优化项 2：S-Box 常量化 | kernel 构造函数 | 消除初始化开销 | 极低 |
| P1 | 优化项 4：循环展开 | kernel AES 函数 | 标量吞吐 +10~20% | 低 |
| P1 | 优化项 5：扩大 Tile | host tiling 函数 | 流水线效率大幅提升 | 中，需测试 UB 不溢出 |
| P2 | 优化项 3：批量写密钥流 | kernel GenerateKeyStream | 减少标量访存 | 中 |
| P2 | 优化项 7：扩充 Tiling 字段 | host + kernel + tiling.h | 消除冗余计算 | 低 |
| P3 | 优化项 6：密钥扩展移 Host | host + kernel + aclnn | 多核场景收益显著 | 中高，需验证 workspace 时序 |

**建议实施顺序：**

```
第一步（无风险，立即可做）：优化项 1 + 2 + 4
第二步（需测试验证）：优化项 5 + 7
第三步（需架构评审）：优化项 3 + 6
```

---

## 四、验证方法

每步优化后，使用现有测试套件验证：

1. **正确性验证：** 运行 `test/final_verify.py`，对比 NPU 输出与 CPU 参考实现
2. **性能基准：** 运行 `test/npu_benchmark.py`（50 次预热 + 1000 次测试），记录吞吐量（MB/s）
3. **Profiling：** 使用 `test/npu_profiler.py` 或 CANN Profiling 工具查看各阶段耗时分布
4. **内存检查：** 编译时关注 UB 溢出警告（CANN 编译器会在 UB 超限时报错）

---

## 五、参考资料

- Ascend C 编程指南（CANN 8.1.RC1）：向量指令集、TBuf/TQue 使用规范
- 昇腾 910ProB 硬件规格：AI Core 数量 24，UB 大小 256KB/Core，向量宽度 256bit
- AscendC API：`AscendC::Xor`、`DataCopy`、`platform_ascendc::GetCoreMemSize`
