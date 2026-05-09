// 🚀 修改点：引用的头文件更新为 encrypt
#include "aes_ctr_encrypt_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context) {

    AesCtrEncryptTilingData tiling;
    auto xShape = context->GetInputShape(0);
    uint32_t totalLength = xShape->GetStorageShape().GetShapeSize();

    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo); // 获取 AscendC 平台信息
    uint32_t maxCoreNum = ascendcPlatform.GetCoreNum(); // 获取当前平台的最大核数

    // --- 核心逻辑：计算多核切分 ---
    const uint32_t BLOCK_SIZE = 16;  // AES 块大小
    const uint32_t ALIGN_SIZE = 32;  // NPU 对齐要求 (必须是 32 的倍数)

    // 1. 每个核最少处理 32 字节（即 2 个 AES 块）
    uint32_t minDataPerCore = ALIGN_SIZE; 
    uint32_t usedCoreNum = maxCoreNum;

    if (totalLength <= minDataPerCore) {
        usedCoreNum = 1;
    } else {
        // 尝试平分，但要确保每个核分到的数据量是 ALIGN_SIZE 的整数倍
        uint32_t avgData = totalLength / maxCoreNum;
        uint32_t alignedAvgData = (avgData / ALIGN_SIZE) * ALIGN_SIZE; // 向下对齐到 ALIGN_SIZE 的整数倍
        
        if (alignedAvgData == 0) {
            usedCoreNum = totalLength / ALIGN_SIZE; // 如果平均分配后每核数据不足 ALIGN_SIZE，则减少核数
        } else {
            usedCoreNum = maxCoreNum;
        }
    }
    
    // 重新计算每个核负责的长度（除了最后一个核，其他核都拿同样多）
    uint32_t dataPerCore = (totalLength / usedCoreNum / ALIGN_SIZE) * ALIGN_SIZE;

    // 2. 内部流水线切片 (tileLength)，必须保持 256 字节对齐
    uint32_t tileLength = 2048;
    if (dataPerCore < tileLength) {
        tileLength = (dataPerCore / 256) * 256;
        if (tileLength == 0) tileLength = 256;
    }

    // 3. 填充 Tiling 数据
    tiling.set_totalLength(totalLength);
    tiling.set_tileLength(tileLength);
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    
    // 4. 重要：设置实际启动的核数
    context->SetBlockDim(usedCoreNum);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* x_shape = context->GetInputShape(0); // 0代表第一个输入 x
    gert::Shape* y_shape = context->GetOutputShape(0);      // 0代表第一个输出 y
    if (x_shape == nullptr || y_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *y_shape = *x_shape; // 将 x 的形状完美克隆给 y
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    // 将输出 0 (y) 的数据类型设置为与输入 0 (x) 一致
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling


// --- 算子注册信息 ---
namespace ops {

class AesCtrEncrypt : public OpDef {
public:
    explicit AesCtrEncrypt(const char* name) : OpDef(name)
    {
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
        this->AICore().AddConfig("ascend910");
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(AesCtrEncrypt);
} // namespace ops