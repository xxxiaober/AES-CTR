#include <iostream>
#include <vector>
#include "acl/acl.h"
// 引入你的自定义算子 ACLNN 接口头文件
#include "aclnn_aes_ctr_encrypt.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            std::cerr << "Error at [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; \
            return_expr; \
        } \
    } while (0)

// 辅助函数：根据 Shape 和 Device 内存指针，创建一个 aclTensor
aclTensor* CreateAclTensor(const std::vector<int64_t>& shape, void* devPtr, aclDataType dataType) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return aclCreateTensor(shape.data(), shape.size(), dataType,
                           strides.data(), 0, ACL_FORMAT_ND,
                           shape.data(), shape.size(), devPtr);
}

int main() {
    std::cout << "[1] 初始化 NPU 设备..." << std::endl;
    CHECK_RET(aclInit(nullptr) == ACL_SUCCESS, return -1);
    int32_t deviceId = 0;
    CHECK_RET(aclrtSetDevice(deviceId) == ACL_SUCCESS, return -1);
    aclrtContext context;
    CHECK_RET(aclrtCreateContext(&context, deviceId) == ACL_SUCCESS, return -1);
    aclrtStream stream;
    CHECK_RET(aclrtCreateStream(&stream) == ACL_SUCCESS, return -1);

    std::cout << "[2] 准备数据与显存..." << std::endl;
    // 模拟 1024 字节数据
    const int64_t DATA_SIZE = 1024;
    std::vector<int64_t> shapeX = {DATA_SIZE};
    std::vector<int64_t> shapeKey = {16};
    std::vector<int64_t> shapeIv = {16};

    // 在 Host 侧构造假数据
    std::vector<uint8_t> hostX(DATA_SIZE, 1);
    std::vector<uint8_t> hostKey(16, 2);
    std::vector<uint8_t> hostIv(16, 3);
    std::vector<uint8_t> hostY(DATA_SIZE, 0);

    // 在 Device 侧申请显存
    void *devX = nullptr, *devKey = nullptr, *devIv = nullptr, *devY = nullptr;
    CHECK_RET(aclrtMalloc(&devX, DATA_SIZE, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtMalloc(&devKey, 16, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtMalloc(&devIv, 16, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtMalloc(&devY, DATA_SIZE, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return -1);

    // 将 Host 数据拷贝到 Device 显存 (H2D)
    CHECK_RET(aclrtMemcpy(devX, DATA_SIZE, hostX.data(), DATA_SIZE, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtMemcpy(devKey, 16, hostKey.data(), 16, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtMemcpy(devIv, 16, hostIv.data(), 16, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS, return -1);

    std::cout << "[3] 构建 aclTensor..." << std::endl;
    aclTensor* tensorX = CreateAclTensor(shapeX, devX, ACL_UINT8);
    aclTensor* tensorKey = CreateAclTensor(shapeKey, devKey, ACL_UINT8);
    aclTensor* tensorIv = CreateAclTensor(shapeIv, devIv, ACL_UINT8);
    aclTensor* tensorY = CreateAclTensor(shapeX, devY, ACL_UINT8);

    std::cout << "[4] 执行 ACLNN 算子 (两段式调用)..." << std::endl;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 步骤 A: 获取 Workspace 大小
    auto ret = aclnnAesCtrEncryptGetWorkspaceSize(tensorX, tensorKey, tensorIv, tensorY, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, return -1);

    // 步骤 B: 申请 Workspace 显存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        CHECK_RET(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return -1);
    }

    // 步骤 C: 真正执行算子
    ret = aclnnAesCtrEncrypt(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, return -1);

    // 同步等待 NPU 执行完毕
    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, return -1);
    std::cout << "    -> NPU 执行成功！" << std::endl;

    std::cout << "[5] 获取结果并清理资源..." << std::endl;
    // 将结果从 Device 拷贝回 Host (D2H)
    CHECK_RET(aclrtMemcpy(hostY.data(), DATA_SIZE, devY, DATA_SIZE, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS, return -1);

    // 打印前 5 个字节看看结果
    std::cout << "    -> Output first 5 bytes: ";
    for (int i = 0; i < 5; i++) printf("%02x ", hostY[i]);
    std::cout << std::endl;

    // 释放资源 (在实际工程中建议用 RAII 封装)
    aclDestroyTensor(tensorX);
    aclDestroyTensor(tensorKey);
    aclDestroyTensor(tensorIv);
    aclDestroyTensor(tensorY);
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtFree(devX); aclrtFree(devKey); aclrtFree(devIv); aclrtFree(devY);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    std::cout << "🎉 纯 C++ ACLNN 测试全流程结束！" << std::endl;
    return 0;
}