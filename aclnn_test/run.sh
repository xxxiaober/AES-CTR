#!/bin/bash
set -e

# 1. 导入 CANN 环境变量 (如果你的安装路径不同，请修改)
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 导出自定义算子动态库路径，防止运行期报找不到 libcust_opapi.so
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib:$LD_LIBRARY_PATH

echo ">>> 开始编译..."
mkdir -p build
cd build
cmake ..
make -j4

echo ""
echo ">>> 开始运行纯 C++ ACLNN 推理..."
./aclnn_aes_test