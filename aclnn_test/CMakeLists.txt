cmake_minimum_required(VERSION 3.14)
project(AclnnAesTest)

set(CMAKE_CXX_STANDARD 17)

# 从环境变量获取 CANN 路径，默认使用 /usr/local/Ascend/ascend-toolkit/latest
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH "/usr/local/Ascend/ascend-toolkit/latest")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

# 设置头文件搜索路径 (CANN 基础头文件 + 自定义算子头文件)
include_directories(
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/include
)

# 设置动态库搜索路径 (CANN 基础库 + 自定义算子库)
link_directories(
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib
)

add_executable(aclnn_aes_test main.cpp)

# 链接核心库：ascendcl (设备管理), nnopbase (算子基础库), cust_opapi (你的自定义算子)
target_link_libraries(aclnn_aes_test ascendcl nnopbase cust_opapi)