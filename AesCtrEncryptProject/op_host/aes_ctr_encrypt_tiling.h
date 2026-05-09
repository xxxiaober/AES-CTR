#ifndef AES_CTR_ENCRYPT_TILING_H
#define AES_CTR_ENCRYPT_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AesCtrEncryptTilingData)
  // 对应我们 Device 侧 Init 函数接收的参数
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AesCtrEncrypt, AesCtrEncryptTilingData)
}
#endif // AES_CTR_ENCRYPT_TILING_H