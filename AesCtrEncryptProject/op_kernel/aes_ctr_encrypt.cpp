#include "kernel_operator.h"

using namespace AscendC;

#define AES_BLOCK_BYTES 16
#define AES_KEY_BYTES 16
#define AES_ROUND_NUM 10
#define AES_EXPANDED_KEY_BYTES 176
constexpr int32_t BUFFER_NUM = 2;

class KernelAesCtr {
public:
    __aicore__ inline KernelAesCtr() {
        // 1. 初始化 S-Box
        uint8_t sbox_init[256] = {
            0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
            0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
            0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
            0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
            0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
            0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
            0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
            0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
            0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
            0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
            0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
            0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
            0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
            0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
            0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
            0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
        };
        for (int i = 0; i < 256; i++) sbox[i] = sbox_init[i];
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR key, GM_ADDR iv, GM_ADDR y, uint32_t totalLen, uint32_t tileLen) {
        this->totalLength = totalLen;
        this->tileLength = tileLen; 
    
        uint32_t blockIdx = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();

        uint32_t dataPerCore = (totalLength / blockNum / 32) * 32;  // 确保每个核心处理的数据量是 32 字节的整数倍，以适配 AES 块大小和向量化处理
        uint32_t offset = blockIdx * dataPerCore;  // 每个核心的起始偏移

        this->currentCoreDataLength = dataPerCore;
        // 最后一个核心可能需要处理剩余的数据
        if (blockIdx == blockNum - 1) {
            this->currentCoreDataLength = totalLength - offset; 
        }
        
        // 计算循环次数，确保覆盖所有数据
        this->loopCount = this->currentCoreDataLength / tileLength;
        if (this->currentCoreDataLength % tileLength != 0) this->loopCount++;

        xGm.SetGlobalBuffer((__gm__ uint8_t*)x + offset, this->currentCoreDataLength);
        yGm.SetGlobalBuffer((__gm__ uint8_t*)y + offset, this->currentCoreDataLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength);
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength);
        pipe.InitBuffer(keyStreamBuf, tileLength); 
        pipe.InitBuffer(tempBuf, tileLength); // 用于魔法异或的临时缓存

        uint8_t localKey[AES_KEY_BYTES];
        for (int i = 0; i < AES_KEY_BYTES; i++) localKey[i] = ((__gm__ uint8_t*)key)[i];
        KeyExpansion(localKey, expandedKey, sbox);

        for (int i = 0; i < 16; i++) baseIv[i] = ((__gm__ uint8_t*)iv)[i];
        BigEndAdd(baseIv, offset / AES_BLOCK_BYTES);
    }

    __aicore__ inline void Process() {
        for (int32_t i = 0; i < this->loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    static __aicore__ inline uint32_t AlignUp32(uint32_t n) {
        return (n + 31) & ~31u;
    }

    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<uint8_t> xLocal = inQueueX.AllocTensor<uint8_t>();

        uint32_t currentTileLen = this->tileLength;
        if (progress == this->loopCount - 1 && this->currentCoreDataLength % this->tileLength != 0) {
            currentTileLen = this->currentCoreDataLength % this->tileLength;
        }

        DataCopy(xLocal, xGm[progress * tileLength], AlignUp32(currentTileLen));
        inQueueX.EnQue(xLocal);
    }

__aicore__ inline void Compute(int32_t progress) {
        LocalTensor<uint8_t> xLocal = inQueueX.DeQue<uint8_t>();
        LocalTensor<uint8_t> yLocal = outQueueY.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> ksLocal = keyStreamBuf.Get<uint8_t>(); 
        LocalTensor<uint8_t> tempLocal = tempBuf.Get<uint8_t>(); 

        uint32_t currentTileLen = this->tileLength;
        // 最后一个循环可能处理的数据长度不足一个完整的 tile
        if (progress == this->loopCount - 1 && this->currentCoreDataLength % this->tileLength != 0) {
            currentTileLen = this->currentCoreDataLength % this->tileLength;
        }

        // 1. 生成密钥流
        GenerateKeyStream(ksLocal, progress, currentTileLen);
        
        // 步骤 A: 迎合硬件逻辑单元，将内存解释为 16 位无符号整数进行 Or / And
        uint32_t elements16 = (currentTileLen + 1) / 2; 
        LocalTensor<uint16_t> x_16 = xLocal.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> ks_16 = ksLocal.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> y_16 = yLocal.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> temp_16 = tempLocal.ReinterpretCast<uint16_t>();

        AscendC::Or(y_16, x_16, ks_16, elements16);
        AscendC::And(temp_16, x_16, ks_16, elements16);

        // 步骤 B: 迎合硬件算术单元，原地将内存重新解释为 32 位有符号整数进行 Sub
        uint32_t elements32 = (currentTileLen + 3) / 4; 
        LocalTensor<int32_t> y_32 = yLocal.ReinterpretCast<int32_t>();
        LocalTensor<int32_t> temp_32 = tempLocal.ReinterpretCast<int32_t>();

        AscendC::Sub(y_32, y_32, temp_32, elements32);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<uint8_t> yLocal = outQueueY.DeQue<uint8_t>();
        
        uint32_t currentTileLen = this->tileLength;
        // 最后一个循环可能处理的数据长度不足一个完整的 tile
        if (progress == this->loopCount - 1 && this->currentCoreDataLength % this->tileLength != 0) {
            currentTileLen = this->currentCoreDataLength % this->tileLength;
        }

        DataCopy(yGm[progress * tileLength], yLocal, AlignUp32(currentTileLen));
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void GenerateKeyStream(LocalTensor<uint8_t>& ksLocal, int32_t progress, uint32_t currentTileLen) {
        uint8_t currentIv[AES_BLOCK_BYTES];
        for (int i = 0; i < 16; i++) currentIv[i] = baseIv[i];

        uint32_t blockOffset = (progress * tileLength) / AES_BLOCK_BYTES;
        BigEndAdd(currentIv, blockOffset);

        uint32_t blocksInTile = (currentTileLen + AES_BLOCK_BYTES - 1) / AES_BLOCK_BYTES;

        for (uint32_t b = 0; b < blocksInTile; b++) {
            uint8_t keyStream[16];
            AesEncryptBlock(currentIv, expandedKey, keyStream, b, sbox);

            for (int j = 0; j < 16; j++) {
                if (b * 16 + j < currentTileLen) {
                    ksLocal.SetValue(b * 16 + j, keyStream[j]);
                }
            }
            BigEndAdd(currentIv, 1);
        }
    }


    static __aicore__ inline uint8_t xtime(uint8_t x) {
        return ((x << 1) ^ (((x >> 7) & 1) * 0x1b));
    }

    static __aicore__ inline void AddRoundKey(uint8_t* state, const uint8_t* roundKey) {
        for (int i = 0; i < 16; i++) state[i] ^= roundKey[i];
    }

    static __aicore__ void KeyExpansion(const uint8_t* key, uint8_t* roundKeys, const uint8_t* sbox) {
        uint8_t Rcon[11] = {0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36};
        for (int i = 0; i < 16; i++) roundKeys[i] = key[i];
        int bytesUsed = 16, rconIdx = 1;
        uint8_t temp[4];
        while (bytesUsed < 176) {
            for (int i = 0; i < 4; i++) temp[i] = roundKeys[bytesUsed - 4 + i];
            if ((bytesUsed % 16) == 0) {
                uint8_t t = temp[0];
                temp[0] = sbox[temp[1]] ^ Rcon[rconIdx++];
                temp[1] = sbox[temp[2]];
                temp[2] = sbox[temp[3]];
                temp[3] = sbox[t];
            }
            for (int i = 0; i < 4; i++) {
                roundKeys[bytesUsed] = roundKeys[bytesUsed - 16] ^ temp[i];
                bytesUsed++;
            }
        }
    }

    static __aicore__ void SubShiftMix(uint8_t* state, const uint8_t* sbox) {
        for (int i = 0; i < 16; i++) state[i] = sbox[state[i]];
        uint8_t t;
        t = state[1]; state[1] = state[5]; state[5] = state[9]; state[9] = state[13]; state[13] = t;
        t = state[2]; state[2] = state[10]; state[10] = t; t = state[6]; state[6] = state[14]; state[14] = t;
        t = state[3]; state[3] = state[15]; state[15] = state[11]; state[11] = state[7]; state[7] = t;
        for (int i = 0; i < 4; i++) {
            uint8_t *s = state + i * 4;
            uint8_t a0 = s[0], a1 = s[1], a2 = s[2], a3 = s[3];
            uint8_t h = a0 ^ a1 ^ a2 ^ a3;
            s[0] ^= h ^ xtime(a0 ^ a1); s[1] ^= h ^ xtime(a1 ^ a2);
            s[2] ^= h ^ xtime(a2 ^ a3); s[3] ^= h ^ xtime(a3 ^ a0);
        }
    }

    static __aicore__ void SubShift(uint8_t* state, const uint8_t* sbox) {
        for (int i = 0; i < 16; i++) state[i] = sbox[state[i]];
        uint8_t t;
        t = state[1]; state[1] = state[5]; state[5] = state[9]; state[9] = state[13]; state[13] = t;
        t = state[2]; state[2] = state[10]; state[10] = t; t = state[6]; state[6] = state[14]; state[14] = t;
        t = state[3]; state[3] = state[15]; state[15] = state[11]; state[11] = state[7]; state[7] = t;
    }

    static __aicore__ void AesEncryptBlock(const uint8_t in[16], const uint8_t* rKeys, uint8_t out[16], uint32_t bId, const uint8_t* sbox) {
        uint8_t state[16];
        for (int i = 0; i < 16; i++) state[i] = in[i];
        AddRoundKey(state, rKeys);
        for (int r = 1; r < 10; r++) {
            SubShiftMix(state, sbox);
            AddRoundKey(state, rKeys + 16 * r);
        }
        SubShift(state, sbox);
        AddRoundKey(state, rKeys + 160);
        for (int i = 0; i < 16; i++) out[i] = state[i];
    }

    static __aicore__ void BigEndAdd(uint8_t* ctr, uint32_t val) {
        uint32_t carry = val;
        for (int i = 15; i >= 0 && carry > 0; i--) {
            uint32_t sum = (uint32_t)ctr[i] + (carry & 0xFF);
            ctr[i] = (uint8_t)(sum & 0xFF);
            carry = (carry >> 8) + (sum >> 8);
        }
    }

    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<TPosition::VECCALC> keyStreamBuf; 
    TBuf<TPosition::VECCALC> tempBuf; 
    
    GlobalTensor<uint8_t> xGm, yGm;
    uint8_t sbox[256], expandedKey[176], baseIv[16];
    uint32_t totalLength, tileLength, currentCoreDataLength, loopCount;
};

extern "C" __global__ __aicore__ void aes_ctr_encrypt(GM_ADDR x, GM_ADDR key, GM_ADDR iv, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling); 
    KernelAesCtr op;
    op.Init(x, key, iv, y, tilingData.totalLength, tilingData.tileLength);
    op.Process();
}