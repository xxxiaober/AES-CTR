import os
import time
import torch
import torch_npu
import npu_custom_ops  

print("=====================================================")
print("⏱️  NPU Profiler: AesCtrEncrypt Operator (Safe Mode) ⏱️")
print("=====================================================")

# ==========================================
# [0] 准备测试数据
# ==========================================
DATA_SIZE = 224 * 224 * 3 * 4 
print(f"[*] 正在 NPU 显存中生成 {DATA_SIZE} 字节的测试数据...")

input_tensor_npu = torch.randint(0, 256, (DATA_SIZE,), dtype=torch.uint8).npu()
key_npu = torch.randint(0, 256, (16,), dtype=torch.uint8).npu()
iv_npu = torch.randint(0, 256, (16,), dtype=torch.uint8).npu()

def run_my_operator():
    _ = npu_custom_ops.aes_ctr_encrypt(input_tensor_npu, key_npu, iv_npu)

# ==========================================
# [1] Profiler 核心逻辑
# ==========================================
def run_profiling():
    print("\n[1] 开始预热 (Warm-up)...")
    for _ in range(5):
        run_my_operator()
    torch.npu.synchronize() 
    print("    -> 预热完成！(算子本身执行极其完美！)")

    print("\n[2] 配置 Profiler (Safe Mode)...")
    # 🚀 修复点：放弃激进的 ExperimentalConfig，使用最基础稳健的 Activity 抓取
    activities = [
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU
    ]
    
    # 🚀 修复点：使用 with 上下文管理器，并关闭 shape/memory 深度检测防崩溃
    with torch_npu.profiler.profile(
        activities=activities,
        record_shapes=False,  
        profile_memory=False, 
        with_stack=False      
    ) as prof:
        print("[3] 启动性能采集并执行算子...")
        run_my_operator()
        torch.npu.synchronize() 
    
    print("\n[4] 导出 Chrome Trace 文件...")
    os.makedirs('./trace', exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    filepath = f'./trace/chrome_trace_{timestamp}.json'
    
    prof.export_chrome_trace(filepath)
    print(f"🎉 性能分析文件已导出至: {filepath}")
    print("👉 请打开 Chrome 浏览器，访问 chrome://tracing 并拖入此 json 文件查看结果！")

if __name__ == "__main__":
    run_profiling()