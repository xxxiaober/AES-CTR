#!/usr/bin/env bash
set -euo pipefail

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TORCH_NPU_DISABLE_DISTRIBUTED=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib:$LD_LIBRARY_PATH

echo "TORCH_DEVICE_BACKEND_AUTOLOAD=${TORCH_DEVICE_BACKEND_AUTOLOAD}"
echo "TORCH_NPU_DISABLE_DISTRIBUTED=${TORCH_NPU_DISABLE_DISTRIBUTED}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

if [[ "${1:-}" == "--check" ]]; then
  /usr/bin/python3 - <<'PY'
import torch
import torch_npu
print('torch', torch.__version__)
print('torch_npu', torch_npu.__version__)
print('npu available', torch.npu.is_available())
print('npu count', torch.npu.device_count())
try:
    x = torch.ones((4,), device='npu')
    y = x + 1
    print('npu smoke test ok', y.cpu().tolist())
except Exception as e:
    print('npu smoke test failed:', repr(e))
    raise
PY
fi
