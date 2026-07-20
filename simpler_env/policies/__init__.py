import numpy as np
import os
import random
import torch

# wx:Test-time learnable feature mask v1.0
# def setup_torch_seed(seed):
#     torch.manual_seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
# wx:Test-time learnable feature mask v1.0
_CUDNN_FLAGS_INITIALIZED = False

def setup_torch_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cudnn flags 是全局设置，不需要每个 episode reset 都重复设置。
    # 某些 PyTorch 状态下，第一次之后再直接修改会触发：
    # RuntimeError: not allowed to set torch.backends.cudnn flags after disable_global_flags
    global _CUDNN_FLAGS_INITIALIZED
    if not _CUDNN_FLAGS_INITIALIZED:
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        except RuntimeError as e:
            if "disable_global_flags" in str(e):
                print(
                    "[setup_torch_seed] Skip setting cudnn global flags "
                    "because PyTorch has disabled direct global flag mutation."
                )
            else:
                raise
        _CUDNN_FLAGS_INITIALIZED = True
# wx:Test-time learnable feature mask v1.0
