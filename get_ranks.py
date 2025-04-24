import torch
from megatron.training.global_vars import get_args

rank = torch.distributed.get_rank()
local_rank = int(os.environ.get('LOCAL_RANK', -1))
device_index = torch.cuda.current_device()
device_name = torch.cuda.get_device_name(device_index)

print(f"[RANK {rank}] LOCAL_RANK={local_rank}, device_index={device_index}, device_name={device_name}")
