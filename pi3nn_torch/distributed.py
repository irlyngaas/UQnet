"""
Distributed-training helpers for pi3nn_torch, mirroring the hand-rolled
DDP pattern already validated on Frontier in hydrogen-emulator-
configurable's emulator-1ts/main.py: reads rank/world_size/local_rank
from SLURM (or torchrun) env vars, maps local_rank to a visible GPU index
via modulo (works whether SLURM restricts each task to 1 GPU via
--gpus-per-task=1 --gpu-bind=closest, or exposes all GPUs per node to
every task), and uses the 'nccl' backend for GPU runs -- PyTorch aliases
this to RCCL automatically on ROCm builds, so it's correct on Frontier's
AMD GPUs too, not just NVIDIA.
"""
import os
import torch
import torch.distributed as dist


def get_distributed_info():
    if 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_NTASKS' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank


def setup_distributed(use_gpu):
    """
    Returns (rank, world_size, device). When world_size==1 this never
    touches torch.distributed at all, so a plain `python3 -m
    pi3nn_torch.run_boston_comparison` behaves exactly as before --
    nothing below changes unless actually launched with more than one
    SLURM/torchrun task.
    """
    rank, world_size, local_rank = get_distributed_info()
    distributed = world_size > 1

    if not use_gpu:
        if distributed:
            dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
        return rank, world_size, 'cpu'

    # local_gpu_id, not local_rank, for the same reason as emulator-1ts's
    # main.py: with --gpus-per-task=1 --gpu-bind=closest, each process
    # only ever sees ONE GPU, always at index 0 from its own vantage
    # point; without per-task binding, every process sees all GPUs on the
    # node and needs local_rank to pick its own. The modulo handles both.
    local_gpu_id = local_rank % torch.cuda.device_count()
    device = f'cuda:{local_gpu_id}'
    torch.cuda.set_device(local_gpu_id)
    if distributed:
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    return rank, world_size, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0
