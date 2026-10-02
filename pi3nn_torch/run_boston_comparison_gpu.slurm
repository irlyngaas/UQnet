#!/bin/bash
#SBATCH -A stf006
#SBATCH -J pi3nn_boston_gpu
#SBATCH -o logs/pi3nn_boston_gpu-%j.o
#SBATCH -e logs/pi3nn_boston_gpu-%j.e
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1   # up to 8 for more GPUs on this node (Frontier has 8 GCDs/node); raise -N too for multiple nodes

# Same modules used to build the environment -- must be reloaded here since
# each SLURM job starts with a fresh environment, not whatever was loaded
# interactively when the conda env was created.
module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load miniforge3/23.11.0-0
module load rocm/7.1.1
module load craype-accel-amd-gfx90a

# Using a non-default CPE
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

# Same conda env used for the emulator training work (hydrogen-emulator-
# configurable) -- it already has torch, just needs scikit-learn added
# once: conda activate <this env> && pip install scikit-learn
conda activate /lustre/orion/stf006/proj-shared/irl1/MINI_CLEAN/envs/groundwater-emulator

# /tmp is node-local and can carry permission-mismatched leftovers from
# other jobs that previously ran on the same compute node ("Permission
# denied [/tmp/.cache/miopen/...]") -- Lustre project space avoids that
# class of problem entirely. Shared with the emulator training scripts
# since MIOpen's cache isn't per-model.
export MIOPEN_USER_DB_PATH="/lustre/orion/stf006/proj-shared/irl1/miopen_cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

mkdir -p logs

# Needed for torch.distributed's rendezvous (env:// init method, the
# default when init_method isn't passed to init_process_group), even when
# running a single task/GPU -- pi3nn_torch/distributed.py's setup_distributed
# falls back to ordinary non-distributed training automatically whenever
# SLURM_NTASKS is 1, no script changes needed to go from 1 GPU to many,
# just the #SBATCH lines above.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=3442

# Run from the repo root: pi3nn_torch is a proper package (relative imports
# in run_boston_comparison.py), so it must be invoked as
# `python -m pi3nn_torch.run_boston_comparison`, not by file path, and the
# repo root must be on the module search path.
#
# NOTE: this path assumes UQnet is checked out at the same Lustre project
# location convention as hydrogen-emulator-configurable. Update it if your
# actual checkout lives elsewhere.
cd /lustre/orion/stf006/proj-shared/irl1/UQnet

# --gpus-per-task=1 --gpu-bind=closest (not --gpus-per-node): this matches
# emulator-1ts's hand-rolled DDP convention, not emulator_configurable's
# Lightning-managed one -- pi3nn_torch/distributed.py does its own
# local_rank % device_count() mapping and expects each task to see exactly
# one GPU, the opposite of what Lightning's SLURM integration wants. Mixing
# the two conventions is a known failure mode (see emulator_configurable's
# run_conus1_forcedstrnn_training.slurm for the actual errors this produces
# the other way around) -- don't swap this for --gpus-per-node here.
srun -N$SLURM_NNODES -n$SLURM_NTASKS -c7 --gpus-per-task=1 --gpu-bind=closest \
    python3 -u -m pi3nn_torch.run_boston_comparison --quantile 0.95 --device cuda
