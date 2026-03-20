#!/bin/bash
# ============================================================================
# SAM3-I Training Script
# ============================================================================
# Usage:
#   Single node 8 GPUs: bash scripts/train.sh
#   Custom config:      CONFIG=configs/sam3i/sam3i_1-2 NPROC=4 bash scripts/train.sh
#   Multi-node:         NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash scripts/train.sh
#
# Available configs:
#   configs/sam3i/sam3i_1-1    Stage 1-1: Simple query training
#   configs/sam3i/sam3i_1-2    Stage 1-2: Complex query training
#   configs/sam3i/sam3i_3_all  Stage 3:   Joint training (all stages)
#
# All hyperparameters can be overridden via environment variables.
# ============================================================================

set -e

# ============================================================================
# Hyperparameters (overridable via environment variables)
# ============================================================================
# Training config (path relative to sam3/sam3/train/)
CONFIG="${CONFIG:-configs/sam3i/sam3i_1-1}"

# Distributed training config
NNODES="${NNODES:-1}"                        # Number of nodes
NPROC="${NPROC:-8}"                          # GPUs per node
NODE_RANK="${NODE_RANK:-0}"                  # Current node rank (0-based)
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"      # Master node address
MASTER_PORT="${MASTER_PORT:-29501}"          # Communication port
# ============================================================================
# Paths
# ============================================================================
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SAM3_ROOT="${PROJ_ROOT}/sam3"

# ============================================================================
# Launch Info
# ============================================================================
echo "========================================"
echo "SAM3-I Training"
echo "  Config:     ${CONFIG}"
echo "  Nodes:      ${NNODES}, GPUs/node: ${NPROC}"
echo "  Node rank:  ${NODE_RANK}"
echo "  Master:     ${MASTER_ADDR}:${MASTER_PORT}"
echo "  SAM3 root:  ${SAM3_ROOT}"
echo "========================================"

cd "${SAM3_ROOT}"

if [ "${NNODES}" -eq 1 ]; then
    # Single-node mode (using SAM3 built-in submitit launcher)
    python sam3/train/train.py \
        -c "${CONFIG}" \
        --use-cluster 0 \
        --num-nodes 1 \
        --num-gpus "${NPROC}"
else
    # Multi-node mode (torchrun)
    torchrun \
        --nnodes="${NNODES}" \
        --nproc_per_node="${NPROC}" \
        --master_port="${MASTER_PORT}" \
        --node_rank="${NODE_RANK}" \
        --master_addr="${MASTER_ADDR}" \
        sam3/train/train.py \
        -c "${CONFIG}" \
        --use-cluster 0 \
        --num-nodes "${NNODES}" \
        --num-gpus "${NPROC}"
fi

echo "Training completed!"
