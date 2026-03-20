#!/bin/bash
# ============================================================================
# SAM3-I Unified Entry Script
# ============================================================================
# Usage:
#   bash run.sh install                    # Install dependencies
#   bash run.sh eval                       # Inference + Evaluation
#   bash run.sh train                      # Training
#   bash run.sh inference [args...]        # Inference only (pass inference.py args)
#   bash run.sh help                       # Show help
#
# All hyperparameters can be overridden via environment variables, e.g.:
#   CHECKPOINT=/path/to/ckpt GPUS=0,1 bash run.sh eval
#   CONFIG=configs/sam3i/sam3i_1-2 NPROC=4 bash run.sh train
# ============================================================================

set -e

MODE="${1:-help}"
PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"

case "${MODE}" in
    install)
        echo ">>> Installing SAM3-I dependencies..."
        cd "${PROJ_ROOT}/sam3"
        pip install -e ".[dev,train]"
        pip install pycocotools tqdm pillow einops
        echo ">>> Installation complete!"
        ;;

    train)
        echo ">>> Starting training..."
        shift
        bash "${PROJ_ROOT}/scripts/train.sh" "$@"
        ;;

    eval)
        echo ">>> Starting inference + evaluation..."
        shift
        bash "${PROJ_ROOT}/scripts/eval.sh" "$@"
        ;;

    inference)
        echo ">>> Starting inference..."
        shift
        python "${PROJ_ROOT}/scripts/inference.py" "$@"
        ;;

    help|*)
        echo "============================================"
        echo "  SAM3-I: Instruction-Guided Segmentation"
        echo "============================================"
        echo ""
        echo "Usage: bash run.sh <command> [options]"
        echo ""
        echo "Commands:"
        echo "  install       Install project dependencies"
        echo "  train         Start training"
        echo "  eval          One-click evaluation (reproduces paper results)"
        echo "  inference     Inference only (pass inference.py args)"
        echo "  help          Show help information"
        echo ""
        echo "Available training configs:"
        echo "  configs/sam3i/sam3i_1-1    Stage 1-1: Simple query training"
        echo "  configs/sam3i/sam3i_1-2    Stage 1-2: Complex query training"
        echo "  configs/sam3i/sam3i_3_all  Stage 3:   Joint training (all stages)"
        echo ""
        echo "Environment variables (eval command):"
        echo "  CHECKPOINT        Model checkpoint path (required)"
        echo "  DATASET_JSON_ROOT Root dir containing per-dataset GT JSON folders (required)"
        echo "  IMAGE_ROOT_BASE   Root dir containing per-dataset image folders (required)"
        echo "  OUTPUT_DIR        Output directory (default: ./outputs/eval_results)"
        echo "  GPUS              GPU list (default: 0,1,2,3,4,5,6,7)"
        echo "  BATCH_SIZE        Batch size (default: 32)"
        echo "  IMAGE_FOLDER_MAP  Custom dataset->folder mapping for shared image folders"
        echo "                    Format: \"DATASET:FOLDER,...\" (see eval.sh for details)"
        echo ""
        echo "Environment variables (train command):"
        echo "  CONFIG        Training config (default: configs/sam3i/sam3i_1-1)"
        echo "  NPROC         GPUs per node (default: 8)"
        echo "  NNODES        Number of nodes (default: 1)"
        echo "  NODE_RANK     Current node rank (default: 0)"
        echo "  MASTER_ADDR   Master node address (default: 127.0.0.1)"
        echo "  MASTER_PORT   Communication port (default: 29501)"
        echo ""
        echo "Config system:"
        echo "  All training configs inherit from the base.yaml base config file."
        echo "  Users only need to modify the 'paths' section in base.yaml to set:"
        echo "    - sam3i_datasets_root  (annotation JSON root)"
        echo "    - sam3i_image_root     (image folder root, can differ from JSON root)"
        echo "  Base config: sam3/sam3/train/configs/sam3i/base.yaml"
        echo ""
        echo "Examples:"
        echo "  # Install"
        echo "  bash run.sh install"
        echo ""
        echo "  # Train Stage 1-1 (Simple query)"
        echo "  CONFIG=configs/sam3i/sam3i_1-1 bash run.sh train"
        echo ""
        echo "  # Train Stage 1-2 (Complex query, 4 GPUs)"
        echo "  CONFIG=configs/sam3i/sam3i_1-2 NPROC=4 bash run.sh train"
        echo ""
        echo "  # Evaluation (one-click, reproduces paper results)"
        echo "  CHECKPOINT=/path/to/ckpt DATASET_JSON_ROOT=/path/to/jsons IMAGE_ROOT_BASE=/path/to/images bash run.sh eval"
        echo ""
        ;;
esac
