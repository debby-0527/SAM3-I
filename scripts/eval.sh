#!/bin/bash
# ============================================================================
# SAM3-I One-Click Evaluation Script
# ============================================================================
# Reproduces all paper results in a single run. Each dataset uses its
# appropriate evaluation mode automatically -- no manual configuration needed.
#
# Usage:
#   CHECKPOINT=/path/to/ckpt DATASET_JSON_ROOT=/path/to/jsons IMAGE_ROOT_BASE=/path/to/images bash scripts/eval.sh
#
# Environment variables:
#   CHECKPOINT        (required) Path to model checkpoint (.pt)
#   DATASET_JSON_ROOT (required) Root dir containing per-dataset GT JSON folders
#   IMAGE_ROOT_BASE   (required) Root dir containing per-dataset image folders
#   OUTPUT_DIR        (optional) Where to save predictions  (default: ./outputs/eval_results)
#   GPUS              (optional) GPU list                   (default: 0,1,2,3,4,5,6,7)
#   BATCH_SIZE        (optional) Inference batch size       (default: 32)
#   DET_THRESHOLD     (optional) Detection threshold        (default: 0.5)
#   IMAGE_FOLDER_MAP  (optional) Custom dataset-to-image-folder mapping.
#                     Format: "DATASET_NAME:FOLDER_NAME,..."
#                     Use this when multiple datasets share the same image folder.
#                     Many datasets share COCO images -- only 3 image sources are needed:
#                       COCO 2014: RefCOCO, RefCOCOplus, RefCOCOg, gRefCOCO, Ref-ZOM
#                       COCO 2017: HMPL-Instruct_1to1, HMPL-Instruct_1toN, HMPL-Instruct_1toAll, MMR
#                       ReasonSeg: standalone
#                     Example:
#                       "RefCOCO:coco2014,RefCOCOplus:coco2014,RefCOCOg:coco2014,gRefCOCO:coco2014,Ref-ZOM:coco2014,HMPL-Instruct_1to1:coco2017,HMPL-Instruct_1toN:coco2017,HMPL-Instruct_1toAll:coco2017,MMR:coco2017"
#
# Directory layout expected:
#   DATASET_JSON_ROOT/
#     ├── HMPL-Instruct_1to1/sam3i_val.json
#     ├── HMPL-Instruct_1toN/sam3i_val.json
#     ├── HMPL-Instruct_1toAll/sam3i_val.json
#     ├── RefCOCO/sam3i_val.json
#     └── Ref-ZOM/sam3i_val.json
#
#   IMAGE_ROOT_BASE/
#     ├── coco2014/   (shared by RefCOCO, RefCOCOplus, RefCOCOg, gRefCOCO, Ref-ZOM)
#     ├── coco2017/   (shared by HMPL-Instruct_1to1/1toN/1toAll, MMR)
#     └── ReasonSeg/  (standalone)
#
#   Use IMAGE_FOLDER_MAP to point datasets to their shared image folder:
#     IMAGE_FOLDER_MAP="RefCOCO:coco2014,RefCOCOplus:coco2014,...,HMPL-Instruct_1to1:coco2017,..."
# ============================================================================

set -e

# ============================================================================
# Required parameters
# ============================================================================
CHECKPOINT="${CHECKPOINT:-}"
DATASET_JSON_ROOT="${DATASET_JSON_ROOT:-}"
IMAGE_ROOT_BASE="${IMAGE_ROOT_BASE:-}"

if [ -z "${CHECKPOINT}" ]; then
    echo "Error: CHECKPOINT is required. Set it via: CHECKPOINT=/path/to/checkpoint.pt"
    exit 1
fi
if [ -z "${DATASET_JSON_ROOT}" ]; then
    echo "Error: DATASET_JSON_ROOT is required. Set it via: DATASET_JSON_ROOT=/path/to/json/root"
    exit 1
fi
if [ -z "${IMAGE_ROOT_BASE}" ]; then
    echo "Error: IMAGE_ROOT_BASE is required. Set it via: IMAGE_ROOT_BASE=/path/to/image/root"
    exit 1
fi

if [ ! -f "${CHECKPOINT}" ]; then
    echo "Error: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

# ============================================================================
# Optional parameters
# ============================================================================
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/eval_results}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DET_THRESHOLD="${DET_THRESHOLD:-0.5}"

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INFERENCE_SCRIPT="${PROJ_ROOT}/scripts/inference.py"
EVALUATE_SCRIPT="${PROJ_ROOT}/scripts/evaluate.py"

# ============================================================================
# Image folder mapping (for shared image folders across datasets)
# ============================================================================
# By default, each dataset maps to IMAGE_ROOT_BASE/<DATASET_NAME>/.
# Override via IMAGE_FOLDER_MAP env var to share image folders across datasets.
# Format: "DATASET:FOLDER,DATASET:FOLDER,..."
#
# Example — group by COCO version:
#   IMAGE_FOLDER_MAP="RefCOCO:coco2014,RefCOCOplus:coco2014,RefCOCOg:coco2014,gRefCOCO:coco2014,Ref-ZOM:coco2014,HMPL-Instruct_1to1:coco2017,HMPL-Instruct_1toN:coco2017,HMPL-Instruct_1toAll:coco2017,MMR:coco2017"
# ============================================================================

# Parse IMAGE_FOLDER_MAP into an associative array
declare -A IMG_DIR_MAP
if [ -n "${IMAGE_FOLDER_MAP:-}" ]; then
    IFS=',' read -ra MAP_ENTRIES <<< "${IMAGE_FOLDER_MAP}"
    for entry in "${MAP_ENTRIES[@]}"; do
        IFS=':' read -r key val <<< "${entry}"
        key="$(echo "${key}" | xargs)"  # trim whitespace
        val="$(echo "${val}" | xargs)"
        IMG_DIR_MAP["${key}"]="${val}"
    done
fi

# Resolve image directory for a given dataset name.
# Priority: IMAGE_FOLDER_MAP entry > dataset name itself.
resolve_img_dir() {
    local dataset="$1"
    if [ -n "${IMG_DIR_MAP[${dataset}]+_}" ]; then
        echo "${IMAGE_ROOT_BASE}/${IMG_DIR_MAP[${dataset}]}/"
    else
        echo "${IMAGE_ROOT_BASE}/${dataset}/"
    fi
}

mkdir -p "${OUTPUT_DIR}"

echo "========================================================"
echo "  SAM3-I Evaluation Pipeline"
echo "========================================================"
echo "  Checkpoint:      ${CHECKPOINT}"
echo "  Dataset JSON:    ${DATASET_JSON_ROOT}"
echo "  Image Root Base: ${IMAGE_ROOT_BASE}"
echo "  Output Dir:      ${OUTPUT_DIR}"
echo "  GPUs:            ${GPUS}"
echo "  Batch Size:      ${BATCH_SIZE}"
echo "  Det Threshold:   ${DET_THRESHOLD}"
if [ -n "${IMAGE_FOLDER_MAP:-}" ]; then
echo "  Folder Map:      ${IMAGE_FOLDER_MAP}"
fi
echo "========================================================"

# ============================================================================
# Task definitions:  "DATASET|CATEGORY|EVAL_MODE"
#   - EVAL_MODE is chosen per dataset to match the paper protocol.
#   - KEEP_TOP1 is applied where appropriate.
# ============================================================================
EVAL_TASKS=(
    "HMPL-Instruct_1to1|simple|1to1"
    "HMPL-Instruct_1to1|complex|1to1"
    "HMPL-Instruct_1toN|simple|multi_inst"
    "HMPL-Instruct_1toN|complex|multi_inst"
    "HMPL-Instruct_1toAll|simple|multi_inst"
    "HMPL-Instruct_1toAll|complex|multi_inst"
    "RefCOCO|simple|1to1"
    "Ref-ZOM|simple|multi_inst"
)

TOTAL_TASKS=${#EVAL_TASKS[@]}
CURRENT=0

for TASK in "${EVAL_TASKS[@]}"; do
    CURRENT=$((CURRENT + 1))

    IFS='|' read -r DATA CATEGORY EVAL_MODE <<< "${TASK}"

    echo ""
    echo "========================================================"
    echo "  [${CURRENT}/${TOTAL_TASKS}] ${DATA} | ${CATEGORY} | mode=${EVAL_MODE}"
    echo "========================================================"

    GT_PATH="${DATASET_JSON_ROOT}/${DATA}/sam3i_val.json"
    PRED_PATH="${OUTPUT_DIR}/${DATA}_${CATEGORY}.json"

    # Resolve image folder via mapping (supports shared folders)
    IMG_ROOT="$(resolve_img_dir "${DATA}")"

    if [ ! -f "${GT_PATH}" ]; then
        echo "[SKIP] GT not found: ${GT_PATH}"
        continue
    fi

    if [ ! -d "${IMG_ROOT}" ]; then
        echo "[WARN] Image dir not found: ${IMG_ROOT}, trying anyway..."
    fi

    # Determine whether to keep only top-1 prediction
    KEEP_TOP1_FLAG=""
    if [ "${EVAL_MODE}" = "1to1" ]; then
        case "${DATA}" in
            HMPL-Instruct_*) ;;
            *) KEEP_TOP1_FLAG="--keep_top1" ;;
        esac
    fi

    echo "  Step 1: Inference..."
    echo "    GT JSON:    ${GT_PATH}"
    echo "    Image Root: ${IMG_ROOT}"
    echo "    Output:     ${PRED_PATH}"

    python "${INFERENCE_SCRIPT}" \
        --json_path "${GT_PATH}" \
        --image_root "${IMG_ROOT}" \
        --output_path "${PRED_PATH}" \
        --checkpoint_path "${CHECKPOINT}" \
        --categories "${CATEGORY}" \
        --batch_size "${BATCH_SIZE}" \
        --gpus "${GPUS}" \
        --detection_threshold "${DET_THRESHOLD}" \
        ${KEEP_TOP1_FLAG}

    echo "  Step 2: Evaluate [mode=${EVAL_MODE}]..."
    python "${EVALUATE_SCRIPT}" \
        --gt_path "${GT_PATH}" \
        --pred_path "${PRED_PATH}" \
        --mode "${EVAL_MODE}"

    echo "  [${CURRENT}/${TOTAL_TASKS}] ${DATA} | ${CATEGORY} done!"
done

echo ""
echo "========================================================"
echo "  All evaluation tasks completed! Results: ${OUTPUT_DIR}/"
echo "========================================================"
echo ""
echo "Prediction files:"
ls -lh "${OUTPUT_DIR}"/*.json 2>/dev/null || echo "  (none)"
