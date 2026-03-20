#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""
SAM3-I Evaluation Script
========================
Inference output is unified as Format B (predictions list). Two evaluation
modes are available, selected explicitly via --mode:

[1to1 mode] (HMPL-Instruct_1to1 / ReasonSeg, etc.)
  - Merge (OR union) all predictions per record into a single mask
  - Align with GT by annotation_id
  - Reference: internal script metrics_005.py
  - Metrics: gIoU, mIoU, Precision@0.50, COCO AP

[multi_inst mode] (HMPL-Instruct_1toAll / HMPL-Instruct_1toN, etc.)
  - Align by image_id: GT is the union of all annotations for the image
  - Pred is the union of all predictions[*].rle across pred items per image
  - Reference: internal script metrics_multi_inst.py
  - Metrics: gIoU, Precision@0.50 (per-image granularity)

Usage:
    # 1to1 datasets
    python scripts/evaluate.py --mode 1to1 \\
        --gt_path /path/to/gt.json --pred_path ./predictions.json

    # Multi-instance datasets
    python scripts/evaluate.py --mode multi_inst \\
        --gt_path /path/to/gt.json --pred_path ./predictions.json
"""

import argparse
import json
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


# ============================================================================
# Utility Functions
# ============================================================================

def safe_decode_rle(rle_dict):
    """Decode RLE to binary mask, handling empty masks properly."""
    if not rle_dict:
        return None

    if not rle_dict.get('counts'):
        size = rle_dict.get('size', [0, 0])
        return np.zeros(size, dtype=bool)

    rle = rle_dict.copy()
    if isinstance(rle['counts'], str):
        rle['counts'] = rle['counts'].encode('utf-8')
    return mask_utils.decode(rle).astype(bool)


def merge_predictions_to_mask(predictions, fallback_shape=None):
    """
    Merge all RLE masks in the predictions list into a single binary mask
    via OR union. Returns a zero mask of fallback_shape (or None) when
    predictions is empty. Also returns the maximum score.
    """
    combined = None
    max_score = 1.0

    if predictions:
        scores = [p.get('score', 1.0) for p in predictions]
        max_score = max(scores)
        for p in predictions:
            mask = safe_decode_rle(p['rle'])
            if mask is None:
                continue
            combined = mask if combined is None else (combined | mask)

    if combined is None and fallback_shape is not None:
        combined = np.zeros(fallback_shape, dtype=bool)

    return combined, max_score


# ============================================================================
# 1to1 Evaluation: align by annotation_id
# Strictly follows internal script metrics_005.py
# ============================================================================

def evaluate_1to1(gt_data, pred_data):
    """
    1to1 evaluation logic.
    Strictly follows metrics_005.py:
      - Merge (OR union) predictions list into a single mask
      - Align by annotation_id
      - Compute gIoU, mIoU, Precision@0.50, COCO AP
    """
    print("\n[1to1 mode] Align by annotation_id")
    print("Reference: metrics_005.py")

    # ---------------------------------------------------------
    # Part 1: Compute gIoU and mIoU
    # ---------------------------------------------------------
    print("Calculating gIoU & mIoU...")
    gt_map = {ann['id']: ann for ann in gt_data['annotations']}

    ious = []
    intersection_sum = 0.0
    union_sum = 0.0
    coco_results = []

    for item in tqdm(pred_data):
        ann_id = item['annotation_id']
        if ann_id not in gt_map:
            continue
        gt_ann = gt_map[ann_id]

        mask_gt = safe_decode_rle(gt_ann['segmentation'])

        # Merge predictions list into a single mask (equivalent to prediction_rle in metrics_005.py)
        predictions = item.get('predictions', [])
        mask_pred, score = merge_predictions_to_mask(
            predictions,
            fallback_shape=mask_gt.shape if mask_gt is not None else None
        )

        if mask_pred is None:
            mask_pred = np.zeros_like(mask_gt, dtype=bool)

        intersection = (mask_pred & mask_gt).sum()
        union = (mask_pred | mask_gt).sum()
        iou = intersection / union if union > 0 else 0.0

        intersection_sum += intersection
        union_sum += union
        ious.append(iou)

        # Prepare for COCO AP: re-encode the merged mask as RLE
        if predictions:
            combined_rle = mask_utils.encode(np.asfortranarray(mask_pred.astype(np.uint8)))
            if isinstance(combined_rle['counts'], bytes):
                combined_rle['counts'] = combined_rle['counts'].decode('utf-8')
            combined_rle['size'] = list(combined_rle['size'])
            if combined_rle['counts'] != '':
                coco_results.append({
                    "image_id": item['image_id'],
                    "category_id": 1,
                    "segmentation": combined_rle,
                    "score": score,
                })

    giou = np.mean(ious) * 100
    miou = (intersection_sum / union_sum * 100) if union_sum > 0 else 0.0
    p_05 = np.mean(np.array(ious) >= 0.50) * 100

    # ---------------------------------------------------------
    # Part 2: Compute COCO AP
    # ---------------------------------------------------------
    print("\nCalculating COCO AP...")

    gt_data_modified = gt_data.copy()
    gt_data_modified['categories'] = [{"id": 1, "name": "object"}]
    for ann in gt_data_modified['annotations']:
        ann['category_id'] = 1

    coco_gt = COCO()
    coco_gt.dataset = gt_data_modified
    coco_gt.createIndex()

    if len(coco_results) > 0:
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'segm')

        img_ids = sorted(list(set(x['image_id'] for x in pred_data if 'image_id' in x)))
        coco_eval.params.imgIds = img_ids

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        ap_coco = coco_eval.stats[0] * 100
        ap_50 = coco_eval.stats[1] * 100
    else:
        ap_coco = 0.0
        ap_50 = 0.0

    # ---------------------------------------------------------
    # Print evaluation report
    # ---------------------------------------------------------
    W = 60
    print("\n" + "=" * W)
    print(f"  FULL EVALUATION REPORT (1to1)")
    print("=" * W)
    print(f"Total Samples: {len(ious)}")
    print("-" * W)
    print(f"{'Metric':<30} | {'Score (%)':<10}")
    print("-" * W)
    print(f"{'gIoU':<30} | {giou:.2f}")
    print(f"{'mIoU':<30} | {miou:.2f}")
    print(f"{'Precision @ 0.50':<30} | {p_05:.2f}")
    print("-" * W)
    print(f"{'AP (COCO 0.5:0.95)':<30} | {ap_coco:.2f}")
    print(f"{'AP @ 0.50':<30} | {ap_50:.2f}")
    print("=" * W)


# ============================================================================
# multi_inst Evaluation: align by image_id
# Strictly follows internal script metrics_multi_inst.py
# ============================================================================

def evaluate_multi_inst(gt_data, pred_data):
    """
    Multi-instance evaluation logic (1toAll / 1toN).
    Strictly follows metrics_multi_inst.py:
      - GT: aggregate all annotation masks per image_id via OR union
      - Pred: merge each pred item's predictions list into one mask;
              pred_results[image_id] is a list (one merged mask per annotation_id)
      - Evaluation: compute IoU between each pred mask and GT per image_id, then average
      - Iterate over gt_data['images'] as the evaluation unit
      - Compute gIoU, Precision@0.50
    """
    print("\n[multi_inst mode] Align by image_id")
    print("Reference: metrics_multi_inst.py")

    # ---------------------------------------------------------
    # Build GT masks: aggregate all annotations per image_id (OR union)
    # ---------------------------------------------------------
    print("Processing Ground Truth...")
    gt_masks = {}
    for ann in tqdm(gt_data['annotations'], desc="Loading GT"):
        image_id = ann['image_id']
        mask = safe_decode_rle(ann['segmentation'])
        if mask is None:
            continue
        if image_id not in gt_masks:
            gt_masks[image_id] = mask
        else:
            gt_masks[image_id] = gt_masks[image_id] | mask

    # ---------------------------------------------------------
    # Build Pred: merge each pred item's predictions list into one mask
    # pred_results[image_id] is a list (one merged mask per annotation_id)
    # Strictly follows metrics_multi_inst.py
    # ---------------------------------------------------------
    print("Processing Predictions...")
    pred_results = {}
    for item in tqdm(pred_data, desc="Loading Preds"):
        image_id = item['image_id']
        predictions = item.get('predictions', [])

        # Merge all predictions[*].rle of this pred item into a single mask
        combined_pred_mask = None
        if predictions:
            for p in predictions:
                rle = p['rle']
                mask = safe_decode_rle(rle)
                if combined_pred_mask is None:
                    combined_pred_mask = mask
                else:
                    combined_pred_mask = combined_pred_mask | mask

        if image_id not in pred_results:
            pred_results[image_id] = []
        pred_results[image_id].append(combined_pred_mask)

    # ---------------------------------------------------------
    # Compute metrics: iterate over gt_data['images']
    # For each image_id, compute IoU per pred mask vs GT, then average
    # Strictly follows metrics_multi_inst.py
    # ---------------------------------------------------------
    print("Calculating Metrics...")

    image_ious = []
    image_p50s = []

    for img_info in tqdm(gt_data['images'], desc="Evaluating"):
        image_id = img_info['id']

        mask_gt = gt_masks.get(image_id)
        if mask_gt is None:
            h, w = img_info['height'], img_info['width']
            mask_gt = np.zeros((h, w), dtype=bool)

        preds_list = pred_results.get(image_id, [])

        if not preds_list:
            # No predictions -> score 0
            image_ious.append(0.0)
            image_p50s.append(0.0)
            continue

        # Compute IoU for each pred mask of this image
        current_img_ious = []
        current_img_p50s = []

        for mask_pred in preds_list:
            if mask_pred is None:
                mask_pred = np.zeros_like(mask_gt, dtype=bool)

            if mask_pred.shape != mask_gt.shape:
                current_img_ious.append(0.0)
                current_img_p50s.append(0.0)
                continue

            intersection = (mask_pred & mask_gt).sum()
            union = (mask_pred | mask_gt).sum()
            iou = intersection / union if union > 0 else 0.0

            current_img_ious.append(iou)
            current_img_p50s.append(1.0 if iou >= 0.5 else 0.0)

        # Average across pred masks
        if current_img_ious:
            image_ious.append(np.mean(current_img_ious))
            image_p50s.append(np.mean(current_img_p50s))
        else:
            image_ious.append(0.0)
            image_p50s.append(0.0)

    if not image_ious:
        print("No images evaluated.")
        return

    giou = np.mean(image_ious) * 100
    p_50 = np.mean(image_p50s) * 100

    W = 60
    print("\n" + "=" * W)
    print(f"  FULL EVALUATION REPORT (multi_inst: 1toAll / 1toN)")
    print("=" * W)
    print(f"Total Images Evaluated: {len(image_ious)}")
    print("-" * W)
    print(f"{'Metric':<30} | {'Score (%)':<10}")
    print("-" * W)
    print(f"{'gIoU':<30} | {giou:.2f}")
    print(f"{'Precision @ 0.50':<30} | {p_50:.2f}")
    print("=" * W)


# ============================================================================
# Main Entry
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SAM3-I Evaluation Script\n'
                    'Inference output is unified as Format B (predictions list). '
                    'Evaluation mode is selected via --mode:\n'
                    '  1to1       -> align by annotation_id (metrics_005.py)\n'
                    '  multi_inst -> align by image_id (metrics_multi_inst.py)'
    )
    parser.add_argument('--gt_path', type=str, required=True,
                        help='Path to ground truth JSON file')
    parser.add_argument('--pred_path', type=str, required=True,
                        help='Path to prediction JSON file')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['1to1', 'multi_inst'],
help='Evaluation mode: 1to1 (HMPL-Instruct_1to1/ReasonSeg) or multi_inst (HMPL-Instruct_1toAll/HMPL-Instruct_1toN)')
    args = parser.parse_args()

    print(f"Loading Data...")
    print(f"  GT Path:   {args.gt_path}")
    print(f"  Pred Path: {args.pred_path}")
    print(f"  Mode:      {args.mode}")

    with open(args.gt_path, 'r') as f:
        gt_data = json.load(f)
    with open(args.pred_path, 'r') as f:
        pred_data = json.load(f)

    if args.mode == '1to1':
        evaluate_1to1(gt_data, pred_data)
    else:
        evaluate_multi_inst(gt_data, pred_data)


if __name__ == "__main__":
    main()