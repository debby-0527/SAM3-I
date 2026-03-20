#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""
SAM3-I Inference Script
=======================
Performs batched inference on multiple GPUs using COCO-style JSON annotations.
Supports concept / simple / complex prompt categories.

Usage:
    python scripts/inference.py \
        --json_path /path/to/sam3i_val.json \
        --image_root /path/to/images/ \
        --output_path ./predictions.json \
        --checkpoint_path /path/to/checkpoint.pt \
        --categories simple \
        --batch_size 32 \
        --gpus 0,1,2,3,4,5,6,7
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ============================================================================
# Project path initialization
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SAM3_PKG = os.path.join(PROJECT_ROOT, "sam3")
if SAM3_PKG not in sys.path:
    sys.path.insert(0, SAM3_PKG)

import sam3
from sam3 import build_sam3_image_model
from sam3.eval.postprocessors import PostProcessImage
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.collator import collate_fn_api as collate
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image as SAMImage,
    InferenceMetadata,
)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)

# ============================================================================
# Paths and constants
# ============================================================================
DEFAULT_BPE = os.path.join(SAM3_PKG, "assets", "bpe_simple_vocab_16e6.txt.gz")
DEFAULT_CHECKPOINT = os.environ.get("CHECKPOINT", None)

# Category-to-stage mapping
CATEGORY_TO_STAGE = {
    "concept": "0_0",
    "simple": "1_1",
    "complex": "1_2",
}

# Default inference parameters
DEFAULT_RESOLUTION = 1008
DEFAULT_NORM = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
DEFAULT_ADAPTER = {"adapter_dim": 64, "adapter_heads": 4, "adapter_scale": 1.0}


# ============================================================================
# Utility functions
# ============================================================================
def setup_torch():
    """Enable TF32 acceleration (Ampere+ GPUs)."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    """Convert a binary mask to COCO RLE format."""
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    assert mask.ndim == 2
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(rle, list):
        rle = rle[0]
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return {"size": list(rle["size"]), "counts": rle["counts"]}


def build_transform(resolution=DEFAULT_RESOLUTION, norm=None):
    """Build image transformation pipeline for inference."""
    norm = norm or DEFAULT_NORM
    return ComposeAPI(transforms=[
        RandomResizeAPI(sizes=resolution, max_size=resolution, square=True, consistent_transform=False),
        ToTensorAPI(),
        NormalizeAPI(mean=norm["mean"], std=norm["std"]),
    ])


# ============================================================================
# Dataset
# ============================================================================
@dataclass
class QueryItem:
    """A single query item (one image + one prompt)."""
    annotation_id: int
    image_id: int
    file_name: str
    prompt: Any       # str or dict
    prompt_type: str
    width: int
    height: int


class SAM3Dataset(Dataset):
    """Load images and queries from a COCO-style JSON file."""

    def __init__(self, json_path: str, image_root: str, transform=None, categories=None):
        self.image_root = image_root
        self.transform = transform
        self.categories = categories or ["concept", "simple", "complex"]

        with open(json_path) as f:
            data = json.load(f)
        self.images = {img["id"]: img for img in data["images"]}
        self.query_items: List[QueryItem] = []

        for ann in data["annotations"]:
            img = self.images[ann["image_id"]]
            text_input = img["text_inst_input"]

            if "concept" in self.categories:
                for concept in text_input.get("concept", []):
                    self.query_items.append(QueryItem(
                        ann["id"], ann["image_id"], img["file_name"],
                        concept, "concept", img["width"], img["height"],
                    ))

            if "simple" in self.categories:
                sq = text_input.get("simple_query", [])
                if sq:
                    ti = {
                        "concept": text_input.get("concept", []),
                        "simple_query": [sq[0], sq[0]],
                        "complex_query": text_input.get("complex_query", []).copy(),
                    }
                    self.query_items.append(QueryItem(
                        ann["id"], ann["image_id"], img["file_name"],
                        ti, "simple_query_0", img["width"], img["height"],
                    ))

            if "complex" in self.categories:
                cq = text_input.get("complex_query", [])
                if cq:
                    ti = {
                        "concept": text_input.get("concept", []),
                        "simple_query": text_input.get("simple_query", []).copy(),
                        "complex_query": [cq[0], cq[0]],
                    }
                    self.query_items.append(QueryItem(
                        ann["id"], ann["image_id"], img["file_name"],
                        ti, "complex_query_0", img["width"], img["height"],
                    ))

        print(f"Loaded {len(self.query_items)} query items (categories: {self.categories})")

    def __len__(self):
        return len(self.query_items)

    def __getitem__(self, idx):
        qi = self.query_items[idx]
        pil = Image.open(os.path.join(self.image_root, qi.file_name)).convert("RGB")
        w, h = pil.size

        dp = Datapoint(find_queries=[], images=[SAMImage(data=pil, objects=[], size=[h, w])])
        dp.find_queries.append(FindQueryLoaded(
            query_text=qi.prompt, image_id=0, object_ids_output=[], is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=idx, original_image_id=qi.image_id,
                original_category_id=1, original_size=[h, w],
                object_id=0, frame_index=0,
            ),
        ))
        if self.transform:
            dp = self.transform(dp)
        return dp, qi


def collate_fn(batch):
    """Custom collate function."""
    dps = [b[0] for b in batch]
    qis = [b[1] for b in batch]
    return collate(dps, dict_key="dummy")["dummy"], qis


# ============================================================================
# Result post-processing
# ============================================================================
def process_batch(output, batch, query_items, postprocessor):
    """Post-process model outputs and generate prediction results.

    Uses unified output format B (predictions list):
      Aligned with sam3_image_batched_inference_multi_inst.py.
      Each sample outputs a predictions list (each detected mask saved individually).
      Compatible with both metrics_005.py (1to1) and metrics_decouple.py (1toAll/1toN).
    """
    results = []
    processed = postprocessor.process_results(output, batch.find_metadatas)

    for idx, qi in enumerate(query_items):
        metadata = batch.find_metadatas[0]
        coco_id = metadata.coco_image_id
        if isinstance(coco_id, torch.Tensor):
            key = int(coco_id[idx]) if coco_id.dim() > 0 and len(coco_id) > idx else coco_id.item()
        else:
            key = int(coco_id)

        # Unified output format B: predictions list
        # Aligned with sam3_image_batched_inference_multi_inst.py
        predictions = []

        if key in processed:
            det = processed[key]
            if len(det["masks"]) > 0:
                masks = det["masks"].cpu().numpy() if isinstance(det["masks"], torch.Tensor) else det["masks"]
                scores = det["scores"].float().cpu().numpy() if isinstance(det["scores"], torch.Tensor) else det["scores"]

                if masks.dtype != np.bool_:
                    masks = masks > 0.5
                if masks.ndim == 4:
                    masks = masks.squeeze(1)

                # Each detected mask saved individually
                for i in range(len(masks)):
                    mask = masks[i]
                    score = scores[i]
                    rle = mask_to_rle(mask.astype(np.uint8))
                    predictions.append({"rle": rle, "score": float(score)})

        results.append({
            "annotation_id": qi.annotation_id,
            "image_id": qi.image_id,
            "file_name": qi.file_name,
            "prompt": qi.prompt,
            "prompt_type": qi.prompt_type,
            "predictions": predictions,
        })
    return results


# ============================================================================
# Single-GPU inference
# ============================================================================
def run_single_gpu(gpu_id, json_path, image_root, output_path, checkpoint_path,
                   batch_size, num_workers, start_idx, end_idx,
                   detection_threshold, categories):
    """Run inference on a single GPU."""
    setup_torch()
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    # Determine model stage
    stage = CATEGORY_TO_STAGE[categories[0]]

    # Build model
    model = build_sam3_image_model(
        bpe_path=DEFAULT_BPE if os.path.exists(DEFAULT_BPE) else None,
        checkpoint_path=checkpoint_path,
        eval_mode=True,
        enable_segmentation=True,
        device="cuda",
        load_from_HF=(checkpoint_path is None),
        inst_stage=stage,
        adapter_config=DEFAULT_ADAPTER,
    )
    model = model.to(device)
    model.eval()

    # Build data
    transform = build_transform()
    postprocessor = PostProcessImage(
        max_dets_per_img=-1, iou_type="segm",
        use_original_sizes_box=True, use_original_sizes_mask=True,
        convert_mask_to_rle=False, detection_threshold=detection_threshold, to_cpu=False,
    )

    dataset = SAM3Dataset(json_path, image_root, transform, categories)
    indices = list(range(start_idx, min(end_idx, len(dataset))))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    print(f"[GPU {gpu_id}] Processing {len(indices)} items...")
    all_results = []

    with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
        for batch, qis in tqdm(loader, desc=f"[GPU {gpu_id}]", position=gpu_id):
            batch = copy_data_to_device(batch, torch.device(device), non_blocking=True)
            output = model(batch, stage)
            all_results.extend(process_batch(output, batch, qis, postprocessor))

    gpu_file = output_path.replace(".json", f"_gpu{gpu_id}.json")
    with open(gpu_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[GPU {gpu_id}] Saved {len(all_results)} results -> {gpu_file}")


# ============================================================================
# Multi-GPU inference
# ============================================================================
def run_multi_gpu(args):
    """Run parallel inference across multiple GPUs with automatic sharding and merging."""
    gpu_ids = [int(x) for x in args.gpus.split(",")] if args.gpus else list(range(torch.cuda.device_count()))
    n = len(gpu_ids)

    # Calculate total data size
    transform = build_transform()
    total = len(SAM3Dataset(args.json_path, args.image_root, transform, args.categories))
    per_gpu = (total + n - 1) // n
    print(f"Total: {total} items, {n} GPUs, ~{per_gpu} items per GPU")

    mp.set_start_method("spawn", force=True)
    procs, files = [], []

    for i, gid in enumerate(gpu_ids):
        s, e = i * per_gpu, min((i + 1) * per_gpu, total)
        if s >= total:
            break
        files.append(args.output_path.replace(".json", f"_gpu{gid}.json"))
        p = mp.Process(target=run_single_gpu, args=(
            gid, args.json_path, args.image_root, args.output_path,
            args.checkpoint_path, args.batch_size, args.num_workers,
            s, e, args.detection_threshold, args.categories,
        ))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Process exited with code {p.exitcode}")

    all_results = []
    for f_path in files:
        with open(f_path) as f:
            all_results.extend(json.load(f))
        os.remove(f_path)

    # Sort by annotation_id and prompt type
    order = {"concept": 0, "simple_query_0": 1, "complex_query_0": 3}
    all_results.sort(key=lambda x: (x["annotation_id"], order.get(x["prompt_type"], 99)))

    if args.keep_top1:
        for item in all_results:
            preds = item.get("predictions", [])
            if len(preds) > 1:
                best = max(preds, key=lambda p: p.get("score", 0.0))
                item["predictions"] = [best]

    with open(args.output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Merge complete: {len(all_results)} results -> {args.output_path}")


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="SAM3-I Batch Inference")
    parser.add_argument("--json_path", required=True, help="Path to COCO-style JSON annotation")
    parser.add_argument("--image_root", required=True, help="Root directory of images")
    parser.add_argument("--output_path", required=True, help="Output JSON path")
    parser.add_argument("--checkpoint_path", default=DEFAULT_CHECKPOINT, help="Model checkpoint path (downloads from HuggingFace if not specified)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader workers (default: 2)")
    parser.add_argument("--gpus", default=None, help="GPU list, e.g. '0,1,2,3' (default: all)")
    parser.add_argument("--detection_threshold", type=float, default=0.5, help="Detection threshold (default: 0.5)")
    parser.add_argument("--categories", nargs="+", choices=["concept", "simple", "complex"], default=["simple"],
                        help="Inference categories: concept / simple / complex (default: simple)")
    parser.add_argument("--keep_top1", action="store_true", default=False,
                        help="Keep only the highest-score prediction per item")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    run_multi_gpu(args)


if __name__ == "__main__":
    main()
