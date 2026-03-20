# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
This evaluator is meant for regular COCO mAP evaluation, for example on the COCO val set.

For Category mAP, we need the model to make predictions for all the categories on every single image.
In general, since the number of classes can be big, and the API model makes predictions individually for each pair (image, class),
we may need to split the inference process for a given image in several chunks.
"""

import logging
import json
import os
import tempfile
from collections import defaultdict

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sam3.train.utils.distributed import is_main_process

try:
    from tidecv import datasets, TIDE

    HAS_TIDE = True
except ImportError:
    HAS_TIDE = False
    print("WARNING: TIDE not installed. Detailed analysis will not be available.")


# the COCO detection metrics (https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L460-L471)
COCO_METRICS = [
    "AP",
    "AP_50",
    "AP_75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_maxDets@1",
    "AR_maxDets@10",
    "AR_maxDets@100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def convert_to_xywh(boxes):
    """Convert bounding boxes from xyxy format to xywh format."""
    xmin, ymin, xmax, ymax = boxes.unbind(-1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=-1)


class HeapElement:
    """Utility class to make a heap with a custom comparator"""

    def __init__(self, val):
        self.val = val

    def __lt__(self, other):
        return self.val["score"] < other.val["score"]


class COCOevalCustom(COCOeval):
    """
    This is a slightly modified version of the original COCO API with added support for positive split evaluation.
    """

    def __init__(
        self, cocoGt=None, cocoDt=None, iouType="segm", dt_only_positive=False
    ):
        super().__init__(cocoGt, cocoDt, iouType)
        self.dt_only_positive = dt_only_positive

    def _prepare(self):
        """
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        """

        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann["segmentation"] = rle

        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(
                self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
            )
            dts = self.cocoDt.loadAnns(
                self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
            )
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == "segm":
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt["ignore"] = gt["ignore"] if "ignore" in gt else 0
            gt["ignore"] = "iscrowd" in gt and gt["iscrowd"]
            if p.iouType == "keypoints":
                gt["ignore"] = (gt["num_keypoints"] == 0) or gt["ignore"]
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation

        _gts_cat_ids = defaultdict(set)  # gt for evaluation on positive split
        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)
            _gts_cat_ids[gt["image_id"]].add(gt["category_id"])

        #### BEGIN MODIFICATION ####
        for dt in dts:
            if (
                self.dt_only_positive
                and dt["category_id"] not in _gts_cat_ids[dt["image_id"]]
            ):
                continue
            self._dts[dt["image_id"], dt["category_id"]].append(dt)
        #### END MODIFICATION ####
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results


class CocoEvaluatorOfflineWithPredFileEvaluators:
    def __init__(
        self,
        gt_path,
        tide: bool = True,
        iou_type: str = "bbox",
        positive_split=False,
    ):
        self.gt_path = gt_path
        self.tide_enabled = HAS_TIDE and tide
        self.positive_split = positive_split
        self.iou_type = iou_type

    def evaluate(self, dumped_file):
        if not is_main_process():
            return {}

        logging.info("OfflineCoco evaluator: Loading groundtruth")
        
        real_gt_path = self.gt_path
        if isinstance(self.gt_path, (list, tuple)):
            logging.info(f"Merging {len(self.gt_path)} GT files...")
            merged_gt = {"images": [], "annotations": [], "categories": []}
            categories_seen = set()
            images_seen = set()
            
            for i, path in enumerate(self.gt_path):
                logging.info(f"Loading {path}")
                with open(path, 'r') as f:
                    data = json.load(f)
                    
                    # Copy info and licenses from the first GT file or provide defaults
                    if i == 0:
                        if "info" in data:
                            merged_gt["info"] = data["info"]
                        else:
                            merged_gt["info"] = {"description": "Merged Dataset", "url": "", "version": "1.0", "year": 2025, "contributor": "", "date_created": ""}
                            
                        if "licenses" in data:
                            merged_gt["licenses"] = data["licenses"]
                        else:
                            merged_gt["licenses"] = []
                    
                    # Merge images
                    for img in data.get("images", []):
                        if img["id"] not in images_seen:
                            merged_gt["images"].append(img)
                            images_seen.add(img["id"])
                            
                    # Merge annotations
                    merged_gt["annotations"].extend(data.get("annotations", []))
                    
                    # Merge categories
                    for cat in data.get("categories", []):
                        if cat["id"] not in categories_seen:
                            merged_gt["categories"].append(cat)
                            categories_seen.add(cat["id"])
            
            fd, temp_gt_path = tempfile.mkstemp(suffix='.json', prefix='merged_gt_')
            with os.fdopen(fd, 'w') as f:
                json.dump(merged_gt, f)
            
            logging.info(f"Merged GT saved to {temp_gt_path}")
            self.gt = COCO(temp_gt_path)
            real_gt_path = temp_gt_path
        else:
            self.gt = COCO(self.gt_path)

        # Creating the result file
        logging.info("Coco evaluator: Creating the result file")
        
        # Filter predictions to match current ground truth
        # We keep this filter logic as a safeguard, but now self.gt should cover all datasets
        with open(str(dumped_file), 'r') as f:
            preds = json.load(f)
        
        gt_img_ids = set(self.gt.getImgIds())
        filtered_preds = [p for p in preds if p['image_id'] in gt_img_ids]
        
        if len(filtered_preds) < len(preds):
            logging.warning(f"Filtered {len(preds) - len(filtered_preds)} predictions that were not in Ground Truth.")
        
        filtered_dumped_file = str(dumped_file).replace('.json', f'_filtered.json')
        with open(filtered_dumped_file, 'w') as f:
            json.dump(filtered_preds, f)
            
        cocoDt = self.gt.loadRes(filtered_dumped_file)

        # Run the evaluation
        logging.info("Coco evaluator: Running evaluation")
        coco_eval = COCOevalCustom(
            self.gt, cocoDt, iouType=self.iou_type, dt_only_positive=self.positive_split
        )
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        outs = {}
        for i, value in enumerate(coco_eval.stats):
            outs[f"coco_eval_{self.iou_type}_{COCO_METRICS[i]}"] = value

        if self.tide_enabled:
            logging.info("Coco evaluator: Loading TIDE")
            self.tide_gt = datasets.COCO(real_gt_path)
            self.tide = TIDE(mode="mask" if self.iou_type == "segm" else "bbox")

            # Run TIDE
            logging.info("Coco evaluator: Running TIDE")
            self.tide.evaluate(
                self.tide_gt, datasets.COCOResult(str(filtered_dumped_file)), name="coco_eval"
            )
            self.tide.summarize()
            for k, v in self.tide.get_main_errors()["coco_eval"].items():
                outs[f"coco_eval_{self.iou_type}_TIDE_{k}"] = v

            for k, v in self.tide.get_special_errors()["coco_eval"].items():
                outs[f"coco_eval_{self.iou_type}_TIDE_{k}"] = v
        
        # Clean up temp file if we created one
        if isinstance(self.gt_path, (list, tuple)) and os.path.exists(real_gt_path):
            try:
                os.remove(real_gt_path)
            except OSError:
                pass

        return outs
