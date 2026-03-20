# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import torch

from sam3.model.model_misc import SAM3Output

from sam3.train.utils.distributed import get_world_size

from .loss_fns import CORE_LOSS_KEY, Det2TrkAssoc, Masks

import torch.nn.functional as F

class DummyLoss(torch.nn.Module):
    """A dummy loss that always returns 0 (as a placeholder for eval)"""

    def __init__(
        self,
        core_loss_key: str = CORE_LOSS_KEY,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()
        self.core_loss_key = core_loss_key
        self.device = torch.device(device)

    def forward(self, *args, **kwargs):
        return {self.core_loss_key: torch.tensor(0.0, device=self.device)}

    def accumulate(self, out_dict):
        """
        Called by iterative losses.
        """
        if self.core_loss_key not in out_dict:
            out_dict[self.core_loss_key] = torch.tensor(0.0, device=self.device)
        return out_dict


class Sam3LossWrapper(torch.nn.Module):
    def __init__(
        self,
        loss_fns_find,
        normalization="global",
        matcher=None,
        o2m_matcher=None,
        o2m_weight=1.0,
        use_o2m_matcher_on_o2m_aux=True,
        loss_fn_semantic_seg=None,
        normalize_by_valid_object_num=False,
        normalize_by_stage_num=False,
        scale_by_find_batch_size=False,
        use_kl=False,
        use_hard=False,
        use_infonce=True,
        use_margin=True,
        margin_loss_m=0.2,
        kl_loss_weight=50.0,
        enhanced_seg_loss_weight=50.0,
        infonce_loss_weight=20.0,
        infonce_loss_rho=0.5,
        margin_loss_weight=20.0,
    ):
        super().__init__()
        self.loss_fns_find = loss_fns_find
        assert normalization in ["global", "local", "none"]
        self.normalization = normalization
        self.normalize_by_valid_object_num = normalize_by_valid_object_num
        self.normalize_by_stage_num = normalize_by_stage_num
        self.matcher = matcher
        self.o2m_matcher = o2m_matcher
        self.o2m_weight = o2m_weight
        # whether to use the o2m matcher on the o2m queries in auxiliary outputs
        self.use_o2m_matcher_on_o2m_aux = use_o2m_matcher_on_o2m_aux
        self.loss_fn_semantic_seg = loss_fn_semantic_seg
        self.scale_by_find_batch_size = scale_by_find_batch_size
        self.use_kl = use_kl
        self.use_hard = use_hard
        self.use_infonce = use_infonce
        self.use_margin = use_margin
        self.margin_loss_m = margin_loss_m
        self.kl_loss_weight = kl_loss_weight
        self.enhanced_seg_loss_weight = enhanced_seg_loss_weight
        self.infonce_loss_weight = infonce_loss_weight
        self.infonce_loss_rho = infonce_loss_rho
        self.margin_loss_weight = margin_loss_weight
        self.base_ce_loss = torch.nn.CrossEntropyLoss()

    def _get_num_boxes(self, targets):
        # the average number of target boxes for loss normalization
        if self.normalize_by_valid_object_num:
            # valid boxes are those with non-zero height and width
            # (while padded invisible boxes are )
            boxes_hw = targets["boxes"].view(-1, 4)  # cx, cy, w, h
            num_boxes = (boxes_hw[:, 2:] > 0).all(dim=-1).sum().float()
        else:
            num_boxes = targets["num_boxes"].sum().float()
        if self.normalization == "global":
            torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1)
        elif self.normalization == "local":
            num_boxes = torch.clamp(num_boxes, min=1)
        elif self.normalization == "none":
            num_boxes = 1
        return num_boxes

    def compute_loss(self, nested_out, targets):
        num_boxes = self._get_num_boxes(targets)
        o2m_out_is_valid = nested_out.get("o2m_out_is_valid", None)
        o2m_target_is_valid_padded = nested_out.get("o2m_target_is_valid_padded", None)

        # Get a list of outputs, including auxiliary and first stage outputs
        output_list = [(nested_out, "", False)]  # (out, suffix, is_aux)
        if "aux_outputs" in nested_out:
            output_list.extend(
                (aux_out, f"_aux_{i}", True)
                for i, aux_out in enumerate(nested_out["aux_outputs"])
            )
        if "first_stage" in nested_out:
            output_list.append((nested_out["first_stage"], "_fs", True))

        # Compute all the requested losses
        losses = {}
        total_core_loss = 0.0
        for out, suffix, is_aux in output_list:
            # o2o matcher indices need to be computed by the model (as the video model requires
            # a specific way of matching free and locked indices beyond just calling the matcher)
            indices = out["indices"]
            has_o2m_out = "pred_logits_o2m" in out
            if has_o2m_out:
                o2m_out = {
                    k[: -len("_o2m")]: v for k, v in out.items() if k.endswith("_o2m")
                }
                # o2m targets are the same as the o2o targets (assuming repeat=1)
                o2m_targets = targets
                if self.use_o2m_matcher_on_o2m_aux or not is_aux:
                    o2m_indices = self.o2m_matcher(
                        o2m_out,
                        o2m_targets,
                        out_is_valid=o2m_out_is_valid,
                        target_is_valid_padded=o2m_target_is_valid_padded,
                    )
                else:
                    o2m_indices = self.matcher(
                        o2m_out,
                        o2m_targets,
                        out_is_valid=o2m_out_is_valid,
                        target_is_valid_padded=o2m_target_is_valid_padded,
                    )

            for loss_fn in self.loss_fns_find:
                l_dict = loss_fn(
                    outputs=out,
                    targets=targets,
                    indices=indices,
                    num_boxes=num_boxes,
                    is_aux=is_aux,
                )
                total_core_loss += l_dict.pop(CORE_LOSS_KEY)
                losses.update({f"{k}{suffix}": v for k, v in l_dict.items()})

                compute_o2m_loss = has_o2m_out
                # a special handling to allow turning off mask loss in o2m
                # (to be compatible with the original implementation)
                if isinstance(loss_fn, Masks):
                    compute_o2m_loss = compute_o2m_loss and "pred_masks" in o2m_out
                if isinstance(loss_fn, Det2TrkAssoc):
                    compute_o2m_loss = False  # Det2TrkAssoc does not support o2m
                if compute_o2m_loss:
                    l_dict = loss_fn(
                        outputs=o2m_out,
                        targets=o2m_targets,
                        indices=o2m_indices,
                        num_boxes=num_boxes,
                        is_aux=is_aux,
                    )
                    for k in l_dict:
                        l_dict[k] *= self.o2m_weight
                    total_core_loss += l_dict.pop(CORE_LOSS_KEY)
                    losses.update({f"{k}{suffix}_o2m": v for k, v in l_dict.items()})

        losses[CORE_LOSS_KEY] = total_core_loss
        return losses

    def forward(self, find_stages: SAM3Output, find_targets, ref_find_stages=None):
        # print(find_stages[0].keys())
        # print('&'*100)
        # exit()
        # print(find_targets[0].keys())
        if find_stages.loss_stages is not None:
            find_targets = [find_targets[i] for i in find_stages.loss_stages]
        with SAM3Output.iteration_mode(
            find_stages, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
        ) as find_stages:
            assert len(find_stages) == len(find_targets)
            total_losses = {}
            for stage_outputs, stage_targets in zip(find_stages, find_targets):
                stage_targets = [stage_targets] * len(stage_outputs)
                # If there are multiple steps within a stage, compute the loss for all of them (e.g. interactivity)
                for outputs, targets in zip(stage_outputs, stage_targets):
                    cur_losses = self.compute_loss(outputs, targets)

                    if self.loss_fn_semantic_seg is not None:
                        cur_losses_semantic = self.loss_fn_semantic_seg(
                            outputs, targets
                        )
                        cur_losses[CORE_LOSS_KEY] += cur_losses_semantic.pop(
                            CORE_LOSS_KEY
                        )
                        # make sure the semantic losses don't overlap with the find losses
                        assert set(cur_losses).isdisjoint(set(cur_losses_semantic))
                        cur_losses.update(cur_losses_semantic)

                    # Optionally, normalize the loss by the number of find stages (training video frames) so that
                    # image batches and video batches have similar loss scales. (Otherwise video batches would
                    # have a much higher loss scale due to summing the losses over all the find stages.)
                    if self.normalize_by_stage_num:
                        cur_losses[CORE_LOSS_KEY] /= len(find_stages)

                    if self.scale_by_find_batch_size:
                        bs = targets["num_boxes"].shape[0]
                        # sqrt scaling based on the "effective" batch size
                        cur_losses[CORE_LOSS_KEY] *= bs**0.5

                    for k, v in cur_losses.items():
                        if k not in total_losses:
                            total_losses[k] = v
                        else:
                            total_losses[k] += v
        # print(total_losses)
        # exit()
        # print(ref_find_stages)
        if ref_find_stages is not None:
            pred_A = find_stages[0]["semantic_seg"]
            pred_B = ref_find_stages[0]["semantic_seg"]
            # print(pred_A.shape, pred_B.shape)
            if self.use_kl:
                # print(find_stages[0].keys())
                # print(ref_find_stages[0].keys())
                loss_kl_A_to_B = self.compute_kl_loss_with_alignment(pred_A, pred_B)
                loss_kl_B_to_A = self.compute_kl_loss_with_alignment(pred_B, pred_A)
                kl_loss = self.kl_loss_weight * (loss_kl_A_to_B + loss_kl_B_to_A) / 2.0
                total_losses['loss_kl'] = kl_loss
                total_losses['loss_semantic_seg'] += kl_loss
                total_losses[CORE_LOSS_KEY] += kl_loss
                # print(total_losses['kl_loss'])
                # exit()
            # print(find_targets[0].keys())
            # print(find_targets[0]['masks'].shape)
            if self.use_hard:
                loss_hard_A = self.compute_enhanced_segmentation_loss(pred_A, pred_B, find_targets[0])
                loss_hard_B = self.compute_enhanced_segmentation_loss(pred_B, pred_A, find_targets[0])
                hard_loss = self.enhanced_seg_loss_weight * (loss_hard_A + loss_hard_B) / 2.0
                total_losses['loss_enhanced_seg'] = hard_loss
                total_losses['loss_semantic_seg'] += hard_loss
                total_losses[CORE_LOSS_KEY] += hard_loss
                # print(total_losses['hard_loss'])
                # exit()
            
            # InfoNCE and Margin Loss
            out_A = find_stages[0]
            out_B = ref_find_stages[0]
            if "proj_feat" in out_A and "proj_feat" in out_B:
                feat_A = out_A["proj_feat"]
                feat_B = out_B["proj_feat"]
                concept_texts = out_A["concept_texts"]
                
                if self.use_infonce:
                    loss_infonce = self.compute_infonce_loss(feat_A, feat_B, concept_texts)
                    total_losses['loss_infonce'] = self.infonce_loss_weight * loss_infonce
                    total_losses[CORE_LOSS_KEY] += total_losses['loss_infonce']
                
                if self.use_margin:
                    concept_feat = out_A["concept_proj_feat"]
                    loss_margin = self.compute_margin_loss(feat_A, feat_B, concept_feat, concept_texts, m=self.margin_loss_m)
                    total_losses['loss_margin'] = self.margin_loss_weight * loss_margin
                    total_losses[CORE_LOSS_KEY] += total_losses['loss_margin']

        # print(total_losses)
        # exit()
        return total_losses

    def weighted_infonce(self, sim_matrix, weights):
        # sim_matrix: [B, B]
        # weights: [B, B]
        
        max_val = sim_matrix.max(dim=1, keepdim=True)[0].detach()
        sim_matrix = sim_matrix - max_val
        
        exp_sim = torch.exp(sim_matrix)
        weighted_exp_sim = exp_sim * weights
        
        # sum over negatives (and positive, but positive weight is 1)
        # Add epsilon to prevent log(0)
        log_sum_exp = torch.log(weighted_exp_sim.sum(dim=1) + 1e-8)
        pos_sim = sim_matrix.diag()
        
        loss = (log_sum_exp - pos_sim).mean()
        return loss

    def compute_infonce_loss(self, feat_A, feat_B, concept_texts, temperature=0.1):
        feat_A = F.normalize(feat_A, dim=-1, eps=1e-8)
        feat_B = F.normalize(feat_B, dim=-1, eps=1e-8)
        
        B = feat_A.size(0)
        assert len(concept_texts) == B, f"concept_texts length {len(concept_texts)} does not match batch size {B}"

        # Concatenate features: [2B, D]
        feats = torch.cat([feat_A, feat_B], dim=0)
        
        # Similarity matrix: [2B, 2B]
        sim_matrix = torch.matmul(feats, feats.T) / temperature
        
        # Create labels for positive pairs
        # For each i in [0, B-1], positive is i+B
        # For each i in [B, 2B-1], positive is i-B
        labels = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)], dim=0).to(feats.device)
        
        # Mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(sim_matrix),
            1,
            torch.arange(2 * B, device=feats.device).view(-1, 1),
            0
        )
        
        # Mask for same text (potential false negatives)
        unique_texts = list(set(concept_texts))
        text_to_id = {t: i for i, t in enumerate(unique_texts)}
        text_ids = torch.tensor([text_to_id[t] for t in concept_texts], device=feats.device)
        text_ids_full = torch.cat([text_ids, text_ids], dim=0) # [2B]
        mask_same_text = (text_ids_full.unsqueeze(0) == text_ids_full.unsqueeze(1)) # [2B, 2B]
        
        # Initialize weights
        weights = torch.ones_like(sim_matrix)
        weights[mask_same_text] = self.infonce_loss_rho
        
        # Restore weight 1.0 for the true positive pairs
        weights.scatter_(1, labels.view(-1, 1), 1.0)
        
        # Apply mask for self-contrast (set weight to 0)
        weights = weights * logits_mask
        
        # Compute Loss
        # Max subtraction for stability
        max_val = sim_matrix.max(dim=1, keepdim=True)[0].detach()
        sim_matrix_sub = sim_matrix - max_val
        
        exp_sim = torch.exp(sim_matrix_sub)
        weighted_exp_sim = exp_sim * weights
        
        # Sum over all samples (except self, handled by weights=0)
        log_sum_exp = torch.log(weighted_exp_sim.sum(dim=1) + 1e-8)
        
        # Positive similarity
        pos_sim = torch.gather(sim_matrix_sub, 1, labels.view(-1, 1)).squeeze(1)
        
        loss = (log_sum_exp - pos_sim).mean()
        
        return loss

    def compute_margin_loss(self, feat_A, feat_B, concept_feat, concept_texts, m=0.2):
        sc_feat = (feat_A + feat_B) / 2.0
        sc_feat = F.normalize(sc_feat, dim=-1, eps=1e-8)
        concept_feat = F.normalize(concept_feat, dim=-1, eps=1e-8)
        
        sim_pos = (sc_feat * concept_feat).sum(dim=-1) # [B]
        
        loss = 0.0
        count = 0
        
        B = sc_feat.size(0)
        sim_matrix = torch.matmul(sc_feat, concept_feat.T) # [B, B]
        
        for i in range(B):
            neg_indices = []
            for j in range(B):
                if concept_texts[i] != concept_texts[j]:
                    neg_indices.append(j)
            
            if len(neg_indices) > 0:
                neg_sims = sim_matrix[i, neg_indices]
                hardest_neg_sim = neg_sims.max()
                
                l = F.relu(hardest_neg_sim - sim_pos[i] + m)
                loss += l
                count += 1
        
        if count > 0:
            loss = loss / count
        else:
            loss = torch.tensor(0.0, device=feat_A.device)
            
        return loss

    def compute_kl_loss_with_alignment(self, pred_A, pred_B, temperature=1.0):
        """
        Compute KL divergence loss for 2D maps (C=1), aligning A toward B (B is detached).
        
        Args:
            pred_A: First inference result [B, 1, H, W]
            pred_B: Second inference result [B, 1, H, W] 
            temperature: Temperature parameter for smoothing probability distribution
        """
        # Freeze B, only backpropagate gradients through A
        pred_B_detached = pred_B.detach()
        
        # Flatten 2D maps into spatial probability distributions
        B, C, H, W = pred_A.shape
        
        # Reshape to [B, H*W] spatial distribution
        flat_A = pred_A.view(B, -1)  # [B, H*W]
        flat_B = pred_B_detached.view(B, -1)  # [B, H*W]
        
        # Apply softmax to obtain spatial probability distributions
        # Use log_softmax for numerical stability
        log_prob_A = F.log_softmax(flat_A / temperature, dim=1)  # [B, H*W]
        prob_B = F.softmax(flat_B / temperature, dim=1)  # [B, H*W]
        
        # Compute KL divergence: align A toward B
        # KLDivLoss expects input to be log-probabilities and target to be probabilities
        kl_loss = F.kl_div(
            log_prob_A, 
            prob_B, 
            reduction='batchmean',
            log_target=False
        )
        
        return kl_loss

    def compute_enhanced_segmentation_loss(self, pred_A, pred_B, targets, 
                                     alpha=1.0, beta=0.5, gamma=0.3):
        """
        Difference-saliency-enhanced segmentation loss with automatic size alignment.
        
        Args:
            pred_A/pred_B: Prediction outputs from two branches
            targets: Ground-truth labels
            alpha: Weight for base segmentation loss
            beta: Weight for difference-focused loss  
            gamma: Weight for Dice loss
        """
        
        # Convert instance masks to semantic masks
        segments = targets["masks"].bool()
        semantic_targets = self.instance_masks_to_semantic_masks(segments, targets["num_boxes"])
        
        # Upsample predictions to match ground-truth size
        size = semantic_targets.shape[-2:]
        pred_A = F.interpolate(pred_A.float(), size=size, mode="bilinear", align_corners=False)
        pred_B = F.interpolate(pred_B.float(), size=size, mode="bilinear", align_corners=False)
        
        # Freeze branch B, used only for computing differences
        pred_B_detached = pred_B
        
        # Compute probability distribution divergence (JS divergence)
        # Detach inputs for weight calculation to prevent trivial solutions
        # Clamp probabilities to avoid log(0)
        prob_A_detached = torch.clamp(torch.sigmoid(pred_A.detach()), min=1e-6, max=1-1e-6)
        prob_B_detached = torch.clamp(torch.sigmoid(pred_B.detach()), min=1e-6, max=1-1e-6)
        
        # Compute JS divergence as the difference saliency map
        m = 0.5 * (prob_A_detached + prob_B_detached)
        # m is also safe because prob_A and prob_B are clamped
        
        js_div = 0.5 * (F.kl_div(prob_A_detached.log(), m, reduction='none') + 
                        F.kl_div(prob_B_detached.log(), m, reduction='none'))
        difference_scores = js_div.squeeze(1)  # [B, H, W]
        
        difference_weights = difference_scores
        
        # Pixel-wise loss computation
        pixelwise_loss = F.binary_cross_entropy_with_logits(
            pred_A.squeeze(1), 
            semantic_targets.float(), 
            reduction='none'
        )  # [B, H, W]
        
        # Difference-saliency weighting: per-pixel weighting
        weighted_pixel_loss_map = pixelwise_loss * difference_weights  # [B, H, W]
        
        # Compute total loss: average over spatial dims first, then over batch
        weighted_pixel_loss = weighted_pixel_loss_map.mean(dim=[1, 2]).mean()

        return weighted_pixel_loss
    
    def instance_masks_to_semantic_masks(
        self, instance_masks: torch.Tensor, num_instances: torch.Tensor
    ) -> torch.Tensor:
        """This function converts instance masks to semantic masks.
        It accepts a collapsed batch of instances masks (ie all instance masks are concatenated in a single tensor) and
        the number of instances in each image of the batch.
        It returns a mask with the same spatial dimensions as the input instance masks, where for each batch element the
        semantic mask is the union of all the instance masks in the batch element.

        If for a given batch element there are no instances (ie num_instances[i]==0), the corresponding semantic mask will be a tensor of zeros.

        Args:
            instance_masks (torch.Tensor): A tensor of shape (N, H, W) where N is the number of instances in the batch.
            num_instances (torch.Tensor): A tensor of shape (B,) where B is the batch size. It contains the number of instances
                in each image of the batch.

        Returns:
            torch.Tensor: A tensor of shape (B, H, W) where B is the batch size and H, W are the spatial dimensions of the
                input instance masks.
        """
        if num_instances.sum() == 0:
            # all negative batch, create a tensor of zeros (B, 1, 1)
            return num_instances.unsqueeze(-1).unsqueeze(-1)

        masks_per_query = torch.split(instance_masks, num_instances.tolist())

        return torch.stack([torch.any(masks, dim=0) for masks in masks_per_query], dim=0)


class TextContrastiveLossWrapper(torch.nn.Module):
    def __init__(
        self,
        feature_key="text_feature",
        temperature=0.1,
        contrast_mode="cosine",
        normalize_features=True,
    ):
        super().__init__()
        self.feature_key = feature_key
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.normalize_features = normalize_features
        
        self.first_inference_features = None
        
    def _normalize_features(self, features):
        """L2-normalize features."""
        if self.normalize_features:
            return torch.nn.functional.normalize(features, p=2, dim=-1)
        return features
    
    def _compute_contrastive_loss(self, features1, features2):
        """Compute contrastive loss between two feature sets."""
        if self.normalize_features:
            features1 = self._normalize_features(features1)
            features2 = self._normalize_features(features2)
        
        batch_size = features1.shape[0]
        
        if self.contrast_mode == "cosine":
            # Compute cosine similarity matrix
            similarity_matrix = torch.matmul(features1, features2.transpose(-2, -1)) / self.temperature
        else:  # L2 mode
            # Compute L2 distance and convert to similarity
            features1_expanded = features1.unsqueeze(1).expand(-1, batch_size, -1)
            features2_expanded = features2.unsqueeze(0).expand(batch_size, -1, -1)
            distances = torch.norm(features1_expanded - features2_expanded, p=2, dim=-1)
            similarity_matrix = -distances / self.temperature
        
        # Diagonal elements are positive pairs, others are negative pairs
        labels = torch.arange(batch_size, device=features1.device)
        
        # Compute contrastive loss (InfoNCE)
        loss = torch.nn.functional.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def forward(self, find_stages, find_targets):
        """
        Forward pass to compute contrastive loss.
        
        Args:
            find_stages: Model outputs containing results from two inference stages
            find_targets: Target data
        """
        assert len(find_stages) >= 2, "At least two inference stages are required for contrastive learning"

        print(find_stages[0].shape, find_stages[1].shape)
        exit()
        total_losses = {}
            
        # Get first inference stage results (no gradient)
        with torch.no_grad():
            first_stage_outputs = find_stages[0]
            if self.feature_key in first_stage_outputs:
                first_features = first_stage_outputs[self.feature_key]
                # Ensure feature shape is [batch_size, feature_dim]
                if first_features.dim() > 2:
                    first_features = first_features.view(first_features.size(0), -1)
            else:
                # Fallback feature extraction logic
                if "pred_logits" in first_stage_outputs:
                    first_features = first_stage_outputs["pred_logits"]
                elif "pred_boxes" in first_stage_outputs:
                    first_features = first_stage_outputs["pred_boxes"]
                else:
                    raise ValueError(f"Feature key not found in output: {self.feature_key}")
        
        # Get second inference stage results (with gradient)
        second_stage_outputs = find_stages[1][0]
        if self.feature_key in second_stage_outputs:
            second_features = second_stage_outputs[self.feature_key]
            if second_features.dim() > 2:
                second_features = second_features.view(second_features.size(0), -1)
        else:
            # Same fallback feature extraction logic
            if "pred_logits" in second_stage_outputs:
                second_features = second_stage_outputs["pred_logits"]
            elif "pred_boxes" in second_stage_outputs:
                second_features = second_stage_outputs["pred_boxes"]
            else:
                raise ValueError(f"Feature key not found in output: {self.feature_key}")
        
        # Ensure feature dimensions match
        if first_features.shape[-1] != second_features.shape[-1]:
            # Project to align dimensions if mismatched
            projection = torch.nn.Linear(second_features.shape[-1], first_features.shape[-1]).to(second_features.device)
            second_features = projection(second_features)
        
        # Compute contrastive loss
        contrast_loss = self._compute_contrastive_loss(first_features, second_features)
        
        total_losses["contrastive_loss"] = contrast_loss
        total_losses[CORE_LOSS_KEY] = contrast_loss
        
        return total_losses


