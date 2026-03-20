"""
SAM3-I Model Builder
====================
Streamlined version — maintains the exact same model construction and weight loading logic
as the original SAM3, while significantly simplifying hyperparameter control and code structure.

Key improvements:
  1. Centralized configuration using dataclass (adapter_config, inst_stage, etc.)
  2. Unified weight loading function `load_checkpoint`, replacing previously scattered loading functions
  3. Single entry point for freeze/unfreeze logic
  4. Removed redundant comments and dead code
"""

import gc
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from huggingface_hub import hf_hub_download
from iopath.common.file_io import g_pathmgr

# ============================================================================
# Model submodule imports
# ============================================================================
from sam3.model.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer_Ada,
    TransformerDecoderLayerv2,
    TransformerEncoderCrossAttention,
)
from sam3.model.encoder import (
    TransformerEncoderFusion,
    TransformerEncoderLayer,
    TransformerEncoderLayer_Ada,
)
from sam3.model.geometry_encoders import SequenceGeometryEncoder
from sam3.model.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from sam3.model.memory import CXBlock, SimpleFuser, SimpleMaskDownSampler, SimpleMaskEncoder
from sam3.model.model_misc import (
    DotProductScoring,
    MLP,
    MultiheadAttentionWrapper as MultiheadAttention,
    TransformerWrapper,
)
from sam3.model.necks import Sam3DualViTDetNeck
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from sam3.model.sam3_image import Sam3Image, Sam3ImageOnVideoMultiGPU
from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor
from sam3.model.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity
from sam3.model.sam3_video_predictor import Sam3VideoPredictorMultiGPU
from sam3.model.text_encoder_ve_ada import create_adapter_text_encoder
from sam3.model.tokenizer_ve import SimpleTokenizer
from sam3.model.vitdet import ViT
from sam3.model.vl_combiner import SAM3VLBackbone
from sam3.sam.transformer import RoPEAttention


# ============================================================================
# Global configuration
# ============================================================================

# Base paths (auto-derived, no hardcoding needed)
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SAM3_PKG_DIR = os.path.dirname(_CURRENT_DIR)  # sam3/ directory
_DEFAULT_ASSETS_DIR = os.path.join(_SAM3_PKG_DIR, "assets")

# HuggingFace model info
HF_MODEL_ID = "facebook/sam3"
HF_CKPT_NAME = "sam3.pt"
HF_CFG_NAME = "config.json"

# Model dimension constants
D_MODEL = 256
D_FEEDFORWARD = 2048
N_HEADS = 8
DROPOUT = 0.1
RESOLUTION = 1008


@dataclass
class AdapterConfig:
    """Adapter configuration."""
    adapter_dim: int = 64
    adapter_heads: int = 4
    adapter_scale: float = 1.0

    def to_dict(self) -> Dict:
        return {"adapter_dim": self.adapter_dim, "adapter_heads": self.adapter_heads, "adapter_scale": self.adapter_scale}


@dataclass
class ModelConfig:
    """Unified configuration for SAM3-I model building."""
    # Basic settings
    device: str = "cuda"
    eval_mode: bool = True
    compile: bool = False

    # Feature toggles
    enable_segmentation: bool = True
    enable_inst_interactivity: bool = False

    # Training stage control: "0" original, "0_a" adapter-only, "0_l" LoRA, "1_1" stage1, "1_2" stage2, "1_a"/"1_b"/"1_c"/"1_d" variants
    inst_stage: str = "0"

    # Adapter configuration
    adapter_config: AdapterConfig = field(default_factory=AdapterConfig)

    # Weight loading
    checkpoint_path: Optional[str] = None
    load_from_hf: bool = True
    bpe_path: Optional[str] = None

    # Training related
    strict_freeze: bool = True  # Strictly freeze non-adapter parameters
    copy_adapter: bool = False
    share_mlp_params: bool = False
    use_margin_loss: bool = False
    use_infonce_loss: bool = False

    @property
    def compile_mode(self):
        return "default" if self.compile else None

    @property
    def need_strict_freeze(self) -> bool:
        """Whether strict freezing is needed (stage != "0" or special mode)."""
        return self.inst_stage[0] != "0" or self.inst_stage in ["0_a", "0_l"]

    @property
    def adapter_dict(self) -> Dict:
        return self.adapter_config.to_dict()


# ============================================================================
# TF32 optimization
# ============================================================================

def _setup_tf32() -> None:
    """Enable TF32 acceleration for Ampere GPUs."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        if props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

_setup_tf32()


# ============================================================================
# Model component construction functions (identical to original logic)
# ============================================================================

def _create_position_encoding(precompute_resolution=None):
    return PositionEmbeddingSine(num_pos_feats=D_MODEL, normalize=True, scale=None, temperature=10000, precompute_resolution=precompute_resolution)


def _create_vit_backbone(compile_mode=None):
    return ViT(
        img_size=RESOLUTION, pretrain_img_size=336, patch_size=14, embed_dim=1024, depth=32,
        num_heads=16, mlp_ratio=4.625, norm_layer="LayerNorm", drop_path_rate=DROPOUT,
        qkv_bias=True, use_abs_pos=True, tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31), rel_pos_blocks=(), use_rope=True,
        use_interp_rope=True, window_size=24, pretrain_use_cls_token=True,
        retain_cls_token=False, ln_pre=True, ln_post=False, return_interm_layers=False,
        bias_patch_embed=False, compile_mode=compile_mode,
    )


def _create_vit_neck(position_encoding, vit_backbone, enable_inst_interactivity=False):
    return Sam3DualViTDetNeck(
        position_encoding=position_encoding, d_model=D_MODEL,
        scale_factors=[4.0, 2.0, 1.0, 0.5], trunk=vit_backbone,
        add_sam2_neck=enable_inst_interactivity,
    )


def _create_vl_backbone(vit_neck, text_encoder):
    return SAM3VLBackbone(visual=vit_neck, text=text_encoder, scalp=1)


def _mha(batch_first=False):
    """Create standard multi-head attention."""
    return MultiheadAttention(num_heads=N_HEADS, dropout=DROPOUT, embed_dim=D_MODEL, batch_first=batch_first)


def _create_transformer_encoder(adapter_config) -> TransformerEncoderFusion:
    encoder_layer = TransformerEncoderLayer_Ada(
        activation="relu", d_model=D_MODEL, dim_feedforward=D_FEEDFORWARD, dropout=DROPOUT,
        pos_enc_at_attn=True, pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False, pre_norm=True,
        self_attention=_mha(batch_first=True), cross_attention=_mha(batch_first=True),
        adapter_config=adapter_config,
    )
    return TransformerEncoderFusion(
        layer=encoder_layer, num_layers=6, d_model=D_MODEL, num_feature_levels=1,
        frozen=False, use_act_checkpoint=True, add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )


def _create_transformer_decoder(adapter_config) -> TransformerDecoder:
    decoder_layer = TransformerDecoderLayer_Ada(
        activation="relu", d_model=D_MODEL, dim_feedforward=D_FEEDFORWARD, dropout=DROPOUT,
        cross_attention=_mha(), n_heads=N_HEADS,
        use_text_cross_attention=True, adapter_config=adapter_config,
    )
    return TransformerDecoder(
        layer=decoder_layer, num_layers=6, num_queries=200, return_intermediate=True,
        box_refine=True, num_o2m_queries=0, dac=True, boxRPB="log", d_model=D_MODEL,
        frozen=False, interaction_layer=None, dac_use_selfatt_ln=True, resolution=RESOLUTION,
        stride=14, use_act_checkpoint=True, presence_token=True,
    )


def _create_dot_product_scoring():
    prompt_mlp = MLP(
        input_dim=D_MODEL, hidden_dim=D_FEEDFORWARD, output_dim=D_MODEL, num_layers=2,
        dropout=DROPOUT, residual=True, out_norm=nn.LayerNorm(D_MODEL),
    )
    return DotProductScoring(d_model=D_MODEL, d_proj=D_MODEL, prompt_mlp=prompt_mlp)


def _create_segmentation_head(compile_mode=None):
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3, interpolation_mode="nearest",
        hidden_dim=D_MODEL, compile_mode=compile_mode,
    )
    cross_attend_prompt = MultiheadAttention(num_heads=N_HEADS, dropout=0, embed_dim=D_MODEL)
    return UniversalSegmentationHead(
        hidden_dim=D_MODEL, upsampling_stages=3, aux_masks=False, presence_head=False,
        dot_product_scorer=None, act_ckpt=True, cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )


def _create_geometry_encoder():
    geo_pos_enc = _create_position_encoding()
    geo_layer = TransformerEncoderLayer(
        activation="relu", d_model=D_MODEL, dim_feedforward=D_FEEDFORWARD, dropout=DROPOUT,
        pos_enc_at_attn=False, pre_norm=True,
        self_attention=_mha(), pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True, cross_attention=_mha(),
    )
    return SequenceGeometryEncoder(
        pos_enc=geo_pos_enc, encode_boxes_as_points=False,
        points_direct_project=True, points_pool=True, points_pos_enc=True,
        boxes_direct_project=True, boxes_pool=True, boxes_pos_enc=True,
        d_model=D_MODEL, num_layers=3, layer=geo_layer, use_act_ckpt=True,
        add_cls=True, add_post_encode_proj=True,
    )


def _create_text_encoder(bpe_path: str, inst_stage="0", adapter_config=None):
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    return create_adapter_text_encoder(
        d_model=D_MODEL, tokenizer=tokenizer, inst_stage=inst_stage,
        adapter_config=adapter_config, width=1024, heads=16, layers=24,
    )


def _create_vision_backbone(compile_mode=None, enable_inst_interactivity=True) -> Sam3DualViTDetNeck:
    position_encoding = _create_position_encoding(precompute_resolution=RESOLUTION)
    vit_backbone = _create_vit_backbone(compile_mode=compile_mode)
    return _create_vit_neck(position_encoding, vit_backbone, enable_inst_interactivity=enable_inst_interactivity)


def _create_sam3_transformer(adapter_config, has_presence_token=True) -> TransformerWrapper:
    encoder = _create_transformer_encoder(adapter_config=adapter_config)
    decoder = _create_transformer_decoder(adapter_config=adapter_config)
    return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=D_MODEL)


# ============================================================================
# Tracker component construction
# ============================================================================

def _create_tracker_maskmem_backbone():
    position_encoding = PositionEmbeddingSine(num_pos_feats=64, normalize=True, scale=None, temperature=10000, precompute_resolution=RESOLUTION)
    mask_downsampler = SimpleMaskDownSampler(kernel_size=3, stride=2, padding=1, interpol_size=[1152, 1152])
    cx_block_layer = CXBlock(dim=D_MODEL, kernel_size=7, padding=3, layer_scale_init_value=1e-6, use_dwconv=True)
    fuser = SimpleFuser(layer=cx_block_layer, num_layers=2)
    return SimpleMaskEncoder(out_dim=64, position_encoding=position_encoding, mask_downsampler=mask_downsampler, fuser=fuser)


def _create_tracker_transformer():
    feat_sizes = [RESOLUTION // 14, RESOLUTION // 14]  # [72, 72]
    self_attention = RoPEAttention(
        embedding_dim=D_MODEL, num_heads=1, downsample_rate=1, dropout=DROPOUT,
        rope_theta=10000.0, feat_sizes=feat_sizes, use_fa3=False, use_rope_real=False,
    )
    cross_attention = RoPEAttention(
        embedding_dim=D_MODEL, num_heads=1, downsample_rate=1, dropout=DROPOUT,
        kv_in_dim=64, rope_theta=10000.0, feat_sizes=feat_sizes,
        rope_k_repeat=True, use_fa3=False, use_rope_real=False,
    )
    encoder_layer = TransformerDecoderLayerv2(
        cross_attention_first=False, activation="relu", dim_feedforward=D_FEEDFORWARD, dropout=DROPOUT,
        pos_enc_at_attn=False, pre_norm=True, self_attention=self_attention,
        d_model=D_MODEL, pos_enc_at_cross_attn_keys=True, pos_enc_at_cross_attn_queries=False,
        cross_attention=cross_attention,
    )
    encoder = TransformerEncoderCrossAttention(
        remove_cross_attention_layers=[], batch_first=True, d_model=D_MODEL,
        frozen=False, pos_enc_at_input=True, layer=encoder_layer,
        num_layers=4, use_act_checkpoint=False,
    )
    return TransformerWrapper(encoder=encoder, decoder=None, d_model=D_MODEL)


def build_tracker(apply_temporal_disambiguation: bool, with_backbone: bool = False, compile_mode=None) -> Sam3TrackerPredictor:
    """Build SAM3 Tracker module."""
    maskmem_backbone = _create_tracker_maskmem_backbone()
    transformer = _create_tracker_transformer()
    backbone = None
    if with_backbone:
        vision_backbone = _create_vision_backbone(compile_mode=compile_mode)
        backbone = SAM3VLBackbone(scalp=1, visual=vision_backbone, text=None)
    return Sam3TrackerPredictor(
        image_size=RESOLUTION, num_maskmem=7, backbone=backbone, backbone_stride=14,
        transformer=transformer, maskmem_backbone=maskmem_backbone,
        multimask_output_in_sam=True, forward_backbone_per_frame_for_eval=True,
        trim_past_non_cond_mem_for_eval=False, multimask_output_for_tracking=True,
        multimask_min_pt_num=0, multimask_max_pt_num=1,
        always_start_from_first_ann_frame=False,
        non_overlap_masks_for_mem_enc=False, non_overlap_masks_for_output=False,
        max_cond_frames_in_attn=4, offload_output_to_cpu_for_eval=False,
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        clear_non_cond_mem_around_input=True, fill_hole_area=0,
        use_memory_selection=apply_temporal_disambiguation,
    )


# ============================================================================
# Unified weight loading logic
# ============================================================================

def _get_rank() -> int:
    """Get the rank of the current process."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _stagger_load():
    """Stagger multi-process loading to avoid IO spikes."""
    if dist.is_available() and dist.is_initialized():
        time.sleep((dist.get_rank() % 8) * 1.0)


def _load_raw_checkpoint(checkpoint_path: str) -> dict:
    """Load raw checkpoint from file and extract state_dict."""
    _stagger_load()
    print(f"[Rank {_get_rank()}] Loading checkpoint: {checkpoint_path}")
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=False)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
        del ckpt
    else:
        state_dict = ckpt
    gc.collect()
    return state_dict


def _filter_matching_weights(ckpt_state: dict, model_state: dict) -> dict:
    """Filter weights with matching shapes."""
    filtered = {}
    for k, v in ckpt_state.items():
        if k in model_state and v.shape == model_state[k].shape:
            filtered[k] = v
        elif k in model_state:
            print(f"  Skipping weight {k}: shape mismatch {v.shape} vs {model_state[k].shape}")
    return filtered


def _remap_detector_tracker_keys(ckpt_state: dict, has_tracker: bool) -> dict:
    """Remap detector.xxx / tracker.xxx prefixes to model internal key names."""
    remapped = {}
    for k, v in ckpt_state.items():
        if "detector" in k:
            remapped[k.replace("detector.", "")] = v
        elif "tracker" in k and has_tracker:
            remapped[k.replace("tracker.", "inst_interactive_predictor.model.")] = v
    return remapped


def _sync_segmentation_head_adapters(model, from_adapter1=False, from_mask_predictor=True):
    """Synchronize segmentation_head adapter weights.
    
    Args:
        model: SAM3 model
        from_adapter1: If True, initialize adapter2 from adapter1
        from_mask_predictor: If True, initialize adapter1 and adapter2 from original mask_predictor
    """
    seg_head = getattr(model, 'segmentation_head', None)
    if seg_head is None:
        return
    if not (hasattr(seg_head, 'mask_predictor') and hasattr(seg_head, 'mask_predictor_adapter1')):
        return

    if from_mask_predictor:
        seg_head.mask_predictor_adapter1.load_state_dict(seg_head.mask_predictor.state_dict())
        seg_head.mask_predictor_adapter2.load_state_dict(seg_head.mask_predictor.state_dict())
        print("  ✓ Synced adapter1/adapter2 from mask_predictor")
    elif from_adapter1:
        seg_head.mask_predictor_adapter2.load_state_dict(seg_head.mask_predictor_adapter1.state_dict())
        print("  ✓ Synced adapter2 from adapter1")


def _zero_init_adapters(model):
    """Zero-initialize all Adapter module output projections (identity mapping)."""
    count = 0
    for name, module in model.named_modules():
        is_adapter = False
        if hasattr(module, 'up_proj') and isinstance(module.up_proj, nn.Linear):
            nn.init.zeros_(module.up_proj.weight)
            nn.init.zeros_(module.up_proj.bias)
            is_adapter = True
        if hasattr(module, 'up_conv') and isinstance(module.up_conv, nn.Conv2d):
            nn.init.zeros_(module.up_conv.weight)
            if module.up_conv.bias is not None:
                nn.init.zeros_(module.up_conv.bias)
            is_adapter = True
        if hasattr(module, 'mha_adapter') and isinstance(module.mha_adapter, nn.MultiheadAttention):
            if hasattr(module.mha_adapter, 'out_proj'):
                nn.init.zeros_(module.mha_adapter.out_proj.weight)
                nn.init.zeros_(module.mha_adapter.out_proj.bias)
            is_adapter = True
        if is_adapter:
            count += 1
    print(f"  ✓ Zero-initialized {count} Adapter outputs")


def _copy_adapter1_to_adapter12(model, ckpt_state: dict):
    """Copy adapter1 weights to adapter12."""
    model_state = model.state_dict()
    adapter12_keys = [k for k in model_state.keys() if 'adapter12' in k]
    copied = 0
    for a12_key in adapter12_keys:
        a1_key = a12_key.replace('adapter12', 'adapter1')
        # Note: cannot use `or` operator as Tensor boolean evaluation would raise an error
        src = ckpt_state.get(a1_key)
        if src is None:
            src = model_state.get(a1_key)
        if src is not None and src.shape == model_state[a12_key].shape:
            with torch.no_grad():
                model_state[a12_key].copy_(src)
            copied += 1
    print(f"  ✓ adapter1 → adapter12 copied {copied} weights")


def _zero_adapter2_non_seg(model):
    """Zero out adapter2 parameters outside segmentation_head."""
    model_state = model.state_dict()
    zeroed = 0
    for key in model_state:
        if 'adapter2' in key and 'segmentation_head' not in key and ('weight' in key or 'bias' in key):
            with torch.no_grad():
                model_state[key].zero_()
            zeroed += 1
    print(f"  ✓ adapter2 (non-seg) zeroed {zeroed} parameters")


def load_checkpoint(model, cfg: ModelConfig):
    """
    Unified weight loading entry point.
    
    Loading strategy:
    1. If checkpoint_path exists → load directly (may contain adapter weights)
    2. If load_from_hf → download base weights from HuggingFace, then add adapters
    3. Neither → do not load weights
    
    For stage migration (e.g., 1_1 → 1_2), perform special initialization.
    """
    checkpoint_path = cfg.checkpoint_path
    strict_freeze = cfg.need_strict_freeze

    # ---- Determine target stage for adapter text encoder ----
    if cfg.inst_stage == "0_l":
        target_te_stage = "0"
    elif cfg.inst_stage == "0_a":
        target_te_stage = "0_a"
    elif cfg.inst_stage[0] != "0":
        target_te_stage = "1_1"
    else:
        target_te_stage = "0"

    should_init_adapter = False  # Whether adapter needs zero-initialization

    if checkpoint_path is not None:
        # ============ Case 1: Load from local checkpoint ============
        ckpt_state = _load_raw_checkpoint(checkpoint_path)

        # Check if special initialization is needed (stage migration)
        need_special_init = False
        if cfg.inst_stage == "1_2":
            if "sam3i_1-2" not in checkpoint_path and "sam3i_3" not in checkpoint_path:
                need_special_init = True
                print(f"  Detected stage migration → 1_2, performing special initialization")

        # Replace text encoder (add adapter)
        _replace_text_encoder(model, cfg.bpe_path, target_te_stage, cfg.adapter_dict)

        # Load weights
        model_state = model.state_dict()
        filtered = _filter_matching_weights(ckpt_state, model_state)
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        has_adapter_in_ckpt = any('adapter' in k for k in filtered)
        print(f"  Loaded {len(filtered)} weights, {len(missing)} missing")

        # Handle segmentation head adapter synchronization
        if need_special_init:
            has_a1 = any('segmentation_head.mask_predictor_adapter1' in k for k in ckpt_state)
            if not has_a1:
                _sync_segmentation_head_adapters(model, from_mask_predictor=True)
            else:
                _sync_segmentation_head_adapters(model, from_adapter1=True, from_mask_predictor=False)
            _copy_adapter1_to_adapter12(model, ckpt_state)
            _zero_adapter2_non_seg(model)
            should_init_adapter = not has_adapter_in_ckpt
        else:
            should_init_adapter = not has_adapter_in_ckpt

        del ckpt_state, filtered
        gc.collect()

    elif cfg.load_from_hf:
        # ============ Case 2: Download from HuggingFace ============
        checkpoint_path = _download_ckpt_from_hf()
        ckpt_state = _load_raw_checkpoint(checkpoint_path)

        # HF weights need detector/tracker prefix remapping
        has_tracker = model.inst_interactive_predictor is not None
        remapped = _remap_detector_tracker_keys(ckpt_state, has_tracker)
        model_state = model.state_dict()
        filtered = _filter_matching_weights(remapped, model_state)
        missing, _ = model.load_state_dict(filtered, strict=False)

        # Check critical missing weights
        critical = [k for k in missing if 'adapter' not in k]
        if critical:
            print(f"  ⚠ Critical weights missing: {len(critical)}: {critical[:5]}")

        print(f"  ✓ Loaded {len(filtered)} base weights from HF")

        # Replace text encoder (add adapter)
        _replace_text_encoder(model, cfg.bpe_path, target_te_stage, cfg.adapter_dict)

        # HF weights never contain adapters, need initialization
        _sync_segmentation_head_adapters(model, from_mask_predictor=True)
        should_init_adapter = True

        del ckpt_state, remapped, filtered
        gc.collect()

    # ---- Freeze non-adapter parameters ----
    if strict_freeze:
        frozen, trainable = _freeze_non_adapter_params(model)
        print(f"  Freeze mode: {frozen} params frozen, {trainable} trainable")

    # ---- Adapter initialization (only when checkpoint has no adapter) ----
    if should_init_adapter:
        print("  Performing Adapter zero-initialization...")
        _sync_segmentation_head_adapters(model, from_mask_predictor=True)
        _zero_init_adapters(model)
    else:
        print("  Skipping Adapter initialization (using checkpoint weights)")

    model.has_adapter = (target_te_stage != "0")


def _replace_text_encoder(model, bpe_path: str, target_stage: str, adapter_config: dict):
    """Replace the text encoder in the backbone (add/remove adapter)."""
    old_state = None
    if hasattr(model.backbone, 'text') and model.backbone.text is not None:
        old_state = model.backbone.text.state_dict()

    new_encoder = _create_text_encoder(bpe_path, inst_stage=target_stage, adapter_config=adapter_config)
    if old_state is not None:
        new_encoder.load_state_dict(old_state, strict=False)
    model.backbone.text = new_encoder


# ============================================================================
# Freeze logic
# ============================================================================

# Trainable parameter keyword whitelist
TRAINABLE_KEYWORDS = ('adapter', 'simple_query_proj', 'complex_query_proj', 'concept_proj')


def _freeze_non_adapter_params(model) -> tuple:
    """Freeze all non-adapter parameters, returns (frozen_count, trainable_count)."""
    frozen, trainable = 0, 0
    for name, param in model.named_parameters():
        if any(kw in name for kw in TRAINABLE_KEYWORDS):
            param.requires_grad = True
            trainable += 1
        else:
            param.requires_grad = False
            param.data.requires_grad = False
            frozen += 1
    return frozen, trainable


# ============================================================================
# HuggingFace download
# ============================================================================

def _download_ckpt_from_hf() -> str:
    _ = hf_hub_download(repo_id=HF_MODEL_ID, filename=HF_CFG_NAME)
    return hf_hub_download(repo_id=HF_MODEL_ID, filename=HF_CKPT_NAME)


# ============================================================================
# Default BPE path
# ============================================================================

def _default_bpe_path() -> str:
    bpe = os.path.join(_DEFAULT_ASSETS_DIR, "bpe_simple_vocab_16e6.txt.gz")
    if os.path.exists(bpe):
        return bpe
    # Fallback to working directory
    return os.path.join(os.getcwd(), "assets", "bpe_simple_vocab_16e6.txt.gz")


# ============================================================================
# Main build entry: Image Model
# ============================================================================

def build_sam3_image_model(
    bpe_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_mode=True,
    checkpoint_path=None,
    load_from_HF=True,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    compile=False,
    inst_stage=None,
    adapter_config=None,
    strict_freeze=True,
    copy_adapter=False,
    share_mlp_params=False,
    use_margin_loss=False,
    use_infonce_loss=False,
):
    """
    Build SAM3 image model.

    Args:
        bpe_path: BPE vocabulary path
        device: Device ("cuda" / "cpu")
        eval_mode: Whether to set evaluation mode
        checkpoint_path: Weight path (None to download from HF)
        load_from_HF: Whether to download from HuggingFace when checkpoint_path is None
        enable_segmentation: Whether to enable segmentation head
        enable_inst_interactivity: Whether to enable instance interactive predictor
        compile: Whether to use torch.compile
        inst_stage: Training stage identifier
        adapter_config: Adapter config dict (e.g., {"adapter_dim": 64, ...})
        strict_freeze: Whether to strictly freeze non-adapter parameters
        copy_adapter: Whether to copy adapter1 weights to adapter12
        share_mlp_params: Whether to share MLP parameters
        use_margin_loss: Whether to use margin loss
        use_infonce_loss: Whether to use InfoNCE loss
    
    Returns:
        Sam3Image model instance
    """
    # Build unified configuration
    if inst_stage is None:
        inst_stage = "0"
    if bpe_path is None:
        bpe_path = _default_bpe_path()

    adapter_cfg = AdapterConfig()
    if adapter_config is not None:
        adapter_cfg = AdapterConfig(
            adapter_dim=adapter_config.get("adapter_dim", 64),
            adapter_heads=adapter_config.get("adapter_heads", 4),
            adapter_scale=adapter_config.get("adapter_scale", 1.0),
        )

    cfg = ModelConfig(
        device=device, eval_mode=eval_mode, compile=compile,
        enable_segmentation=enable_segmentation,
        enable_inst_interactivity=enable_inst_interactivity,
        inst_stage=inst_stage, adapter_config=adapter_cfg,
        checkpoint_path=checkpoint_path, load_from_hf=load_from_HF,
        bpe_path=bpe_path, strict_freeze=strict_freeze,
        copy_adapter=copy_adapter, share_mlp_params=share_mlp_params,
        use_margin_loss=use_margin_loss, use_infonce_loss=use_infonce_loss,
    )

    # Override strict_freeze based on stage
    cfg.strict_freeze = cfg.need_strict_freeze

    # ---- 1. Build model components ----
    vision_encoder = _create_vision_backbone(
        compile_mode=cfg.compile_mode,
        enable_inst_interactivity=cfg.enable_inst_interactivity,
    )
    text_encoder = _create_text_encoder(bpe_path, inst_stage, cfg.adapter_dict)
    backbone = _create_vl_backbone(vision_encoder, text_encoder)
    transformer = _create_sam3_transformer(adapter_config=cfg.adapter_dict)
    dot_prod_scoring = _create_dot_product_scoring()
    segmentation_head = _create_segmentation_head(cfg.compile_mode) if cfg.enable_segmentation else None
    input_geometry_encoder = _create_geometry_encoder()

    if cfg.enable_inst_interactivity:
        tracker_base = build_tracker(apply_temporal_disambiguation=False)
        inst_predictor = SAM3InteractiveImagePredictor(tracker_base)
    else:
        inst_predictor = None

    # ---- 2. Assemble model ----
    from sam3.train.matcher import BinaryHungarianMatcherV2
    matcher = BinaryHungarianMatcherV2(
        focal=True, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0,
        alpha=0.25, gamma=2, stable=False,
    )
    model = Sam3Image(
        backbone=backbone, transformer=transformer,
        input_geometry_encoder=input_geometry_encoder,
        segmentation_head=segmentation_head, num_feature_levels=1,
        o2m_mask_predict=True, dot_prod_scoring=dot_prod_scoring,
        use_instance_query=False, multimask_output=True,
        inst_interactive_predictor=inst_predictor, inst_stage=inst_stage,
        matcher=matcher, share_mlp_params=share_mlp_params,
        use_margin_loss=use_margin_loss, use_infonce_loss=use_infonce_loss,
    )

    # ---- 3. Load weights ----
    load_checkpoint(model, cfg)

    # ---- 4. Device and mode ----
    if device == "cuda":
        model = model.cuda()
    if eval_mode:
        model.eval()

    # ---- 5. LoRA support ----
    if inst_stage == "0_l":
        from sam3.model.lora import apply_lora, mark_only_lora_as_trainable
        print("Applying LoRA...")
        apply_lora(model, r=16, alpha=32, dropout=0.05)
        mark_only_lora_as_trainable(model)

    return model


# ============================================================================
# Main build entry: Video Model
# ============================================================================

def build_sam3_video_model(
    checkpoint_path: Optional[str] = None,
    load_from_HF=True,
    bpe_path: Optional[str] = None,
    has_presence_token: bool = True,
    geo_encoder_use_img_cross_attn: bool = True,
    strict_state_dict_loading: bool = True,
    apply_temporal_disambiguation: bool = True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    compile=False,
    inst_stage="0",
    adapter_config=None,
) -> Sam3VideoInferenceWithInstanceInteractivity:
    """Build SAM3 video model."""
    if bpe_path is None:
        bpe_path = _default_bpe_path()

    tracker = build_tracker(apply_temporal_disambiguation=apply_temporal_disambiguation)
    visual_neck = _create_vision_backbone()
    text_encoder = _create_text_encoder(bpe_path, inst_stage=inst_stage, adapter_config=adapter_config)
    backbone = SAM3VLBackbone(scalp=1, visual=visual_neck, text=text_encoder)
    transformer = _create_sam3_transformer(adapter_config=adapter_config, has_presence_token=has_presence_token)
    segmentation_head = _create_segmentation_head()
    input_geometry_encoder = _create_geometry_encoder()

    main_dot_prod_mlp = MLP(
        input_dim=D_MODEL, hidden_dim=D_FEEDFORWARD, output_dim=D_MODEL, num_layers=2,
        dropout=DROPOUT, residual=True, out_norm=nn.LayerNorm(D_MODEL),
    )
    main_dot_prod_scoring = DotProductScoring(d_model=D_MODEL, d_proj=D_MODEL, prompt_mlp=main_dot_prod_mlp)

    detector = Sam3ImageOnVideoMultiGPU(
        num_feature_levels=1, backbone=backbone, transformer=transformer,
        segmentation_head=segmentation_head, semantic_segmentation_head=None,
        input_geometry_encoder=input_geometry_encoder,
        use_early_fusion=True, use_dot_prod_scoring=True,
        dot_prod_scoring=main_dot_prod_scoring,
        supervise_joint_box_scores=has_presence_token,
    )

    common_kwargs = dict(
        detector=detector, tracker=tracker,
        score_threshold_detection=0.5, assoc_iou_thresh=0.1,
        det_nms_thresh=0.1, new_det_thresh=0.7,
        suppress_unmatched_only_within_hotstart=True,
        min_trk_keep_alive=-1, max_trk_keep_alive=30, init_trk_keep_alive=30,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
        suppress_det_close_to_boundary=False,
        fill_hole_area=16, masklet_confirmation_enable=False,
        decrease_trk_keep_alive_for_empty_masklets=False,
        image_size=RESOLUTION, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5),
        compile_model=compile,
    )

    if apply_temporal_disambiguation:
        model = Sam3VideoInferenceWithInstanceInteractivity(
            **common_kwargs,
            hotstart_delay=15, hotstart_unmatch_thresh=8, hotstart_dup_thresh=8,
            recondition_every_nth_frame=16,
        )
    else:
        model = Sam3VideoInferenceWithInstanceInteractivity(
            **common_kwargs,
            hotstart_delay=0, hotstart_unmatch_thresh=0, hotstart_dup_thresh=0,
            recondition_every_nth_frame=0,
        )

    # Load weights
    if load_from_HF and checkpoint_path is None:
        checkpoint_path = _download_ckpt_from_hf()
    if checkpoint_path is not None:
        with g_pathmgr.open(checkpoint_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
        missing, unexpected = model.load_state_dict(ckpt, strict=strict_state_dict_loading)
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")

    model.to(device=device)
    return model


def build_sam3_video_predictor(*model_args, gpus_to_use=None, **model_kwargs):
    return Sam3VideoPredictorMultiGPU(*model_args, gpus_to_use=gpus_to_use, **model_kwargs)
