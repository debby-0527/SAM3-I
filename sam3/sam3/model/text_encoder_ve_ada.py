# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# Adapter-enhanced text encoder for SAM3

import torch
import torch.nn as nn
from typing import Optional, List, Union, Callable, Tuple
from collections import OrderedDict
import torch.nn.functional as F

from .model_misc import LayerScale


class AdapterResidualBlock(nn.Module):
    """Residual attention block with adapter."""
    
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        adapter_config: Optional[dict] = None,
    ):
        super().__init__()
        
        # Original attention mechanism
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = norm_layer(d_model)
        self.ln_2 = norm_layer(d_model)
        
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        # Original MLP
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model)),
            ])
        )
        
        # Adapter configuration
        self.adapter_config = adapter_config or {}
        self.use_adapter = adapter_config is not None
        
        if self.use_adapter:
            self._setup_adapters(d_model)
    
    def _setup_adapters(self, d_model: int):
        """Set up adapter layers."""
        adapter_dim = self.adapter_config.get("adapter_dim", 64)
        
        # Adapter 1: Multi-head attention + MLP
        self.adapter1 = MultiHeadMLPAdapter(
            d_model=d_model,
            adapter_dim=adapter_dim,
            num_heads=self.adapter_config.get("adapter_heads", 8)
        )
        self.adapter12 = MultiHeadMLPAdapter(
            d_model=d_model,
            adapter_dim=adapter_dim,
            num_heads=self.adapter_config.get("adapter_heads", 8)
        )
        # Adapter 2: Second adapter (serial connection)
        self.adapter2 = MultiHeadMLPAdapter(
            d_model=d_model,
            adapter_dim=adapter_dim,
            num_heads=self.adapter_config.get("adapter_heads", 8)
        )
    
    def adapter_forward(self, x: torch.Tensor, text_type: int) -> torch.Tensor:
        """Adapter forward pass, selecting path based on text type."""
        # return x
        if not self.use_adapter:
            return x
            
        # Select adapter path based on text type
        if text_type == 1:  # Type 1: bypass adapter
            return x
        elif text_type == 2:  # Type 2: first adapter only
            adapter_output = self.adapter1(x)
        elif text_type == 3:  # Type 3: serial through two adapters (no grad on first)
            with torch.no_grad():
                x1 = self.adapter12(x)
            adapter_output = self.adapter2(x1)
        elif text_type == 4:  # Type 4: serial through two adapters (with grad)
            x1 = self.adapter12(x)
            adapter_output = self.adapter2(x1)
        else:
            raise ValueError(f"Unsupported text type: {text_type}")
        
        # Residual connection + scaling
        # scale = self.adapter_config.get("adapter_scale", 1.0)
        # return x + scale * adapter_output
        return adapter_output
    
    def attention(self, q_x: torch.Tensor, k_x: Optional[torch.Tensor] = None, 
                 v_x: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x
        
        if attn_mask is not None and not attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.to(q_x.dtype)
            
        return self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]
    
    def forward(self, q_x: torch.Tensor, k_x: Optional[torch.Tensor] = None, 
                v_x: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None,
                text_type: Optional[int] = None) -> torch.Tensor:
        # Original Transformer forward pass
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        
        # Attention part
        attn_output = self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        x = q_x + self.ls_1(attn_output)
        
        # MLP part + Adapter
        mlp_output = self.mlp(self.ln_2(x))
        
        if self.use_adapter and text_type is not None:
            # Apply adapter to MLP output based on text type
            x = self.adapter_forward(x, text_type)
        
        x = x + self.ls_2(mlp_output)
        return x


class MultiHeadMLPAdapter(nn.Module):
    """Multi-head attention + MLP adapter module."""
    
    def __init__(self, d_model: int, adapter_dim: int, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.adapter_dim = adapter_dim
        
        # Multi-head attention part
        self.mha_adapter = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Down-up projection bottleneck
        self.down_proj = nn.Linear(d_model, adapter_dim)
        self.up_proj = nn.Linear(adapter_dim, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Activation function
        self.activation = nn.GELU()
        
        # Initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        # Initialize up-projection to near-zero, ensuring adapter approximates identity mapping initially
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        
        # Also initialize down-projection to zero for safety
        nn.init.zeros_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        
        # Initialize MHA output projection to zero
        nn.init.zeros_(self.mha_adapter.out_proj.weight)
        nn.init.zeros_(self.mha_adapter.out_proj.bias)
        
        # Also initialize MHA input projection to zero for safety
        if self.mha_adapter.in_proj_weight is not None:
            nn.init.zeros_(self.mha_adapter.in_proj_weight)
        if self.mha_adapter.in_proj_bias is not None:
            nn.init.zeros_(self.mha_adapter.in_proj_bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        
        # 1. Multi-head attention adapter (Pre-Norm)
        residual = x
        x_norm = self.norm1(x)
        attn_output, _ = self.mha_adapter(x_norm, x_norm, x_norm)
        x = residual + 1.0 * attn_output
        
        # 2. MLP adapter (bottleneck) (Pre-Norm)
        residual = x
        x_norm = self.norm2(x)
        mlp_output = self.down_proj(x_norm)
        mlp_output = self.activation(mlp_output)
        mlp_output = self.up_proj(mlp_output)
        x = residual + 1.0 * mlp_output
        
        return x


class AdapterTransformer(nn.Module):
    """Transformer with adapter."""
    
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        compile_mode: Optional[str] = None,
        use_act_checkpoint: bool = False,
        inst_stage: Optional[str] = None,
        adapter_config: Optional[dict] = None,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = use_act_checkpoint
        self.adapter_config = adapter_config
        
        # Use adapter-enhanced residual blocks
        self.resblocks = nn.ModuleList([
            AdapterResidualBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                adapter_config=adapter_config,
            )
            for _ in range(layers)
        ])

        if compile_mode is not None:
            self.forward = torch.compile(self.forward, mode=compile_mode, fullgraph=True)
            if self.grad_checkpointing:
                torch._dynamo.config.optimize_ddp = False

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, 
                text_type: Optional[int] = None) -> torch.Tensor:
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting() and self.training:
                x = torch.utils.checkpoint.checkpoint(r, x, None, None, attn_mask, text_type, use_reentrant=False)
            else:
                x = r(x, attn_mask=attn_mask, text_type=text_type)
        return x


class AdapterTextTransformer(nn.Module):
    """Text Transformer with adapter."""
    
    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        mlp_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        output_dim: int = 512,
        no_causal_mask: bool = False,
        pool_type: str = "argmax",
        proj_bias: bool = False,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        output_tokens: bool = False,
        use_ln_post: bool = True,
        compile_mode: Optional[str] = None,
        use_act_checkpoint: bool = False,
        inst_stage: Optional[str] = None,
        adapter_config: Optional[dict] = None,
    ):
        super().__init__()
        
        # Save adapter configuration
        self.adapter_config = adapter_config
        self.use_adapter = adapter_config is not None
        
        # Original components
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pool_type = pool_type
        self.output_tokens = output_tokens

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        
        # Use adapter-enhanced Transformer
        self.transformer = AdapterTransformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            compile_mode=compile_mode,
            use_act_checkpoint=use_act_checkpoint,
            inst_stage=inst_stage,
            adapter_config=adapter_config,
        )
        
        self.ln_final = norm_layer(width) if use_ln_post else nn.Identity()
        
        if no_causal_mask:
            self.attn_mask = None
        else:
            self.register_buffer("attn_mask", self.build_causal_mask(), persistent=False)
            
        if proj_bias:
            self.text_projection = nn.Linear(width, output_dim)
        else:
            self.text_projection = nn.Parameter(torch.empty(width, output_dim))
        
        # Initialize parameters
        self.init_parameters()
        
        # Freeze parameters (if using adapter)
        # if self.use_adapter:
        #     self._freeze_original_parameters(inst_stage=inst_stage)

    def _freeze_original_parameters(self, inst_stage=None):
        """Freeze original parameters, only train adapter."""
        if inst_stage[0] != "0" or inst_stage == "0_a":
            # Freeze all original parameters
            for name, param in self.named_parameters():
                if 'adapter' not in name:  # Only freeze non-adapter parameters
                    param.requires_grad = False
                    # print(f"Frozen: {name}")
                # else:
                #     print(f"Trainable: {name}")

    def init_parameters(self):
        """Initialize parameters."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        
        if hasattr(self, 'text_projection'):
            if isinstance(self.text_projection, nn.Parameter):
                nn.init.normal_(self.text_projection, std=self.width** -0.5)

    def build_causal_mask(self) -> torch.Tensor:
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(self, text: torch.Tensor, text_type: Optional[int]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # print(text)
        # print(text_type)
        # print('@'*100)
        seq_len = text.shape[1]
        x = self.token_embedding(text)
        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask[:seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len]
        x = self.transformer(x, attn_mask=attn_mask, text_type=text_type)
        x = self.ln_final(x)

        # Pooling (consistent with original implementation)
        # print(text)
        if self.pool_type == "argmax":
            pooled = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
            tokens = x
        elif self.pool_type == "first":
            pooled, tokens = x[:, 0], x[:, 1:]
        elif self.pool_type == "last":
            pooled, tokens = x[:, -1], x[:, :-1]
        else:
            pooled = tokens = x

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens
        return pooled


class AdapterVETextEncoder(nn.Module):
    """VETextEncoder with adapter - main interface class."""
    
    def __init__(
        self,
        d_model: int,
        tokenizer: Callable,
        width: int = 1024,
        heads: int = 16,
        layers: int = 24,
        context_length: int = 32,
        vocab_size: int = 49408,
        use_ln_post: bool = True,
        compile_mode: Optional[str] = None,
        use_act_checkpoint: bool = True,
        inst_stage: Optional[str] = None,
        adapter_config: Optional[dict] = None,
    ):
        super().__init__()
        
        self.context_length = context_length
        self.use_ln_post = use_ln_post
        self.tokenizer = tokenizer
        self.adapter_config = adapter_config

        # Use adapter-enhanced text encoder
        self.encoder = AdapterTextTransformer(
            context_length=context_length,
            vocab_size=vocab_size,
            width=width,
            heads=heads,
            layers=layers,
            output_tokens=True,  # We need tokens, not just pooled output
            use_ln_post=use_ln_post,
            compile_mode=compile_mode,
            use_act_checkpoint=use_act_checkpoint,
            inst_stage=inst_stage,
            adapter_config=adapter_config,
        )
        
        self.resizer = nn.Linear(self.encoder.width, d_model)

    def forward(
        self,
        text: Union[List[str], Tuple[torch.Tensor, torch.Tensor, dict]],
        text_type: Optional[int] = 1,
        input_boxes: Optional[List] = None,
        device: torch.device = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if isinstance(text[0], str):
            # Process string input
            assert input_boxes is None or len(input_boxes) == 0, "not supported"

            tokenized = self.tokenizer(text, context_length=self.context_length).to(device)
            text_attention_mask = (tokenized != 0).bool()

            inputs_embeds = self.encoder.token_embedding(tokenized)
            text_pool, text_memory = self.encoder(tokenized, text_type=text_type)

            assert text_memory.shape[1] == inputs_embeds.shape[1]
            text_attention_mask = text_attention_mask.ne(1)
            text_memory = text_memory.transpose(0, 1)
            text_memory_resized = self.resizer(text_memory)
        else:
            # Process pre-encoded input
            text_attention_mask, text_memory_resized, tokenized = text
            inputs_embeds = tokenized["inputs_embeds"]
            assert input_boxes is None or len(input_boxes) == 0

        return (
            text_attention_mask,
            text_memory_resized,
            inputs_embeds.transpose(0, 1),
        )

    def print_parameter_status(self):
        """Print parameter status."""
        total_params = 0
        trainable_params = 0
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                status = "Trainable"
            else:
                status = "Frozen"
            # print(f"{name}: {status} ({param.numel()} parameters)")
        
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"Frozen parameters: {total_params-trainable_params:,} ({(total_params-trainable_params)/total_params*100:.1f}%)")


# Factory function for convenience
def create_adapter_text_encoder(
    d_model: int,
    tokenizer: Callable,
    inst_stage: Optional[str] = None,
    adapter_config: Optional[dict] = None,
   **kwargs
) -> AdapterVETextEncoder:
    """
    Create an adapter-enhanced text encoder.
    
    Args:
        inst_stage: Enable adapter if not None
        adapter_config: Adapter configuration dictionary
    """
    
    # Use adapter configuration if inst_stage is not None
    if inst_stage[0] != "0" or inst_stage == "0_a":
        assert adapter_config is not None
        print(f"Using adapter text encoder with config: {adapter_config}")
    else:
        adapter_config = None
        print("Using standard text encoder (no adapter)")
    
    return AdapterVETextEncoder(
        d_model=d_model,
        tokenizer=tokenizer,
        inst_stage=inst_stage,
        adapter_config=adapter_config,
       **kwargs
    )