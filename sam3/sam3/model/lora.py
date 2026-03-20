import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.linear = linear_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(p=dropout)
        
        self.lora_A = nn.Parameter(torch.zeros(linear_layer.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, linear_layer.out_features))
        
        self.reset_parameters()
        
        # Freeze the original linear layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = self.linear(x)
        lora_result = (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        return result + lora_result

def apply_lora(model, r=8, alpha=16, dropout=0.0):
    """
    Recursively apply LoRA to all Linear layers in the model.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Skip classification/prediction heads; only replace Linear layers
            # in the Transformer and Text Encoder
            if "class_embed" in name or "bbox_embed" in name or "mask_embed" in name or "output" in name:
                continue
            
            # Replace Linear with LoRALinear
            lora_layer = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
            setattr(model, name, lora_layer)
        else:
            apply_lora(module, r=r, alpha=alpha, dropout=dropout)

def mark_only_lora_as_trainable(model):
    """
    Mark only LoRA parameters as trainable; freeze all others.
    """
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
