import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig

__all__ = [
    "PackedDropoutFn",
    "PackedDropout",
    "apply_packed_dropout",
    "get_model",
]


class PackedDropoutFn(torch.autograd.Function):
    """Packed-mask dropout storing boolean mask in a bit-packed uint8 tensor."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, p: float, training: bool):
        if not training or p == 0.0:
            ctx.save_for_backward(None)
            ctx.p = p
            return x
        keep_prob = 1 - p
        mask = torch.rand_like(x) < keep_prob  # bool tensor
        flat = mask.flatten()
        pad = (-flat.numel()) % 8
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])
        flat = flat.view(-1, 8)
        packed = sum((flat[:, i].byte() << i) for i in range(8)).contiguous()
        ctx.save_for_backward(packed)
        ctx.shape = x.shape
        ctx.p = p
        return x * mask.div(keep_prob)

    @staticmethod
    def backward(ctx, grad_out):
        packed, = ctx.saved_tensors
        if packed is None:
            return grad_out, None, None
        byte = packed.view(-1, 1)
        bits = (byte >> torch.arange(8, device=byte.device)) & 1
        mask = bits.flatten()[:grad_out.numel()].view(ctx.shape).bool()
        grad_in = grad_out * mask.div(1 - ctx.p)
        return grad_in, None, None


class PackedDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        return PackedDropoutFn.apply(x, self.p, self.training)


# Recursive replacement helper

def apply_packed_dropout(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, nn.Dropout):
            setattr(module, name, PackedDropout(child.p))
        else:
            apply_packed_dropout(child)


def get_model(run_cfg, tokenizer):
    """Return a HF model; applies Packed Mask replacement if requested."""
    model_name = run_cfg["model"].get("name", "bert-base-uncased")
    num_labels = run_cfg["dataset"].get("num_classes", 2)
    cfg = AutoConfig.from_pretrained(model_name, num_labels=num_labels, use_auth_token=None)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg, use_auth_token=None)

    if run_cfg["training"].get("use_pmdrop", False):
        apply_packed_dropout(model)
    return model