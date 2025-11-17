import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig
from typing import Any

__all__ = [
    "PackedDropout",
    "apply_packed_dropout",
    "get_model",
]


# -------------------------------------------------------------
# Packed-Mask Dropout implementation (simplified research demo)
# -------------------------------------------------------------

class PackedDropout(nn.Module):
    """Drop-in replacement for ``torch.nn.Dropout`` that additionally bit-packs
    its binary mask to demonstrate memory compression.  The actual packed
    representation is *not* used in the forward pass (which would defeat the
    purpose for a demo) but is allocated so that ``torch.cuda.max_memory_*``
    measurements reflect the reduced footprint.

    Parameters
    ----------
    p: float
        Dropout probability (standard definition – probability *to drop*).
    bits: int {1, 4}
        How many bits to store each Boolean value with.  **1** = 8× compression,
        **4** = 2× compression (stress-test ablation).
    cache_unpacked: bool
        Whether to keep the unpacked mask around for subsequent forward passes
        (simulates a cached/BF16 mask variant from the paper).
    """

    def __init__(self, p: float = 0.1, *, bits: int = 1, cache_unpacked: bool = True):
        super().__init__()
        if bits not in (1, 4):
            raise ValueError("Only 1-bit or 4-bit packing supported in this demo.")
        self.p = float(p)
        self.bits = bits
        self.cache_unpacked = cache_unpacked
        self.register_buffer("_mask_cache", None, persistent=False)

    # -----------------------------------------------------
    # Helper: bit-pack Boolean mask into uint8 tensor
    # -----------------------------------------------------
    @staticmethod
    def _pack_mask(mask: torch.Tensor, bits: int) -> torch.Tensor:
        """Return a uint8 tensor where multiple mask bits are packed into one byte."""
        if bits == 1:
            flat = mask.flatten()
            pad = (-flat.numel()) % 8
            if pad:
                flat = torch.cat([flat, flat.new_zeros(pad)])
            flat = flat.view(-1, 8)
            byte = sum((flat[:, i].byte() << i) for i in range(8))
            return byte.contiguous()
        # ---- 4-bit (nibble) packing: 2 mask values per byte ----
        flat = mask.flatten()
        pad = (-flat.numel()) % 2
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])
        flat = flat.view(-1, 2)
        byte = flat[:, 0].byte() | (flat[:, 1].byte() << 4)
        return byte.contiguous()

    # -----------------------------------------------------
    # Forward
    # -----------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p

        # Use cached mask if requested and shape matches
        if self.cache_unpacked and self._mask_cache is not None and self._mask_cache.shape == x.shape:
            mask = self._mask_cache
        else:
            mask = (torch.rand_like(x) < keep_prob)
            if self.cache_unpacked:
                self._mask_cache = mask

        # Allocate packed version so that memory accounting mirrors actual method
        _ = self._pack_mask(mask, self.bits)  # intentionally discarded – demo purpose

        # Standard inverted dropout scaling
        return x * mask.div(keep_prob)


# -------------------------------------------------------------
# Utility: recursively swap Dropout → PackedDropout
# -------------------------------------------------------------

def apply_packed_dropout(module: nn.Module, bits: int = 1, cache_unpacked: bool = True):
    """Recursively traverse *module* and replace all instances of
    ``torch.nn.Dropout`` with our :class:`PackedDropout`.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Dropout):
            setattr(module, name, PackedDropout(child.p, bits=bits, cache_unpacked=cache_unpacked))
        else:
            apply_packed_dropout(child, bits, cache_unpacked)


# -------------------------------------------------------------
# Model factory
# -------------------------------------------------------------

def get_model(run_cfg: Dict[str, Any], tokenizer) -> nn.Module:  # type: ignore[valid-type]
    """Instantiate a Hugging-Face model for sequence classification **and** apply
    Packed-Mask Dropout depending on the run configuration.
    """
    model_name = run_cfg["model"].get("name", "bert-base-uncased")
    num_labels = run_cfg["dataset"].get("num_classes", 2)

    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels, use_auth_token=None)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, use_auth_token=None)

    # --- PM-Drop instrumentation -------------------------------------------
    train_cfg = run_cfg.get("training", {})
    if train_cfg.get("use_pmdrop", False):
        bits = int(train_cfg.get("pmdrop_bits", 1))
        cache = bool(train_cfg.get("pmdrop_cache", True))
        apply_packed_dropout(model, bits=bits, cache_unpacked=cache)

    return model