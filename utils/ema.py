"""Exponential moving average(EAM) Code.

* Reference: https://www.zijianhu.com/post/pytorch/ema/
"""

import torch
from torch import nn
from copy import deepcopy
from collections import OrderedDict


class EMA(nn.Module):
    """EAM."""

    def __init__(self, model: nn.Module, decay: float, device: torch.device):
        """Get a model and decay parameter."""
        super().__init__()
        self.decay = decay
        self.model = model

        self.shadow = deepcopy(self.model)
        self.shadow.eval()
        self.shadow.to(device)
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self):
        """Update."""
        if not self.training:
            print("EMA update should only be called during training")
            return

        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())

        # check if both model contains the same set of keys
        assert model_params.keys() == shadow_params.keys()

        for name, param in model_params.items():
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            shadow_params[name].sub_(
                (1. - self.decay) * (shadow_params[name] - param)
            )

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            shadow_buffers[name].copy_(buffer)
