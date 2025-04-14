#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Custom trainer implementation to force CPU usage.
"""

import os
import torch
from transformers import Trainer

class CPUTrainer(Trainer):
    """
    A custom trainer class that forces CPU usage regardless of environment settings.
    """
    
    def __init__(self, *args, **kwargs):
        # We cannot set device directly on args, so we'll handle it in _setup_devices
        super().__init__(*args, **kwargs)
    
    def _setup_devices(self, *args, **kwargs):
        """
        Overridden to force CPU usage
        """
        # Force CPU device
        self._n_gpu = 0
        self.is_model_parallel = False
        self.device = torch.device("cpu")
        self.use_cuda = False
        self.use_mps_device = False
        
        # Move model to CPU if it's already on a device
        if hasattr(self, "model") and self.model is not None:
            self.model = self.model.to(self.device)
            
        return self.device
    
    def _prepare_inputs(self, inputs):
        """
        Ensure all tensors are on CPU
        """
        if not isinstance(inputs, dict):
            return inputs
            
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(torch.device("cpu"))
        
        return inputs
    
    def get_train_dataloader(self):
        """
        Force CPU tensors for train dataloader
        """
        dataloader = super().get_train_dataloader()
        dataloader.pin_memory = False
        return dataloader
    
    def get_eval_dataloader(self, eval_dataset=None):
        """
        Force CPU tensors for eval dataloader
        """
        dataloader = super().get_eval_dataloader(eval_dataset)
        dataloader.pin_memory = False
        return dataloader 