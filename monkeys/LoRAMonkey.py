import torch
import torch.nn as nn
from copy import deepcopy

from monkeys.TransformerMonkey import TransformerMonkey


class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=4, alpha=1):
        super().__init__()

        #freeze old weights
        self.original_layer = original_layer
        self.original_layer.weight.requires_grad = False
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros((in_features, rank)))
        self.lora_B = nn.Parameter(torch.zeros((rank, out_features)))
        self.scaling = alpha / rank
        
        # normal for layer A, 0 for layer B is standard, as described in:
        #https://apxml.com/courses/lora-peft-efficient-llm-training/chapter-4-advanced-lora-variants/lora-initialization-strategies 
        nn.init.normal_(self.lora_A) 
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Original frozen path + LoRA bypass path
        return self.original_layer(x) + (x @ self.lora_A @ self.lora_B) * self.scaling
    
    @property
    def weight(self):
        return self.original_layer.weight

    @property
    def bias(self):
        return self.original_layer.bias

class LoRAMonkey(TransformerMonkey):
    def __init__(self, base_monkey:TransformerMonkey, rank=8, alpha=2):
            # calling nn.module.__init__(), the shared parent of TransformerMonkey and Self
            super(TransformerMonkey, self).__init__() 
            
            # copy the base_monkey
            self.__dict__.update(deepcopy(base_monkey.__dict__))
            
            # freeze original parameters
            for param in self.parameters():
                param.requires_grad = False
                
            # add LoRA to the NN components of the transformer blocks
            for block in self.blocks.layers:
                block.linear1 = LoRALinear(block.linear1, rank, alpha)
                block.linear2 = LoRALinear(block.linear2, rank, alpha)
                
                block.self_attn.out_proj = LoRALinear(block.self_attn.out_proj, rank, alpha)

            # add LoRA to the lm_head, lowkey just for funsies 
            self.lm_head = LoRALinear(self.lm_head, rank, alpha)



    