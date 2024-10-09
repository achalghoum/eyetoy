from abc import ABC
from typing import Generic, Type, List, Tuple

import torch
from torch.nn import Module

from generics import ConvNATTransformerType, TransformerStackType
from params import TransformerParams
from transformer import ConvNATTransformer1D, ConvNATTransformer2D, ConvNATTransformer3D, \
    GlobalAttentionBlock


class TransformerStack(Module, Generic[ConvNATTransformerType]):
    transformer_type: Type[ConvNATTransformerType]

    def __init__(self, transformer_params: List[TransformerParams]):
        super().__init__()
        self.transformers = torch.nn.ModuleList([
            self.transformer_type(**transformer_param.__dict__) for transformer_param in
            transformer_params
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for transformer in self.transformers:
            x = transformer(x)
        return x


class TransformerStack1D(TransformerStack[ConvNATTransformer1D]):
    transformer_type = ConvNATTransformer1D


class TransformerStack2D(TransformerStack[ConvNATTransformer2D]):
    transformer_type = ConvNATTransformer2D


class TransformerStack3D(TransformerStack[ConvNATTransformer3D]):
    transformer_type = ConvNATTransformer3D


class GlobalAttentionStack(Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, dropout: float = 0.1,
                 ):
        super().__init__()
        self.attention_blocks = torch.nn.ModuleList([
            GlobalAttentionBlock(d_model, num_heads, dropout, use_input_context_token=(i != 0))
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        output, context_token = self.attention_blocks[0](x)
        for i in range(1, len(self.attention_blocks)):
            output, context_token = self.attention_blocks[i](output, context_token)
        return output, context_token


class Encoder(ABC, Module, Generic[TransformerStackType]):
    transformer_stack_type: Type[TransformerStackType]

    def __init__(self, transformer_params: List[TransformerParams], num_global_attention_heads: int,
                 global_attention_dropout: float, num_global_attention_layers: int):
        super().__init__()
        self.transformer_stack = self.transformer_stack_type(**transformer_params.__dict__)
        self.global_attention = GlobalAttentionStack(
            transformer_params[-1].out_channels,
            num_global_attention_heads, num_global_attention_layers,
            global_attention_dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        transformed_input = self.transformer_stack(x)
        output, context_token = self.global_attention(transformed_input)
        return output, context_token


class Encoder1D(Encoder[TransformerStack1D]):
    transformer_sequence_type = TransformerStack1D


class Encoder2D(Encoder[TransformerStack2D]):
    transformer_sequence_type = TransformerStack2D


class Encoder3D(Encoder[TransformerStack3D]):
    transformer_sequence_type = TransformerStack3D
