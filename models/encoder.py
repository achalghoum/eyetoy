from abc import ABC
from typing import Generic, Type, List, Tuple, Dict, Any

import torch
from torch import nn
from torch.nn import Module, Conv1d, Conv2d, Conv3d, LayerNorm
from torchvision.models.convnext import LayerNorm2d

from generics import MSNATTransformerType, TransformerStackType
from layers.norms import LayerNorm3d
from layers.embeddings import PositionalEmbedding2D, get_positional_embedding
from layers.transformer import (
    MSNATTransformer1D,
    MSNATTransformer2D,
    MSNATTransformer3D,
    GlobalAttentionTransformer,
)
from params import ConvParams, GlobalAttentionParams
from params import DEFAULT_IMG_ENCODER_PARAMS, TransformerParams


class TransformerStack(Module, Generic[MSNATTransformerType]):
    transformer_type: Type[MSNATTransformerType]
    norm_type: Type[Module]

    def __init__(self, transformer_params: List[TransformerParams]):
        super().__init__()
        self.transformers = torch.nn.Sequential(
            *[
                self.transformer_type(**transformer_param.__dict__)
                for transformer_param in transformer_params
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transformers(x)

        return x


class TransformerStack1D(TransformerStack[MSNATTransformer1D]):
    transformer_type = MSNATTransformer1D


class TransformerStack2D(TransformerStack[MSNATTransformer2D]):
    transformer_type = MSNATTransformer2D
    norm_type = LayerNorm2d
    conv_type = Conv2d


class TransformerStack3D(TransformerStack[MSNATTransformer3D]):
    transformer_type = MSNATTransformer3D
    norm_type = LayerNorm3d
    conv_type = Conv3d


class GlobalAttentionStack(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        num_register_tokens: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.transformer_blocks = nn.ModuleList(
            [
                GlobalAttentionTransformer(
                    d_model,
                    num_heads,
                    num_register_tokens,
                    dropout,
                    use_input_context_token=(i != 0),
                    use_input_register_tokens=(i != 0),
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, context_token, register_tokens = self.transformer_blocks[0](x)
        for transformer_block in self.transformer_blocks[1:]:
            x, context_token, register_tokens = transformer_block(
                x, context_token, register_tokens
            )
        return x, context_token


class Encoder(ABC, Module, Generic[TransformerStackType]):
    transformer_stack_type: Type[TransformerStackType]
    conv_type: Type
    positional_embedding_type: Type[Module]
    dim: int

    def __init__(
        self,
        transformer_params: List[TransformerParams],
        global_attention_params: GlobalAttentionParams,
        initial_conv_params: ConvParams,
        **kwargs
    ):
        super(Encoder, self).__init__()
        self.initial_proj = self.conv_type(**initial_conv_params.__dict__)
        self.transformer_stack = self.transformer_stack_type(transformer_params)
        self.d_model = transformer_params[-1].out_channels

        # Create positional embedding layer
        self.positional_embedding = get_positional_embedding(self.dim, self.d_model)

        self.global_attention = GlobalAttentionStack(**global_attention_params.__dict__)
        self._init_weights()

        # Embedding scale factor
        self.pe_scale = 0.1

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.initial_proj(x)

        # Add positional embeddings
        pe = self.positional_embedding(x)
        x = x + pe * self.pe_scale

        transformed_input = self.transformer_stack(x)
        output, context_token = self.global_attention(transformed_input)
        return output, context_token


class Encoder1D(Encoder[TransformerStack1D]):
    transformer_stack_type = TransformerStack1D
    conv_type = Conv1d
    dim = 1


class Encoder2D(Encoder[TransformerStack2D]):
    transformer_stack_type = TransformerStack2D
    conv_type = Conv2d
    dim = 2


class Encoder3D(Encoder[TransformerStack3D]):
    transformer_stack_type = TransformerStack3D
    conv_type = Conv3d
    dim = 3


class SimpleEncoder2D(Module):
    def __init__(
        self,
        transformer_params: List[TransformerParams],
        initial_conv_params: ConvParams,
        **kwargs
    ):
        super(SimpleEncoder2D, self).__init__()
        self.initial_proj = Conv2d(**initial_conv_params.__dict__)
        self.transformer_stack = TransformerStack2D(transformer_params)
        self.d_model = transformer_params[-1].out_channels

        # Add positional embedding
        self.positional_embedding = PositionalEmbedding2D(self.d_model)
        self.pe_scale = 0.1

        self.layer_norm = LayerNorm2d(self.d_model)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.initial_proj(x)

        # Add positional embeddings
        pe = self.positional_embedding(x)
        x = x + pe * self.pe_scale

        x = self.layer_norm(self.transformer_stack(x))
        context_token = self.global_avg_pool(x)
        context_token = context_token.view(context_token.size(0), -1)
        return x, context_token


DEFAULT_2D_ENCODER = SimpleEncoder2D(**DEFAULT_IMG_ENCODER_PARAMS.__dict__)
