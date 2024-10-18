from layers.positional_encoding import positional_encoding
from params import ConvParams, TransformerParams
from generics import TransformerStackType
from typing import List, Tuple, Generic, Type
from torch import conv1d, conv2d, layer_norm, nn
from abc import ABC
from typing import Generic, Type, List, Tuple

import torch
from torch.nn import Module, Conv1d, Conv2d, Conv3d

from generics import ConvNATTransformerType, TransformerStackType
from params import DEFAULT_IMG_ENCODER_PARAMS, TransformerParams
from layers.transformer import ConvNATTransformer1D, ConvNATTransformer2D, ConvNATTransformer3D, \
    GlobalAttentionBlock


class TransformerStack(Module, Generic[ConvNATTransformerType]):
    transformer_type: Type[ConvNATTransformerType]

    def __init__(self, transformer_params: List[TransformerParams]):
        super().__init__()
        self.transformers = torch.nn.ModuleList([
            self.transformer_type(**transformer_param.__dict__) for transformer_param, index in
            zip(transformer_params, range(len(transformer_params)))
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
                 num_register_tokens: int = 4):
        super().__init__()
        self.d_model = d_model
        self.attention_blocks = torch.nn.ModuleList([
            GlobalAttentionBlock(d_model, num_heads, num_register_tokens, dropout,
                                 use_input_context_token=(i != 0), use_input_register_tokens=(i != 0))
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[
            torch.Tensor, torch.Tensor]:
        output, context_token, registers = self.attention_blocks[0](x)
        for i in range(1, len(self.attention_blocks)):
            output, context_token, registers = self.attention_blocks[i](
                output, context_token, registers)
        return output, context_token


class Encoder(ABC, Module, Generic[TransformerStackType]):
    transformer_stack_type: Type[TransformerStackType]
    conv_type: Type

    def __init__(self, transformer_params: List[TransformerParams], num_global_attention_heads: int,
                 global_attention_dropout: float, num_global_attention_layers: int, initial_conv_params: ConvParams):
        super(Encoder, self).__init__()
        self.norm = nn.GroupNorm(num_channels=initial_conv_params.out_channels,num_groups=32)
        self.initial_conv = self.conv_type(**initial_conv_params.__dict__)
        self.transformer_stack = self.transformer_stack_type(
            transformer_params)
        self.global_attention = GlobalAttentionStack(
            transformer_params[-1].out_channels,
            num_global_attention_heads, num_global_attention_layers,
            global_attention_dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.initial_conv(x)
        x= self.norm(x)
        x = x + positional_encoding(x, d=x.shape[1]) * 0.1
        transformed_input = self.transformer_stack(x)
        output, context_token = self.global_attention(transformed_input)
        return output, context_token


class Encoder1D(Encoder[TransformerStack1D]):
    transformer_stack_type = TransformerStack1D
    conv_type = Conv1d


class Encoder2D(Encoder[TransformerStack2D]):
    transformer_stack_type = TransformerStack2D
    conv_type = Conv2d


class Encoder3D(Encoder[TransformerStack3D]):
    transformer_stack_type = TransformerStack3D
    conv_type = Conv3d


class BiLSTMBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.1):
        super(BiLSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.layernorm = nn.LayerNorm(input_size*num_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Flatten spatial dimensions if working with 2D/3D data
        x_flat = x.flatten(2).transpose(1, 2)  # (B, L, C)

        output, (h_n, _) = self.lstm(x_flat)

        # Use the final hidden state as the "context"
        context = h_n.transpose(0, 1).contiguous().view(x.size(0), -1)
        context = self.layernorm(context)

        return x, context


class SimplifiedEncoder(ABC, nn.Module, Generic[TransformerStackType]):
    transformer_stack_type: Type[TransformerStackType]
    conv_type: Type

    def __init__(self, transformer_params: List[TransformerParams],
                 bilstm_hidden_size: int, bilstm_num_layers: int,
                 bilstm_dropout: float, initial_conv_params: ConvParams):
        super(SimplifiedEncoder, self).__init__()
        self.initial_conv = nn.Sequential(self.conv_type(**initial_conv_params.__dict__),
                                          nn.ReLU())

        self.transformer_stack = self.transformer_stack_type(
            transformer_params)
        self.bilstm = BiLSTMBlock(transformer_params[-1].out_channels,
                                  bilstm_hidden_size, bilstm_num_layers,
                                  bilstm_dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.initial_conv(x)
        x = x + positional_encoding(x, d=x.shape[1])
        
        transformed_input = self.transformer_stack(x)
        output, context = self.bilstm(transformed_input)
        return output, context


class SimplifiedEncoder1D(SimplifiedEncoder[TransformerStack1D]):
    conv_type = Conv1d
    transformer_stack_type = TransformerStack1D


class SimplifiedEncoder2D(SimplifiedEncoder[TransformerStack2D]):
    conv_type = Conv2d
    transformer_stack_type = TransformerStack2D


class SimplifiedEncoder3D(SimplifiedEncoder[TransformerStack3D]):
    conv_type = Conv2d
    transformer_stack_type = TransformerStack3D


# You can define default parameters for the simplified encoder if needed
DEFAULT_SIMPLIFIED_ENCODER_PARAMS = {
    "transformer_params": DEFAULT_IMG_ENCODER_PARAMS.transformer_params,
    "initial_conv_params": DEFAULT_IMG_ENCODER_PARAMS.initial_conv_params,
    "bilstm_hidden_size": 256,
    "bilstm_num_layers": 4,
    "bilstm_dropout": 0.1
}

DEFAULT_SIMPLIFIED_2D_ENCODER = SimplifiedEncoder2D(
    **DEFAULT_SIMPLIFIED_ENCODER_PARAMS)
DEFAULT_2D_ENCODER = Encoder2D(**DEFAULT_IMG_ENCODER_PARAMS.__dict__)
