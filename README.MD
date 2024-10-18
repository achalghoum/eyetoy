#EyeToy

# Convolutional Neighborhood Attention (ConvNAT)

## Overview

Convolutional Neighborhood Attention (ConvNAT) is a novel approach that integrates convolutional neural networks (CNNs) with attention mechanisms to enhance the performance of various tasks in computer vision and natural language processing. This project implements ConvNAT in a modular way, allowing for easy integration and experimentation with different configurations.

## Key Components

1. **Attention Mechanism**: The core of ConvNAT is its attention mechanism, which allows the model to focus on relevant parts of the input data while processing it. This is achieved through the use of convolutional layers that operate on local neighborhoods.

2. **Convolutional Layers**: The model utilizes various types of convolutional layers (1D, 2D, and 3D) to process different types of data, such as time series, images, and volumetric data.

3. **Transformer Architecture**: ConvNAT is designed to work within a transformer architecture, leveraging multi-head attention and residual connections to improve learning and representation capabilities.

4. **Global Attention Block**: This component allows the model to incorporate global context into the attention mechanism, enhancing its ability to capture long-range dependencies in the data.

5. **Simplified Encoder**: A simplified version of the encoder is provided for ease of use, allowing users to quickly set up and experiment with ConvNAT without delving into the complexities of the full architecture.

### Custom Configuration

You can customize the parameters of the ConvNAT model by modifying the `DEFAULT_IMG_ENCODER_PARAMS` or by creating your own parameter dictionary.

## Architecture

The architecture consists of several key classes:

- **ConvNAT**: The main class implementing the convolutional neighborhood attention mechanism.
- **ConvNAT1D, ConvNAT2D, ConvNAT3D**: Specialized classes for handling 1D, 2D, and 3D data respectively.
- **ConvMultiHeadNAT**: Implements multi-head attention using the ConvNAT mechanism.
- **Encoder**: The main encoder class that integrates the ConvNAT and transformer components.
- **GlobalAttentionBlock**: A block that incorporates global attention into the model.
