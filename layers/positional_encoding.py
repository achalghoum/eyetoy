import torch
import math
from torch.nn.utils.rnn import pad_sequence


def compute_positional_encoding(positions, d=8):
    """
    Compute the positional encodings based on the input positions.

    Args:
        positions (tuple): A tuple containing one, two or three tensors representing the coordinates.
        d (int): Dimensionality of the positional encoding (should be even for 1D and 2D, and a multiple of 3 for 3D).

    Returns:
        torch.Tensor: Positional encodings.
    """
    # Number of spatial dimensions (1D, 2D or 3D)
    num_dims = len(positions)

    # Validate the dimensionality
    if num_dims not in [1, 2, 3]:
        raise ValueError("Only 1D, 2D or 3D positional encoding is supported.")

    # Ensure d is divisible by the number of spatial dimensions
    if d % num_dims != 0:
        raise ValueError(f"For {num_dims}D encoding, 'd' must be divisible by {num_dims}.")

    # Calculate encoding dimensions per spatial axis
    d_per_dim = d // num_dims

    # Initialize the positional encoding tensor based on the input dimensions
    if num_dims == 1:  # 1D Encoding
        batch_size, length = positions[0].shape
        pe = torch.zeros(batch_size, length, d, device=positions[0].device)

        # Iterate over the spatial dimension
        pos = positions[0].unsqueeze(-1)  # Shape: (batch_size, length, 1)
        div_term = torch.exp(torch.arange(0, d, 2, device=positions[0].device) * -(math.log(10000.0) / d))
        div_term = div_term.view(1, 1, -1)  # Shape: (1, 1, d//2)
        
        pe[:, :, 0::2] = torch.sin(pos * div_term)
        pe[:, :, 1::2] = torch.cos(pos * div_term)

    elif num_dims == 2:  # 2D Encoding
        batch_size, height, width = positions[0].shape
        pe = torch.zeros(batch_size, d, height, width, device=positions[0].device)

        # Iterate over each spatial dimension (x and y)
        for dim_idx, pos in enumerate(positions):
            for i in range(d_per_dim // 2):
                div_term = 10000 ** (2 * i / d)
                pe[:, dim_idx * d_per_dim + 2 * i, :, :] = torch.sin(pos / div_term)
                pe[:, dim_idx * d_per_dim + 2 * i + 1, :, :] = torch.cos(pos / div_term)

    elif num_dims == 3:  # 3D Encoding
        batch_size, depth, height, width = positions[0].shape[0], positions[0].shape[2], positions[0].shape[3], positions[0].shape[4]
        pe = torch.zeros(batch_size, d, depth, height, width, device=positions[0].device)

        # Iterate over each spatial dimension (x, y, z)
        for dim_idx, pos in enumerate(positions):
            for i in range(d_per_dim // 2):
                div_term = 10000 ** (2 * i / d)
                pe[:, dim_idx * d_per_dim + 2 * i, :, :, :] = torch.sin(pos / div_term)
                pe[:, dim_idx * d_per_dim + 2 * i + 1, :, :, :] = torch.cos(pos / div_term)

    return pe


def positional_encoding(input_tensors, d=8):
    """
    Generate positional encodings for a batch of 1D, 2D or 3D input tensors of different sizes.

    Args:
        input_tensors (torch.Tensor): Input tensor of shape (batch_size, channels, length) for 1D,
                                      (batch_size, channels, height, width) for 2D, or
                                      (batch_size, channels, depth, height, width) for 3D.
        d (int): Dimensionality of the positional encoding.

    Returns:
        torch.Tensor: Positional encodings corresponding to the input tensor.
    """
    if input_tensors.is_nested:
        return nested_positional_encoding(input_tensors, d)

    batch_size = input_tensors.shape[0]

    if input_tensors.ndim == 3:  # 1D Tensor: (batch_size, channels, length)
        length = input_tensors.shape[1]

        # Create a sequence of positions for 1D
        position = torch.arange(length, device=input_tensors.device).unsqueeze(0).repeat(batch_size, 1)  # Shape: (batch_size, length)

        pe = compute_positional_encoding((position,), d)

    elif input_tensors.ndim == 4:  # 2D Tensor: (batch_size, channels, height, width)
        height, width = input_tensors.shape[2], input_tensors.shape[3]

        # Create a grid of coordinates for 2D
        y_position = torch.arange(height, device=input_tensors.device).unsqueeze(1).repeat(1, width)  # Shape: (height, width)
        x_position = torch.arange(width, device=input_tensors.device).unsqueeze(0).repeat(height, 1)  # Shape: (height, width)

        # Expand to batch size
        x_position = x_position.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: (batch_size, height, width)
        y_position = y_position.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: (batch_size, height, width)

        pe = compute_positional_encoding((x_position, y_position), d)

    elif input_tensors.ndim == 5:  # 3D Tensor: (batch_size, channels, depth, height, width)
        depth, height, width = input_tensors.shape[2], input_tensors.shape[3], input_tensors.shape[4]

        # Create grids of coordinates for 3D
        z_position = torch.arange(depth, device=input_tensors.device).view(1, depth, 1, 1).repeat(batch_size, 1, height, width)  # Shape: (batch_size, depth, height, width)
        y_position = torch.arange(height, device=input_tensors.device).view(1, 1, height, 1).repeat(batch_size, depth, 1, width)  # Shape: (batch_size, depth, height, width)
        x_position = torch.arange(width, device=input_tensors.device).view(1, 1, 1, width).repeat(batch_size, depth, height, 1)  # Shape: (batch_size, depth, height, width)

        pe = compute_positional_encoding((x_position, y_position, z_position), d)

    else:
        raise ValueError(
            f"Input tensor must be 1D, 2D or 3D (shape: (batch_size, channels, length) or (batch_size, channels, height, width) or (batch_size, channels, depth, height, width)). Instead \
            got shape {input_tensors.shape}")

    return pe


def nested_positional_encoding(nested_tensors, d=8):
    """
    Generate positional encodings for a batch of nested 1D, 2D or 3D input tensors of different sizes.

    Args:
        nested_tensors (list): A list of lists of input tensors, each of shape (channels, length) for 1D,
                               (channels, height, width) for 2D or (channels, depth, height, width) for 3D.
        d (int): Dimensionality of the positional encoding.

    Returns:
        list: A list of lists of positional encodings, each corresponding to an input tensor in the nested batch.
    """
    nested_positional_encodings = []

    for batch in nested_tensors:
        batch_encodings = []
        for tensor in batch:
            if tensor.ndim == 2:  # 1D Tensor: (channels, length)
                length = tensor.shape[1]

                # Create a sequence of positions for 1D
                position = torch.arange(length, device=tensor.device).unsqueeze(0)  # Shape: (1, length)

                pe = compute_positional_encoding((position,), d)
                batch_encodings.append(pe.squeeze(0))

            elif tensor.ndim == 3:  # 2D Tensor: (channels, height, width)
                height, width = tensor.shape[1], tensor.shape[2]

                # Create a grid of coordinates for 2D
                y_position = torch.arange(height, device=tensor.device).unsqueeze(1).repeat(1, width)  # Shape: (height, width)
                x_position = torch.arange(width, device=tensor.device).unsqueeze(0).repeat(height, 1)  # Shape: (height, width)

                pe = compute_positional_encoding((x_position.unsqueeze(0), y_position.unsqueeze(0)), d)
                batch_encodings.append(pe.squeeze(0))

            elif tensor.ndim == 4:  # 3D Tensor: (channels, depth, height, width)
                depth, height, width = tensor.shape[1], tensor.shape[2], tensor.shape[3]

                # Create grids of coordinates for 3D
                z_position = torch.arange(depth, device=tensor.device).view(depth, 1, 1).repeat(1, height, width)  # Shape: (depth, height, width)
                y_position = torch.arange(height, device=tensor.device).view(1, height, 1).repeat(depth, 1, width)  # Shape: (depth, height, width)
                x_position = torch.arange(width, device=tensor.device).view(1, 1, width).repeat(depth, height, 1)  # Shape: (depth, height, width)

                pe = compute_positional_encoding((x_position.unsqueeze(0), y_position.unsqueeze(0), z_position.unsqueeze(0)), d)
                batch_encodings.append(pe.squeeze(0))

            else:
                raise ValueError(
                    "Input tensor must be 1D, 2D or 3D (shape: (channels, length) or (channels, height, width) or (channels, depth, height, width)).")

        nested_positional_encodings.append(batch_encodings)

    return nested_positional_encodings
