import torch


def compute_positional_encoding(positions, d=8):
    """
    Compute the positional encodings based on the input positions.

    Args:
        positions (tuple): A tuple containing two or three tensors representing the coordinates.
        d (int): Dimensionality of the positional encoding (should be even for 2D and a multiple of 3 for 3D).

    Returns:
        torch.Tensor: Positional encodings.
    """
    # Number of spatial dimensions (2D or 3D)
    num_dims = len(positions)

    # Validate the dimensionality
    if num_dims not in [2, 3]:
        raise ValueError("Only 2D or 3D positional encoding is supported.")

    # Ensure d is divisible by the number of spatial dimensions
    if d % num_dims != 0:
        raise ValueError(f"For {num_dims}D encoding, 'd' must be divisible by {num_dims}.")

    # Calculate encoding dimensions per spatial axis
    d_per_dim = d // num_dims

    # Initialize the positional encoding tensor based on the input dimensions
    if num_dims == 2:  # 2D Encoding
        batch_size, height, width = positions[0].shape[0], positions[0].shape[1], positions[1].shape[2]
        pe = torch.zeros(batch_size, height, width, d, device=positions[0].device)

        # Iterate over each spatial dimension (x and y)
        for dim_idx, pos in enumerate(positions):
            for i in range(d_per_dim // 2):
                div_term = 10000 ** (2 * i / d)
                pe[..., dim_idx * d_per_dim + 2 * i] = torch.sin(pos / div_term)
                pe[..., dim_idx * d_per_dim + 2 * i + 1] = torch.cos(pos / div_term)

    elif num_dims == 3:  # 3D Encoding
        batch_size, depth, height, width = positions[0].shape[0], positions[0].shape[1], positions[1].shape[2], positions[2].shape[3]
        pe = torch.zeros(batch_size, depth, height, width, d, device=positions[0].device)

        # Iterate over each spatial dimension (x, y, z)
        for dim_idx, pos in enumerate(positions):
            for i in range(d_per_dim // 2):
                div_term = 10000 ** (2 * i / d)
                pe[..., dim_idx * d_per_dim + 2 * i] = torch.sin(pos / div_term)
                pe[..., dim_idx * d_per_dim + 2 * i + 1] = torch.cos(pos / div_term)

    return pe


def positional_encoding(input_tensors, d=8):
    """
    Generate positional encodings for a batch of 2D or 3D input tensors of different sizes.

    Args:
        input_tensors (list): A list of input tensors, each of shape (batch_size, height, width, channels)
                              for 2D or (batch_size, depth, height, width, channels) for 3D.
        d (int): Dimensionality of the positional encoding.

    Returns:
        list: A list of positional encodings, each corresponding to an input tensor in the batch.
    """
    positional_encodings = []

    for input_tensor in input_tensors:
        batch_size = input_tensor.shape[0]

        if input_tensor.ndim == 4:  # 2D Tensor: (batch_size, height, width, channels)
            height, width = input_tensor.shape[1], input_tensor.shape[2]

            # Create a grid of coordinates for 2D
            y_position = torch.arange(height, device=input_tensor.device).unsqueeze(1).repeat(1, width)  # Shape: (height, width)
            x_position = torch.arange(width, device=input_tensor.device).unsqueeze(0).repeat(height, 1)  # Shape: (height, width)

            # Expand to batch size
            x_position = x_position.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: (batch_size, height, width)
            y_position = y_position.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: (batch_size, height, width)

            pe = compute_positional_encoding((x_position, y_position), d)
            positional_encodings.append(pe)

        elif input_tensor.ndim == 5:  # 3D Tensor: (batch_size, depth, height, width, channels)
            depth, height, width = input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]

            # Create grids of coordinates for 3D
            z_position = torch.arange(depth, device=input_tensor.device).view(1, depth, 1, 1).repeat(batch_size, 1, height, width)  # Shape: (batch_size, depth, height, width)
            y_position = torch.arange(height, device=input_tensor.device).view(1, 1, height, 1).repeat(batch_size, depth, 1, width)  # Shape: (batch_size, depth, height, width)
            x_position = torch.arange(width, device=input_tensor.device).view(1, 1, 1, width).repeat(batch_size, depth, height, 1)  # Shape: (batch_size, depth, height, width)

            pe = compute_positional_encoding((x_position, y_position, z_position), d)
            positional_encodings.append(pe)

        else:
            raise ValueError(
                "Input tensor must be 2D or 3D (shape: (batch_size, height, width, channels) or (batch_size, depth, height, width, channels)).")

    return positional_encodings