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
    # Number of dimensions (2D or 3D)
    num_dims = len(positions)

    # Create a tensor for positional encodings
    if num_dims == 2:  # For 2D
        height, width = positions[0].shape[1], positions[1].shape[1]
        pe = torch.zeros(positions[0].shape[0], height, width, d, device=positions[0].device)

        for i in range(d // 2):
            # Compute sine and cosine for the first dimension (x)
            pe[..., 2 * i] = torch.sin(positions[0] / (10000 ** (2 * i / d)))
            pe[..., 2 * i + 1] = torch.cos(positions[0] / (10000 ** (2 * i / d)))

            # Compute sine and cosine for the second dimension (y)
            pe[..., 2 * (i + d // 2)] = torch.sin(positions[1] / (10000 ** (2 * i / d)))
            pe[..., 2 * (i + d // 2) + 1] = torch.cos(positions[1] / (10000 ** (2 * i / d)))

        return pe

    elif num_dims == 3:  # For 3D
        depth, height, width = positions[0].shape[1], positions[1].shape[1], positions[2].shape[1]
        pe = torch.zeros(positions[0].shape[0], depth, height, width, d, device=positions[0].device)

        for i in range(d // 3):
            # Compute sine and cosine for the first dimension (x)
            pe[..., 3 * i] = torch.sin(positions[0] / (10000 ** (2 * i / d)))
            pe[..., 3 * i + 1] = torch.cos(positions[0] / (10000 ** (2 * i / d)))

            # Compute sine and cosine for the second dimension (y)
            pe[..., 3 * (i + d // 3)] = torch.sin(positions[1] / (10000 ** (2 * i / d)))
            pe[..., 3 * (i + d // 3) + 1] = torch.cos(positions[1] / (10000 ** (2 * i / d)))

            # Compute sine and cosine for the third dimension (z)
            pe[..., 3 * (i + 2 * d // 3)] = torch.sin(positions[2] / (10000 ** (2 * i / d)))
            pe[..., 3 * (i + 2 * d // 3) + 1] = torch.cos(positions[2] / (10000 ** (2 * i / d)))

        return pe
    else:
        raise ValueError("Input tensor must be 2D or 3D.")


def positional_encoding(input_tensors, d = 8):
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
            y_position = torch.arange(height, device=input_tensor.device).unsqueeze(0).repeat(
                batch_size, 1, 1)  # Shape: (batch_size, height, width)
            x_position = torch.arange(width, device=input_tensor.device).unsqueeze(1).repeat(
                batch_size, 1, 1)  # Shape: (batch_size, height, width)

            # Add batch dimension
            x_position = x_position.unsqueeze(1)  # Shape: (batch_size, 1, width)
            y_position = y_position.unsqueeze(1)  # Shape: (batch_size, 1, height)

            pe = compute_positional_encoding((x_position, y_position), d)
            positional_encodings.append(pe)

        elif input_tensor.ndim == 5:  # 3D Tensor: (batch_size, depth, height, width, channels)
            depth, height, width = input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[
                3]

            # Create grids of coordinates for 3D
            z_position = torch.arange(depth, device=input_tensor.device).unsqueeze(0).unsqueeze(
                2).unsqueeze(3).repeat(batch_size, 1, height,
                                       width)  # Shape: (batch_size, depth, height, width)
            y_position = torch.arange(height, device=input_tensor.device).unsqueeze(0).unsqueeze(
                2).repeat(batch_size, depth, 1, width)  # Shape: (batch_size, depth, height, width)
            x_position = torch.arange(width, device=input_tensor.device).unsqueeze(0).repeat(
                batch_size, depth, height, 1)  # Shape: (batch_size, depth, height, width)

            pe = compute_positional_encoding((x_position, y_position, z_position), d)
            positional_encodings.append(pe)

        else:
            raise ValueError(
                "Input tensor must be 2D or 3D (shape: (batch_size, height, width, channels) or (batch_size, depth, height, width, channels)).")

    return positional_encodings