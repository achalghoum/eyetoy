import unittest
import torch
from positional_encoding import compute_positional_encoding, positional_encoding

class TestPositionalEncoding(unittest.TestCase):
    def test_compute_positional_encoding_2d(self):
        batch_size = 2
        height, width = 4, 4
        d = 8
        x = torch.arange(width).unsqueeze(0).repeat(batch_size, height, 1).float()
        y = torch.arange(height).unsqueeze(1).repeat(batch_size, 1, width).float()
        positions = (x, y)

        pe = compute_positional_encoding(positions, d)

        # Check the shape
        expected_shape = (batch_size, height, width, d)
        self.assertEqual(pe.shape, expected_shape)

        # Check that sine and cosine have been computed
        self.assertFalse(torch.isnan(pe).any())

    def test_compute_positional_encoding_3d(self):
        batch_size = 2
        depth, height, width = 3, 4, 4
        d = 9  # Must be a multiple of 3 for 3D
        x = torch.arange(width).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(batch_size, depth, height, 1).float()
        y = torch.arange(height).unsqueeze(0).unsqueeze(1).unsqueeze(3).repeat(batch_size, depth, 1, width).float()
        z = torch.arange(depth).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(batch_size, 1, height, width).float()
        positions = (x, y, z)

        pe = compute_positional_encoding(positions, d)

        # Check the shape
        expected_shape = (batch_size, depth, height, width, d)
        self.assertEqual(pe.shape, expected_shape)

        # Check that sine and cosine have been computed
        self.assertFalse(torch.isnan(pe).any())

    def test_positional_encoding_2d_batch(self):
        batch_size = 2
        height, width, channels = 4, 4, 3
        d = 8
        input_tensors = [torch.randn(batch_size, height, width, channels) for _ in range(2)]

        positional_encodings = positional_encoding(input_tensors, d)

        # Check the number of positional encodings
        self.assertEqual(len(positional_encodings), 2)

        for pe in positional_encodings:
            expected_shape = (batch_size, height, width, d)
            self.assertEqual(pe.shape, expected_shape)
            self.assertFalse(torch.isnan(pe).any())

    def test_positional_encoding_3d_batch(self):
        batch_size = 2
        depth, height, width, channels = 3, 4, 4, 3
        d = 9
        input_tensors = [torch.randn(batch_size, depth, height, width, channels) for _ in range(2)]

        positional_encodings = positional_encoding(input_tensors, d)

        # Check the number of positional encodings
        self.assertEqual(len(positional_encodings), 2)

        for pe in positional_encodings:
            expected_shape = (batch_size, depth, height, width, d)
            self.assertEqual(pe.shape, expected_shape)
            self.assertFalse(torch.isnan(pe).any())

    def test_invalid_input_dim(self):
        # Test with invalid number of dimensions (neither 4 nor 5)
        input_tensor = torch.randn(2, 4, 4)  # 3D tensor instead of 4D or 5D
        with self.assertRaises(ValueError):
            positional_encoding([input_tensor], d=8)

    def test_invalid_positional_encoding_dim_2d(self):
        # Test with invalid encoding dimension for 2D (d should be even)
        batch_size = 2
        height, width = 4, 4
        d = 7  # Not even
        x = torch.arange(width).unsqueeze(0).repeat(batch_size, height, 1).float()
        y = torch.arange(height).unsqueeze(1).repeat(batch_size, 1, width).float()
        positions = (x, y)

        with self.assertRaises(ValueError):
            compute_positional_encoding(positions, d)

    def test_invalid_positional_encoding_dim_3d(self):
        # Test with invalid encoding dimension for 3D (d should be multiple of 3)
        batch_size = 2
        depth, height, width = 3, 4, 4
        d = 10  # Not a multiple of 3
        x = torch.arange(width).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(batch_size, depth, height, 1).float()
        y = torch.arange(height).unsqueeze(0).unsqueeze(1).unsqueeze(3).repeat(batch_size, depth, 1, width).float()
        z = torch.arange(depth).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(batch_size, 1, height, width).float()
        positions = (x, y, z)

        with self.assertRaises(ValueError):
            compute_positional_encoding(positions, d)

if __name__ == '__main__':
    unittest.main()