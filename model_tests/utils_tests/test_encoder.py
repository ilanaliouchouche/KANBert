import unittest
import torch
from model.utils import KANBertConfig
from model.utils.encoder import Encoder


class TestEncoder(unittest.TestCase):
    """
    Unit tests for the Encoder class in the KANBert model.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the configuration and the Encoder model.
        """

        cls.config = KANBertConfig(
            vocabulary_size=30522,
            hidden_dim=768,
            max_sequence_len=512,
            n_layers=12,
            intermediate_dim=3072,
            num_attention_heads=12,
            periodicity=10000
        )
        cls.encoder = Encoder(cls.config)

    def setUp(self) -> None:
        """
        Set up the inputs for the Encoder model.
        """

        self.batch_size = 2
        self.seq_length = 10
        self.hidden_dim = self.config.hidden_dim
        self.input_tensor = torch.randn(self.batch_size,
                                        self.seq_length,
                                        self.hidden_dim)
        self.input_attn_mask = torch.ones(self.batch_size,
                                          self.seq_length,
                                          dtype=torch.bool)
        self.device = "cpu"
        self.encoder.to(self.device)
        self.input_tensor = self.input_tensor.to(self.device)

    def test_forward_shape(self) -> None:
        """
        Test the shape of the output of the Encoder model.
        """

        output = self.encoder(self.input_tensor, self.input_attn_mask)
        expected_shape = (self.batch_size,
                          self.seq_length,
                          self.hidden_dim)
        self.assertEqual(output.shape,
                         expected_shape,
                         f"Expected shape {expected_shape}, "
                         f"but got {output.shape}")

    def test_forward_type(self) -> None:
        """
        Test the type of the output of the Encoder model.
        """

        output = self.encoder(self.input_tensor, self.input_attn_mask)
        self.assertIsInstance(output,
                              torch.Tensor,
                              "Output should be a torch.Tensor, "
                              f"but got {type(output)}")

    def test_device(self) -> None:
        """
        Test the device of the output of the Encoder model.
        """

        output = self.encoder(self.input_tensor, self.input_attn_mask)
        self.assertEqual(output.device.type,
                         self.device,
                         f"Expected device {self.device}, "
                         f"but got {output.device}")


if __name__ == "__main__":
    unittest.main()
