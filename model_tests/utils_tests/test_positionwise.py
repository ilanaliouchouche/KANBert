import unittest
import torch
from model.utils import KANBertConfig
from model.utils.positionwise import PositionWiseFeedForward


class TestPositionWiseFeedForward(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        cls.config = KANBertConfig(
            vocabulary_size=30522,
            hidden_dim=768,
            max_sequence_len=512,
            n_layers=12,
            intermediate_dim=3072,
            num_attention_heads=12,
            periodicity=10000
        )
        cls.pwff = PositionWiseFeedForward(cls.config)

    def setUp(self) -> None:

        self.batch_size = 2
        self.seq_length = 10
        self.hidden_dim = self.config.hidden_dim
        self.input_tensor = torch.randn(self.batch_size,
                                        self.seq_length,
                                        self.hidden_dim)
        self.device = "cpu"
        self.pwff.to(self.device)
        self.input_tensor = self.input_tensor.to(self.device)

    def test_forward_shape(self) -> None:

        output = self.pwff(self.input_tensor)
        expected_shape = (self.batch_size,
                          self.seq_length,
                          self.hidden_dim)
        self.assertEqual(output.shape,
                         expected_shape,
                         f"Expected shape {expected_shape}, "
                         f"but got {output.shape}")

    def test_forward_type(self) -> None:

        output = self.pwff(self.input_tensor)
        self.assertIsInstance(output,
                              torch.Tensor,
                              "Output should be a torch.Tensor, "
                              f"but got {type(output)}")

    def test_device(self) -> None:

        output = self.pwff(self.input_tensor)
        self.assertEqual(output.device.type,
                         self.device,
                         f"Expected device {self.device}, "
                         f"but got {output.device}")


if __name__ == "__main__":
    unittest.main()
